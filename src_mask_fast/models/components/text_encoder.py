"""Segment (token) encoder + duration predictor.

Ported from Matcha-TTS (``matcha/models/components/text_encoder.py``, itself
from https://github.com/jaywalnut310/glow-tts). ``LayerNorm``, ``ConvReluNorm``,
``DurationPredictor``, ``RotaryPositionalEmbeddings``, ``MultiHeadAttention``,
``FFN`` and ``Encoder`` are unchanged.

The one real adaptation is the top-level module. Matcha embeds discrete phoneme
ids with ``nn.Embedding``; here a "token" is one annotated segment of a sentence
(a sign, or an in-between transition) and it arrives as the continuous pooled
STS-Net vectors already used for frame-level conditioning, so the embedding
table becomes a linear projection over ``3 * label_dim`` (body|dom|nondom).

It also carries three ``proj_m`` heads instead of one, because the three motion
streams (body / right hand / left hand) have their own CFM decoders and each
needs a prior ``mu`` of its own feature width.
"""

import math

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
from einops import rearrange

from src_mask_fast.utils.model import sequence_mask
from src_mask_fast.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """

    def __init__(self, d: int, base: int = 10_000):
        r"""
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        r"""Cache $\cos$ and $\sin$ values"""
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        seq_len = x.shape[0]

        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """`x` is the Tensor at the head of a key or a query with shape `[b, h, t, d]`"""
        x = rearrange(x, "b h t d -> t b h d")

        self._build_cache(x)

        x_rope, x_pass = x[..., : self.d], x[..., self.d :]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])

        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        heads_share=True,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)

        # from https://nn.labml.ai/transformers/rope/index.html
        self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)

        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    @staticmethod
    def _attention_bias_proximal(length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class SegmentEncoder(nn.Module):
    """Matcha's TextEncoder over annotated segments instead of phonemes.

    Takes one pooled STS-Net vector triple per segment and returns, per stream,
    the prior mean ``mu_x`` at segment resolution, plus one log-duration per
    segment from the shared duration predictor.
    """

    def __init__(
        self,
        encoder_params,
        duration_predictor_params,
        label_dim,
        out_channels_body,
        out_channels_hand,
        out_channels_face=None,
        use_is_sign=True,
        face_label_dim=0,
    ):
        super().__init__()
        self.label_dim = label_dim
        # Optional face-specific tail on every token (e.g. the segment's median
        # [expression | jaw 6D]). 0 -> tokens are body|dom|nondom only and emb keeps
        # its old shape, so existing checkpoints load unchanged.
        self.face_label_dim = int(face_label_dim)
        # Switchable, and the ONLY gate: when False the sign/gap one-hot is not
        # concatenated and the model cannot tell a sign from an in-between gap. The
        # dataset emits `is_sign` either way, so flipping this flag is all it takes.
        #
        # Set from configs/model/encoder/default.yaml, which currently ships False
        # (the 2026-09-18 ablation). It changes emb.weight's width, so it is baked
        # into every checkpoint's hparams: a run trained with True rebuilds with True
        # through load_from_checkpoint, whatever the config says today.
        self.use_is_sign = bool(use_is_sign)
        self.n_channels = encoder_params.n_channels
        self.out_channels_body = out_channels_body
        self.out_channels_hand = out_channels_hand
        # Face expression stream. None -> no face head is built at all, so a
        # checkpoint trained without the face stream still loads cleanly.
        self.out_channels_face = out_channels_face

        # Matcha uses nn.Embedding(n_vocab, n_channels) because phonemes are
        # discrete ids. A segment token is a real-valued vector, not an id, so the
        # "vocabulary lookup" becomes a projection -- of body|dom|nondom, plus the
        # is_sign one-hot below. ("Real-valued", not "continuous": in the *_fused
        # labels the same vector is repeated across every frame of a segment, which
        # is anything but continuous in time. That repetition exists only so the
        # frame-level path -- use_duration=False, where every frame needs its own
        # conditioning vector -- has something to read; _build_tokens mean-pools it
        # straight back to the single vector it came from.)
        #
        # With use_is_sign the layer is really both at once: a projection of the
        # continuous part and a genuine embedding lookup for the discrete part.
        # Since emb is Linear, W @ concat(x, onehot) = W_c @ x + W_s @ onehot, and
        # W_s @ onehot just selects a column -- so the one-hot IS a learned
        # per-class type embedding (the BERT trick), no extra module needed.
        #
        # TODO (from the original note here): 4-way rather than 2-way --
        # not-a-sign / begin / mid / end. That would additionally tell the encoder
        # where in a sign each segment sits, which 2-way cannot express; worth
        # trying if predicted durations are flat within multi-segment signs.
        in_dim = 3 * label_dim + self.face_label_dim + (2 if self.use_is_sign else 0)
        self.emb = torch.nn.Linear(in_dim, self.n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, self.n_channels**-0.5)
        torch.nn.init.zeros_(self.emb.bias)

        if encoder_params.prenet:
            self.prenet = ConvReluNorm(
                self.n_channels,
                self.n_channels,
                self.n_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        else:
            self.prenet = lambda x, x_mask: x

        self.encoder = Encoder(
            self.n_channels,
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )

        # One prior head per motion stream — each CFM decoder needs a mu of its
        # own feature width for the packing and for the prior loss.
        self.proj_m_body = torch.nn.Conv1d(self.n_channels, out_channels_body, 1)
        self.proj_m_r_hand = torch.nn.Conv1d(self.n_channels, out_channels_hand, 1)
        self.proj_m_l_hand = torch.nn.Conv1d(self.n_channels, out_channels_hand, 1)
        # The face prior head reads the SAME encoder representation as the three
        # motion heads -- one shared trunk over the fused segment tokens, one
        # projection per stream. Only the output width differs (10 expression
        # coefficients against 126/90).
        self.proj_m_face = (
            torch.nn.Conv1d(self.n_channels, out_channels_face, 1)
            if out_channels_face else None
        )

        self.proj_w = DurationPredictor(
            self.n_channels,
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )

    def forward(self, x, x_lengths, is_sign=None):
        """
        Args:
            x: per-segment tokens, shape (batch_size, n_segments, 3 * label_dim + face_label_dim)
            x_lengths: number of real segments per item, shape (batch_size,)
            is_sign: (batch_size, n_segments) bool -- True for an annotated sign,
                False for an inter-sign gap. Required when use_is_sign=True,
                ignored (may be None) when it is False.

        Returns:
            mu_x: tuple of (body, r_hand, l_hand, face), each
                (batch_size, feat, n_segments). `face` is None unless the encoder
                was built with out_channels_face.
            logw: predicted log durations, (batch_size, 1, n_segments)
            x_mask: (batch_size, 1, n_segments)
        """
        if self.use_is_sign:
            if is_sign is None:
                raise ValueError(
                    "SegmentEncoder(use_is_sign=True) needs is_sign; the dataset "
                    "emits it whenever data.return_tokens=True.")
            sg = is_sign.to(dtype=x.dtype)
            # [gap, sign], built by hand so this does not depend on F.one_hot's
            # dtype/device behaviour for a padded bool tensor.
            x = torch.cat([x, torch.stack([1.0 - sg, sg], dim=-1)], dim=-1)
        x = self.emb(x) * math.sqrt(self.n_channels)   # (B, n_seg, n_channels)
        x = torch.transpose(x, 1, -1)                  # (B, n_channels, n_seg)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.prenet(x, x_mask)
        x = self.encoder(x, x_mask)

        mu_x = (
            self.proj_m_body(x) * x_mask,
            self.proj_m_r_hand(x) * x_mask,
            self.proj_m_l_hand(x) * x_mask,
            self.proj_m_face(x) * x_mask if self.proj_m_face is not None else None,
        )

        # Detached, as in Matcha: the duration predictor must not shape the
        # encoder representation the prior heads rely on.
        x_dp = torch.detach(x)
        logw = self.proj_w(x_dp, x_mask)

        return mu_x, logw, x_mask
