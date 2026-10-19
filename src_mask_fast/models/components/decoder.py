# --- Author: Shivam Mehta ---
# --- Matcha-TTS ---

import math
from typing import Optional

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F
from diffusers.models.activations import get_activation
from einops import pack, rearrange, repeat, unpack

from src_mask_fast.models.components.transformer import BasicTransformerBlock

#seed = 42
#torch.manual_seed(seed=seed)
# torch.cuda.manual_seed(seed)
#torch.use_deterministic_algorithms(True)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MaskedGroupNorm(torch.nn.GroupNorm):
    """GroupNorm that computes its statistics over valid frames only.

    Deviates from original Matcha, because upstream GroupNorm averages over the
    padded zeros too, so the same sequence produces a different output depending
    on how much padding its batch-mates force on it.
    """

    def forward(self, x, mask=None):
        if mask is None:
            return super().forward(x)
        # Statistics are computed in fp32 no matter what autocast is doing.
        #
        # torch's own group_norm sits on autocast's fp32 promotion list, so stock
        # GroupNorm (what Matcha uses) is protected automatically. This one is hand
        # rolled from mul/sum/rsqrt, and autocast leaves those in the input dtype.
        # Under 16-mixed that would put self.eps (1e-5) below fp16's smallest normal
        # (6.1e-5), so it stops acting as a floor and rsqrt(var + eps) returns inf
        # for any group whose variance underflows. In fp32 the casts below are
        # no-ops, so this changes nothing for the 32-bit runs.
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            B, C, T = xf.shape
            G, cg = self.num_groups, self.num_channels // self.num_groups
            xg = xf.view(B, G, cg, T)
            mg = mask.to(torch.float32).view(B, 1, 1, T)
            # elements per group = valid frames x channels-per-group
            n = mg.sum(dim=(2, 3), keepdim=True).clamp(min=1.0) * cg
            mean = (xg * mg).sum(dim=(2, 3), keepdim=True) / n
            var = (((xg - mean) * mg) ** 2).sum(dim=(2, 3), keepdim=True) / n
            xg = (xg - mean) * torch.rsqrt(var + self.eps)
            out = xg.view(B, C, T)
            if self.affine:
                out = out * self.weight.float().view(1, C, 1) + self.bias.float().view(1, C, 1)
        return out.to(in_dtype)


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        # Kept as a Sequential purely so state_dict keys stay block.0.* / block.1.*;
        # forward indexes into it because the norm needs the mask.
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            MaskedGroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):  # , mask
        #print(f"x shape in Block1D forward: {x.shape}, mask shape: {mask.shape}")  # Debugging line
        output = self.block[0](x * mask)      # Conv1d
        output = self.block[1](output, mask)  # MaskedGroupNorm, see class docstring
        output = self.block[2](output)        # Mish
        return output * mask  # * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):  # mask, 
        # print(f"x shape in ResnetBlock1D forward: {x.shape}, mask shape: {mask.shape}, time emb {time_emb.shape}")  # Debugging line
        h = self.block1(x, mask)  # , mask
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)  # , mask
        output = h + self.res_conv(x * mask)  # x * mask
        return output


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        pos_emb_dim = 128                       # choose any even number (e.g., 64, 128, 256)
        self.time_embeddings = SinusoidalPosEmb(pos_emb_dim)

        # self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=pos_emb_dim,  # in_channels,  # 39
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        ###
        #self.frame_pos_emb = SinusoidalPosEmb(self.in_channels)
        ###

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i]
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2

            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        up_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        # print(f"Decoder channels[-1]: {channels[-1]}")  # Debugging line
        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    @staticmethod
    def get_block(block_type, dim, attention_head_dim, num_heads, dropout, act_fn):
        """if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )"""
        if block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

        return block

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, t, cond=None, mu=None):  # mask,  mu, spks=None, cond=None):
        # y, mask, t.squeeze())
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            mu (torch.Tensor): output of encoder shape: (batch_size, n_feats, mel_timesteps)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        ###
        #seq_len = x.shape[-1]
        #frame_idx = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        #pos_emb = self.frame_pos_emb(frame_idx)        # (seq_len, in_channels)
        #pos_emb = pos_emb.permute(1, 0).unsqueeze(0)    # (1, in_channels, seq_len)
        #x = x + pos_emb
        ###

        # x = pack(x, "b * t")[0]  # [42, 10, 90]
        # print(f"x shape in Decoder forward: {x.shape} it was working before with x = pack([x, mu], 'b * t')[0]")
        # x shape in Decoder forward: torch.Size([10, 42, 90]) it was working before with x = pack([x, mu], 'b * t')[0]
        
        if mu is not None:
            # Matcha packs the aligned encoder prior alongside x exactly like this;
            # here it sits next to cond, so in_channels = pose + label_dim + mu width.
            x, _ = pack([x, mu], "b * t")

        if cond is not None:
            # print(f"x.shape before packing: {x.shape}")
            #twod_pose_condioning = repeat(conditioning, "b c -> b c t", t=x.shape[-1])
            
            # packing from matcha-tts
            x, ps = pack([x, cond], "b * t")

            # Addition instead of packing:
            # x = x + cond[:, :x.shape[1], :]
            # print(f"x.shape after packing: {x.shape}") # torch.Size([32, 45, 90])  # ([32, 426, 90]) x.shape after packing: torch.Size([2, 663, 200])

            # At inference time it is hardcoded, which is not good for the stability of this code: x.shape before packing: torch.Size([4, 65, 90]) x.shape after packing: torch.Size([4, 91, 90])

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)  # mask_down, 
            # print(f"x shape after resnet in Decoder forward: {x.shape}")  # Debugging line
            x = rearrange(x, "b c t -> b t c")  # "b c t -> b t c"
            mask_down = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down) #x)  # (x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)  # , mask_mid,
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)  # , mask_up, 
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            x = upsample(x * mask_up)  # x)  # (x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up) # x

        #print(f"Decoder forward output shape {output.shape}")
        #if cond is not None:
        #    output_x, _ = unpack(output, ps, 'b * t')
        #print(f"Decoder forward output_x shape {output_x.shape}")

        return output * mask # * mask


class Decoder_small_test(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Write my own decoder that doesn't do masking and that is simpler
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = in_channels
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=in_channels,
            act_fn="silu",
        )
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_embed_dim, in_channels))

        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=in_channels, batch_first=True)

        self.linear = nn.Linear(in_channels, out_channels)
        

    def forward(self, x, t):
        """Forward pass of the Decoder model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            t (torch.Tensor): shape (batch_size) - random timestamp sampled during flow matching

        Returns:
            torch.Tensor: shape (batch_size, out_channels, time)
        """
        # (f"Shape of x in simple Decoder: {x.shape}, t shape: {t.shape}")
        # Shape of x in simple Decoder: torch.Size([2, 42, 90]), t shape: torch.Size([2])
        # x = rearrange(x, "b c t -> b t c")  # Rearrange to (batch_size, time, in_channels)
        t = self.time_embeddings(t)
        t = self.time_mlp(t)
        # print(f"Shape of t after time embeddings and time mlp: {t.shape}")  # Debugging line
        # Shape of t after time embeddings and time mlp: torch.Size([2, 42]) b, c
        t_to_concat = self.mlp(t).unsqueeze(-1)
        x += t_to_concat
        # print(f"Shape of x after adding time embeddings: {x.shape}")  # Debugging line
        # Shape of x after adding time embeddings: torch.Size([2, 42, 90])
        # LSTM expects (batch_size, time, in_channels)
        x = rearrange(x, "b c t -> b t c")
        x, _ = self.lstm(x)
        # print(f"Shape of x after LSTM: {x.shape}")
        # Shape of x after LSTM: torch.Size([2, 90, 84])
        x = self.linear(x)
        # print(f"Shape of x after linear layer: {x.shape}")
        # Shape of x after linear layer: torch.Size([2, 90, 42])
        x = rearrange(x, "b t c -> b c t")
        # print(f"Final output shape of Decoder: {x.shape}")
        # Final output shape of Decoder: torch.Size([2, 42, 90])
        return x

        
