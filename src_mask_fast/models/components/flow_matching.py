# --- Author: Shivam Mehta ---
# --- Matcha-TTS ---

from abc import ABC
import math

import torch
import torch.nn.functional as F

from src_mask_fast.models.components.decoder import Decoder
from src_mask_fast.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

#seed = 42
# torch.manual_seed(seed=seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed=seed)
# torch.use_deterministic_algorithms(True)


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        out_channels,
        cfm_params
        #n_spks=1,
        #spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.out_channels = out_channels
        #self.n_spks = n_spks
        #self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None


    @torch.inference_mode()
    def forward(self, n_timesteps, batch_size, sequence_length, mask=None, cond=None, x0=None, mu=None, temperature=1.0, device='cuda'):  #  mu, mask, n_timesteps, cond=None, temperature=1.0
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        #batch_size = x.shape[0]
        #sequence_length = x.shape[-1]
        # randn
        # torch.manual_seed(42)
        # z = torch.randn(batch_size, self.out_channels, sequence_length) * temperature
        print("self.out_channels ", self.out_channels)
        if self.out_channels == 39:
            # body
            z = x0[:, :39, :] * temperature
        elif self.out_channels == 48:
            #    left OR right hand — same noise for both
            z = x0[:, 39:39+48, :] * temperature
        elif self.out_channels == 63:  # body smplx
            # z = torch.cat((x0[:, :63, :], x0[:,63+45+45:,:]), dim=1) * temperature
            # z = x0[:, :63, :] * temperature
            z = torch.randn(batch_size, 63, sequence_length, device=device) * temperature
        elif self.out_channels == 45:
            #    left OR right hand — same noise for both
            # z = x0[:, 63:63+45, :] * temperature
            z = torch.randn(batch_size, 45, sequence_length, device=device) * temperature
        elif self.out_channels == 126:  # body smplx
            # z = torch.cat((x0[:, :63, :], x0[:,63+45+45:,:]), dim=1) * temperature
            # z = x0[:, :126, :] * temperature
            z = torch.randn(batch_size, 126, sequence_length, device=device) * temperature
        elif self.out_channels == 90:
            #    left OR right hand — same noise for both
            # z = x0[:, 126:126+90, :] * temperature
            z = torch.randn(batch_size, 90, sequence_length, device=device) * temperature
        elif self.out_channels == 363:  # body smplx
            # z = torch.cat((x0[:, :63, :], x0[:,63+45+45:,:]), dim=1) * temperature
            # z = x0[:, :363, :] * temperature
            z = torch.randn(batch_size, 363, sequence_length, device=device) * temperature
        elif self.out_channels == 345:
            #    left OR right hand — same noise for both
            # z = x0[:, 363:363+345, :] * temperature
            z = torch.randn(batch_size, 345, sequence_length, device=device) * temperature
        else:
            # Any width not in the table above -- the 10-coefficient face stream is
            # the first. Every smplx branch above does exactly this; without the
            # fallback an unlisted width left z unbound and crashed one line down.
            z = torch.randn(batch_size, self.out_channels, sequence_length,
                            device=device) * temperature

        z = z.to(device=device)
        #if cond is None:
        #    z = torch.randn(batch_size, self.n_feats, sequence_length) * temperature  # torch.randn_like(mu) * temperature  this is an initial random noise to go to from z to 0/data
        # elif cond is not None:
            # At inference time it is hardcoded, which is not good for the stability of this code
        #    print("cond.shape, self.n_feats ", cond.shape, self.n_feats)
            #if cond.shape[1] == 26:
            #    x_feats = 39
            #else:
            #    x_feats = 48

            # cond = cond.clone()
            # cond[..., 1::2] = 0   # 6417
            # cond[..., 1::2] = cond[..., 0:-1:2]   # 6418, 6623
            # feature_stride = 2  # 6419
            # cond[:, ::feature_stride, :] = 0    # 6419

            # 6570
            # Instead of zeroes previous frame: mask every 3rd frame
            # cond[..., 1::3] = cond[..., 0:-1:3]   # 6570

            # 6573
            # Instead of zeroes previous frame: mask every 4th frame
            # cond[..., 1::4] = cond[..., 0:-1:4]   # 6573

            # 6574
            # Instead of zeroes previous frame: mask every 5th frame
            # cond[..., 1::5] = cond[..., 0:-1:5]   # 6574

            # 6576
            # not mask every 3rd frame, masking with previous frame
            # keep = (torch.arange(cond.size(-1), device=cond.device) % 3 == 0)  # keep every 3rd starting from index 0
            # cond[..., ~keep] = torch.cat([cond[..., :1], cond[..., :-1]], dim=-1)[..., ~keep]  # everything not kept

            # 6578
            # not mask every 4th frame, masking with previous frame
            # keep = (torch.arange(cond.size(-1), device=cond.device) % 4 == 0)  # keep every 4th starting from index 0
            # cond[..., ~keep] = torch.cat([cond[..., :1], cond[..., :-1]], dim=-1)[..., ~keep]  # everything not kept
 
            # 6582
            # not mask every 5th frame, masking with previous frame
            # keep = (torch.arange(cond.size(-1), device=cond.device) % 5 == 0)  # keep every 5th starting from index 0
            # cond[..., ~keep] = torch.cat([cond[..., :1], cond[..., :-1]], dim=-1)[..., ~keep]  # everything not kept

            # z = torch.randn(batch_size, x_feats, sequence_length) * temperature
        
        if mask is None:
            # No lengths given -> every frame is valid. Decoder expects (B, 1, T).
            mask = torch.ones(z.shape[0], 1, z.shape[-1], dtype=z.dtype, device=z.device)
        mask = mask.to(dtype=z.dtype, device=z.device)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)
        return self.solve_euler(z, t_span=t_span, mask=mask, cond=cond, mu=mu)

    def solve_euler(self, x, t_span, mask=None, cond=None, mu=None):   # , mask, mu, spks, cond)
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        # print(self.estimator)
        # print(f"t_span shape should be  (n_timesteps + 1,): {t_span.shape}")  # torch.Size([11]
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            # print(f"Solve Euler x.shape: {x.shape}, t: {t}")
            # x.shape torch.Size([8, 42, 90]), out_channels: torch.Size([8, 1, 90]), t: 0.0
            # In inference Solve Euler x.shape: torch.Size([2, 42, 90]), t: 0.0 -> Solve Euler x.shape: torch.Size([2, 42, 90]), t: 0.8999999761581421
            dphi_dt = self.estimator(x, mask, t, cond=cond, mu=mu) # mask, spks, mu, cond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        # print(f"ODE solutions: {sol}")
        return sol[-1]

    def compute_loss(self, x1, mask, cond=None, mu=None, bicrossattention=None, t=None,
                     return_extras=False):  # t/return_extras: this is for contacts aware loss -- accepted but IGNORED, see docstring
        """Computes diffusion loss

        Args:
            # this is for contacts aware loss -- REVERTED to the 31fc847 behaviour.
            # `t` and `return_extras` are still accepted so the existing
            # compute_loss(t=t_shared, ...) call sites in pretrain_body_module.py
            # keep running, but both are IGNORED here: t is always drawn inside
            # this function again, as in 31fc847. Uncomment the blocks below to
            # bring the shared timestep + extras back for the contact loss.
            # t (torch.Tensor, optional): flow-matching timestep, shape (B, 1, 1).
            #     None -> drawn here, the original behaviour. The 3-stream caller
            #     in pretrain_body_module.forward() passes ONE draw to all three
            #     decoders so body and both hands sit at the same noise level in a
            #     given step. See the comment at the sampling site there for why.
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        # sample noise p(x_0)
        # this is for contacts aware loss -- the shared-timestep path below is
        # commented out; every stream draws its own t again, as in 31fc847.
        # if t is None:
        #     t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        # else:
        #     t = t.to(device=x1.device, dtype=x1.dtype)
        # # Noise p(x_0) stays per-stream: the streams have different channel counts
        # # (126 / 90 / 90), so there is no shared z to speak of. What is shared is
        # # the noise LEVEL t, not the noise draw.
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        # Deviates from original Matcha, because u is built from randn and stays
        # non-zero on padded frames, so padding noise leaked into the reported loss
        # that ModelCheckpoint and EarlyStopping select on.
        u = u * mask

        # what happens if we don't include noise at all?
        # y = t * x1
        # u = x1

        if cond is not None:
            # 6417
            # mask every other frame with 0
            # zero-out frames 1,3,5,... along the last (time) dimension
            cond = cond.clone()
            # cond[..., 1::2] = 0   # 6417

            # 6418, 6623
            # Instead of zeroes previous frame: for indices 1,3,5,... copy from 0,2,4,...
            # cond[..., 1::2] = cond[..., 0:-1:2]   # 6418

            # 6570
            # Instead of zeroes previous frame: mask every 3rd frame
            # cond[..., 1::3] = cond[..., 0:-1:3]   # 6570

            # 6573
            # Instead of zeroes previous frame: mask every 4th frame
            # cond[..., 1::4] = cond[..., 0:-1:4]   # 6573

            # 6574
            # Instead of zeroes previous frame: mask every 5th frame
            # cond[..., 1::5] = cond[..., 0:-1:5]   # 6574

            # 6576
            # not mask every 3rd frame, masking with previous frame
            # keep = (torch.arange(cond.size(-1), device=cond.device) % 3 == 0)  # keep every 3rd starting from index 0
            # cond[..., ~keep] = torch.cat([cond[..., :1], cond[..., :-1]], dim=-1)[..., ~keep]  # everything not kept

            # 6578
            # not mask every 4th frame, masking with previous frame
            # keep = (torch.arange(cond.size(-1), device=cond.device) % 4 == 0)  # keep every 4th starting from index 0
            # cond[..., ~keep] = torch.cat([cond[..., :1], cond[..., :-1]], dim=-1)[..., ~keep]  # everything not kept

            # 6582
            # not mask every 5th frame, masking with previous frame
            # keep = (torch.arange(cond.size(-1), device=cond.device) % 5 == 0)  # keep every 5th starting from index 0
            # cond[..., ~keep] = torch.cat([cond[..., :1], cond[..., :-1]], dim=-1)[..., ~keep]  # everything not kept

            # 6419
            # Instead subsample features with features stridde
            # feature_stride = 2
            # cond[:, ::feature_stride, :] = 0    # 6419
            
            output = self.estimator(y, mask, t.squeeze(), cond=cond, mu=mu)
        else:
            output = self.estimator(y, mask, t.squeeze(), mu=mu)  # , spks, mu, cond)

        if bicrossattention:
            return output, u
        
        loss = F.mse_loss(output, u, reduction="sum") / (torch.sum(mask) * u.shape[1])
        # this is for contacts aware loss
        # if return_extras:
        #     # y is y_t and output is the predicted velocity field, so the caller can
        #     # reconstruct x1_hat = y_t + (1-t)*v_hat. Exact for this path as
        #     # sigma_min -> 0 (it is 1e-4), which is what the contact loss needs:
        #     # the estimator never predicts a pose directly.
        #     return loss, y, t, output
        # loss = F.mse_loss(output, u, reduction="sum") / u.shape[1]  # mask, 
        # print(f"Loss in CFM: {loss.item()}")

        # Add regurlarization to keep exp-maps within [0, pi]
        #overshoot = torch.relu(torch.norm(output, dim=-1) - math.pi)
        #weight = 0.1
        #loss += weight * overshoot.pow(2).mean()
        # print(f"Regularized loss: {loss.item()}")

        # loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y  # 3115.2578125


class CFM(BASECFM):
    def __init__(self, in_channels, out_channels, cfm_params, decoder_params):  #, n_spks=1, spk_emb_dim=64
        super().__init__(
            n_feats=in_channels,
            out_channels=out_channels,
            cfm_params=cfm_params,
            # n_spks=n_spks,
            # spk_emb_dim=spk_emb_dim,
        )

        # in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channels, **decoder_params)
