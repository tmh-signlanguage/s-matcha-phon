"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
import os
from abc import ABC
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm, GradClipAlgorithmType
import torch.nn.functional as F

from src_mask_fast import utils
from src_mask_fast.utils.utils import plot_tensor
from src_mask_fast.utils.model import fix_len_compatibility

log = utils.pylogger.get_pylogger(__name__)

#seed = 42
#torch.manual_seed(seed=seed)
#torch.use_deterministic_algorithms(True)


def axis_angle_to_matrix(axis_angle):
    """
    axis_angle: (..., 3)
    returns: (..., 3, 3)
    """

    if axis_angle.dim() == 1:
        axis_angle = axis_angle.unsqueeze(0)

    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)

    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    cos = torch.cos(angle)[..., 0]
    sin = torch.sin(angle)[..., 0]
    one_minus = 1 - cos

    R = torch.zeros(axis.shape[0], 3, 3, device=axis_angle.device)

    R[:,0,0] = cos + x*x*one_minus
    R[:,0,1] = x*y*one_minus - z*sin
    R[:,0,2] = x*z*one_minus + y*sin

    R[:,1,0] = y*x*one_minus + z*sin
    R[:,1,1] = cos + y*y*one_minus
    R[:,1,2] = y*z*one_minus - x*sin

    R[:,2,0] = z*x*one_minus - y*sin
    R[:,2,1] = z*y*one_minus + x*sin
    R[:,2,2] = cos + z*z*one_minus

    return R


def matrix_to_axis_angle(R):
    """
    R: (T,3,3)
    returns: (T,3)
    """
    cos_angle = (R[:,0,0] + R[:,1,1] + R[:,2,2] - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1+1e-6, 1-1e-6)

    angle = torch.acos(cos_angle)

    rx = R[:,2,1] - R[:,1,2]
    ry = R[:,0,2] - R[:,2,0]
    rz = R[:,1,0] - R[:,0,1]

    axis = torch.stack([rx, ry, rz], dim=1)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)
    return axis * angle.unsqueeze(1)


def normalize_smplx_yaw(batch):

    transl = batch["smplx_mesh_transl"]
    orient = batch["smplx_mesh_global_orient"]

    # center
    transl = transl - transl[0]

    # rotation matrices
    R = axis_angle_to_matrix(orient.squeeze(1))
    R0 = R[0]

    # extract yaw
    yaw = torch.atan2(R0[0,2], R0[2,2])

    cos = torch.cos(-yaw)
    sin = torch.sin(-yaw)

    R_yaw_inv = torch.tensor([
        [ cos, 0, sin],
        [ 0,   1, 0  ],
        [-sin, 0, cos]
    ], device=R.device)

    # rotate sequence
    R = R_yaw_inv @ R
    transl = (R_yaw_inv @ transl.T).T

    batch["smplx_mesh_transl"] = transl
    batch["smplx_mesh_global_orient"] = matrix_to_axis_angle(R)

    return batch


def smplx_to_center(batch, smplx_model=None):
    transl = batch["smplx_mesh_transl"] #.to(device)         # (T,3)  -> (1, T, 3)
    orient = batch["smplx_mesh_global_orient"] #.to(args.device)     # (T,3) -> (1, T, 3)
    body_pose = batch["smplx_mesh_body_pose"] #.to(args.device)     # (T,63)
    betas = batch["smplx_mesh_betas"] #.to(args.device)              # (10)  -> torch.Size([1, 600, 300])
    lhand = batch["smplx_mesh_left_hand_pose"] #.to(args.device)     # (T,45)
    rhand = batch["smplx_mesh_right_hand_pose"] #.to(args.device)    # (T,45)

    T = transl.shape[1]

    betas = betas.reshape(-1, betas.shape[-1])
    if betas.shape[0] == 1:
        betas = betas.repeat(T, 1)

    print(f"Betas shape {betas.shape}")

    # -------------------------------------------------
    # 1 normalize translation
    transl = transl - transl[0]
    # 2 remove root yaw
    orient = orient.squeeze(0)
    # print("batch[smplx_mesh_global_orient].shape: ", orient.shape)
    R = axis_angle_to_matrix(orient)
    R0 = R[0]

    yaw = torch.atan2(R0[0, 2], R0[2, 2])

    cos = torch.cos(-yaw)
    sin = torch.sin(-yaw)

    R_yaw_inv = torch.tensor([
        [cos, 0, sin],
        [0,   1, 0],
        [-sin, 0, cos]
    ], device=R.device)

    R = R_yaw_inv @ R
    orient = matrix_to_axis_angle(R)
    transl = transl.squeeze(0)
    print(f"transl shape: {transl.shape}")  # transl shape: torch.Size([1, 600, 3])
    transl = (R_yaw_inv @ transl.T).T  # transl.T
    
    if smplx_model:
        # 3 compute joints from SMPL-X
        output = smplx_model(
            global_orient=orient,
            body_pose=body_pose,
            left_hand_pose=lhand,
            right_hand_pose=rhand,
            betas=betas,
            transl=transl,
            return_joints=True
        )

        joints = output.joints   # (T, J, 3)
        print(f"Output joints shape  (T, J, 3)? {joints.shape}")  # ([600, 127, 3])

        # 4 pelvis-relative joints
        pelvis = joints[:, 0:1, :]      # pelvis joint
        joints_rel = joints - pelvis

        batch["joints_rel"] = joints_rel
    
    batch["smplx_mesh_transl"] = transl.unsqueeze(0)
    batch["smplx_mesh_global_orient"] = orient.unsqueeze(0)

    return batch


class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def _clip_groups(self):
        """Parameter groups that get their own gradient-norm budget.

        Mirrors the module split src_mask gave a separate optimizer each.
        """
        if self.streams != 3:
            return [self]
        groups = [self.decoder_body, self.decoder_r_hand, self.decoder_l_hand]
        if getattr(self, "use_face", False):
            # Its own budget like every other stream, so an expression spike cannot
            # throttle the body update (and vice versa).
            groups.append(self.decoder_face)
        if getattr(self, "use_duration", False):
            # The segment encoder and duration predictor sit outside the three
            # decoders and are trained by the same optimizer, so they get their
            # own clip budget rather than being folded into a decoder's.
            groups.append(self.encoder)
        return groups

    def configure_optimizers(self) -> Any:
        # ONE optimizer over every parameter, for both stream counts.
        #
        # Deviates from src_mask, which built a separate AdamW per decoder (plus one
        # for the encoder under use_duration) and drove them by hand under manual
        # optimisation. The gradients are unchanged by this: the stream losses are
        # additive, the decoders' parameter sets are disjoint, and Adam state is
        # per-parameter, so d(L_b+L_r+L_l)/d(theta_body) = d(L_b)/d(theta_body)
        # either way. The only thing the split actually bought was a separate
        # gradient-norm budget per stream, and configure_gradient_clipping() below
        # reproduces that exactly.
        #
        # What the split cost, and what collapsing it gets back:
        #   * automatic optimisation, so Lightning unscales AMP gradients BEFORE
        #     clipping. The manual version clipped still-scaled gradients and then
        #     divided by the loss scale, shrinking every update by ~1/65536 -- which
        #     is why 16-mixed was unusable. Both half precisions work now.
        #   * one GradScaler.update() per iteration instead of one per optimizer,
        #     so the loss scale no longer backs off 8x on a single bad step.
        #   * the scheduler branch below. It previously existed ONLY in the
        #     streams == 1 arm; the 3-stream arm returned a bare tuple, so any
        #     scheduler config was silently ignored.
        #   * global_step that counts batches instead of 3x batches.
        #   * gradient accumulation and Trainer(gradient_clip_val=...).
        n_total = sum(p.numel() for p in self.parameters())
        n_clipped = sum(p.numel() for g in self._clip_groups() for p in g.parameters())
        print(f"optimizer covers {n_total} params; clip groups cover {n_clipped} "
              f"({'all' if n_total == n_clipped else 'MISMATCH -- some params are unclipped'})")

        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        """Give each stream its own gradient-norm budget.

        This is the one behaviour the per-decoder optimizers in src_mask actually
        provided, kept here without them. clip_grad_norm_ normalises over whatever
        parameters it is handed, so clipping the three decoders separately lets a
        spike in one hand stream be cut back without throttling the body update. A
        single global clip over self.parameters() would rescale all three together.

        Lightning calls this from Precision._after_closure, i.e. AFTER
        GradScaler.unscale_(), so the gradients seen here are in true units under
        16-mixed and bf16-mixed alike. That ordering is exactly what the
        manual-optimisation version in src_mask could not reach.

        gradient_clip_val comes from Trainer(gradient_clip_val=...); it is None in
        the shipped trainer configs, so the default reproduces the hardcoded 1.0
        that src_mask's training_step used.
        """
        clip_val = 1.0 if gradient_clip_val is None else gradient_clip_val
        if clip_val <= 0:
            return
        if gradient_clip_algorithm not in (None, "norm", GradClipAlgorithmType.NORM):
            raise ValueError(
                f"configure_gradient_clipping only implements norm clipping, got "
                f"{gradient_clip_algorithm!r}."
            )
        for module in self._clip_groups():
            torch.nn.utils.clip_grad_norm_(module.parameters(), clip_val)

    def get_losses(self, batch):
        
        subject_id = None

        if self.conditional:
            if self.smplx:
                # Get only first 10 betas -> switched to first 2 betas
                # print("batch[smplx_mesh_betas].shape: ", batch["smplx_mesh_betas"].shape) # torch.Size([32, 90, 300])
                y_lengths, subject_id = batch["length"], batch["subject_id"]
                # print("batch.keys()")
                #for key in batch.keys():
                #    print(key)
                cond_body, cond_rhand, cond_lhand = batch["body_labels"], batch["dom_labels"], batch["nondom_labels"]
                # Segment tokens for the duration module; absent unless
                # data.return_tokens=True.
                tokens = batch.get("tokens")
                x_lengths = batch.get("x_lengths")
                durations = batch.get("durations")
                # Sign vs inter-sign gap, straight from the annotations. Handed over
                # unconditionally; SegmentEncoder.use_is_sign decides whether it is
                # concatenated as a 2-way one-hot or dropped on the floor (it is the
                # single gate, currently False -- the 2026-09-18 ablation).
                is_sign = batch.get("is_sign")
                #cond = batch["smplx_mesh_betas"][:,:,:2].permute(0, 2, 1)
                #cond = cond.permute(0, 2, 1)
                #print("cond_body.shape in baselightningmodule should be (B, T, 512): ", cond_body.shape)  # should be (B, T, 512)

                # print("cond.shape after permute: ", cond.shape)  # torch.Size([32, 10, 90])
                smplx_mesh_body_pose, smplx_mesh_global_orient, smplx_mesh_transl, smplx_mesh_left_hand_pose, smplx_mesh_right_hand_pose = batch["smplx_mesh_body_pose"], batch["smplx_mesh_global_orient"], batch["smplx_mesh_transl"], batch["smplx_mesh_left_hand_pose"], batch["smplx_mesh_right_hand_pose"]
                # print(f"Shapes: smplx_mesh_body_pose {smplx_mesh_body_pose.shape}, smplx_mesh_left_hand_pose {smplx_mesh_left_hand_pose.shape}, smplx_mesh_global_orient {smplx_mesh_global_orient.shape}, smplx_mesh_transl {smplx_mesh_transl.shape}")
                # Shapes: smplx_mesh_body_pose torch.Size([16, 90, 63]), smplx_mesh_left_hand_pose torch.Size([16, 90, 45]),
                # smplx_mesh_global_orient torch.Size([16, 90, 3]), smplx_mesh_transl torch.Size([16, 90, 3])
                y = torch.cat(
                    (smplx_mesh_body_pose, smplx_mesh_right_hand_pose, smplx_mesh_left_hand_pose),
                    dim=2
                )
                # Face expression coefficients, appended LAST so the 126/90/90
                # slicing in forward() is untouched when the stream is off.
                if getattr(self, "use_face", False):
                    expression = batch.get("smplx_mesh_expression")
                    if expression is None:
                        raise KeyError(
                            "model.use_face=True but the batch has no "
                            "'smplx_mesh_expression'; the dataset must emit the "
                            f"{getattr(self, 'face_dim', 10)} FLAME expression "
                            "coefficients per frame, shaped (B, T, face_dim)."
                        )
                    y = torch.cat((y, expression), dim=2)
            else:
                y, y_lengths, cond = batch["y"], batch["y_lengths"], batch["cond"]
        else:
            # print(f"Batch keys: {batch.keys()}")
            # Batch keys: dict_keys([
            # 'subject_id', 'sequence_id', 'start_frame', 'length',
            # 'smplx_mesh_body_pose', 'smplx_mesh_global_orient', 'smplx_mesh_transl',
            # 'smplx_mesh_betas', 'smplx_mesh_left_hand_pose', 'smplx_mesh_right_hand_pose',
            # 'other_ids'])
            if self.smplx:
                # y_lengths, subject_id, cond = batch["length"], batch["subject_id"], batch["smplx_mesh_betas"].permute(0, 2, 1) # [:,:,:10]
                # cond = cond.permute(0, 2, 1)
                y_lengths, subject_id = batch["length"], batch["subject_id"]
                # cond_body, cond_rhand, cond_lhand = batch["body_labels"], batch["dom_labels"], batch["nondom_labels"]
                # batch = smplx_to_center(batch)
                # print("batch[smplx_mesh_betas].shape: ", batch["smplx_mesh_betas"].shape) [8, 90, 300]
                # Flatten first two dims - I was doing this to check that the betas are stable
                #flat_betas = batch["smplx_mesh_betas"].reshape(-1, batch["smplx_mesh_betas"].shape[2])
                # unique_slices = torch.unique(flat_betas, dim=0)
                # print(f"n unique slices: len {len(unique_slices)}")
                smplx_mesh_body_pose, smplx_mesh_global_orient, smplx_mesh_transl, smplx_mesh_left_hand_pose, smplx_mesh_right_hand_pose = batch["smplx_mesh_body_pose"], batch["smplx_mesh_global_orient"], batch["smplx_mesh_transl"], batch["smplx_mesh_left_hand_pose"], batch["smplx_mesh_right_hand_pose"]
                # print(f"Shapes: smplx_mesh_body_pose {smplx_mesh_body_pose.shape}, smplx_mesh_left_hand_pose {smplx_mesh_left_hand_pose.shape}, smplx_mesh_global_orient {smplx_mesh_global_orient.shape}, smplx_mesh_transl {smplx_mesh_transl.shape}")
                # Shapes: smplx_mesh_body_pose torch.Size([16, 90, 63]), smplx_mesh_left_hand_pose torch.Size([16, 90, 45]),
                # smplx_mesh_global_orient torch.Size([16, 90, 3]), smplx_mesh_transl torch.Size([16, 90, 3])
                y = torch.cat(
                    (smplx_mesh_body_pose, smplx_mesh_right_hand_pose, smplx_mesh_left_hand_pose),
                    dim=2
                )
                # print("Print y.shape after concatenating input data", y.shape)
            else:
                y, y_lengths = batch["y"], batch["y_lengths"]
            cond = None

        if self.streams == 1:
            # x, x_lengths = batch["x"], batch["x_lengths"]
            # print("Batch.shape:", batch.shape)  # torch.Size([10, 42, 90])
            # batch shape: torch.Size([10, 42, 90]), batch lengths: torch.Size([10])
            # print("y in base lightning .shape:", y.shape, y_lengths.shape) # y.shape: torch.Size([10, 42, 90]) ; torch.Size([10])
            diff_loss = self(
                # x=x,
                # x_lengths=x_lengths,
                y=y,
                y_lengths=y_lengths,
                cond=cond
                # spks=spks,
                # out_size=self.out_size,
                # durations=batch["durations"],
            )
            return {
                "diff_loss": diff_loss,
            }
        elif self.streams == 3:
            T = y.shape[1]  # current time dim
            T_fixed = fix_len_compatibility(T)    # rounds up to nearest 2^N multiple
            if T_fixed != T:
                pad = T_fixed - T
                y = F.pad(y, (0, 0, 0, pad))   # pad time dim: (B, T, features)
                # pad conditions to match
                cond_body = F.pad(cond_body, (0, 0, 0, pad))
                cond_rhand = F.pad(cond_rhand, (0, 0, 0, pad))
                cond_lhand = F.pad(cond_lhand, (0, 0, 0, pad))
            if self.conditional:
                if getattr(self, "use_duration", False):
                    assert tokens is not None, (
                        "use_duration=True but the batch has no `tokens`; "
                        "set data.return_tokens=True in the data config"
                    )
                    out = self(
                        y=y,
                        y_lengths=y_lengths,
                        cond=(cond_body, cond_rhand, cond_lhand),
                        x=tokens,
                        x_lengths=x_lengths,
                        durations=durations,
                        is_sign=is_sign,
                    )
                    # forward() appends the face term when the stream is on, so the
                    # 5-tuple every existing run produced is still a 5-tuple.
                    if getattr(self, "use_face", False):
                        (diff_loss_body, diff_loss_r_hand, diff_loss_l_hand,
                         dur_loss, prior_loss, diff_loss_face) = out
                    else:
                        (diff_loss_body, diff_loss_r_hand, diff_loss_l_hand,
                         dur_loss, prior_loss) = out
                        diff_loss_face = None
                    losses = {
                        "diff_loss_body": diff_loss_body,
                        "diff_loss_r_hand": diff_loss_r_hand,
                        "diff_loss_l_hand": diff_loss_l_hand,
                        "dur_loss": dur_loss,
                        "prior_loss": prior_loss,
                    }
                    if diff_loss_face is not None:
                        losses["diff_loss_face"] = diff_loss_face
                    return losses
                streams_out = self(
                    # x=x,
                    # x_lengths=x_lengths,
                    y=y,
                    y_lengths=y_lengths,
                    cond=(cond_body, cond_rhand, cond_lhand),
                    # Only present when model.use_contact_loss is on; forward()
                    # treats them as optional and skips the term without them.
                    betas=batch.get("smplx_mesh_betas"),
                    soft_dist=batch.get("soft_dist"),
                    # spks=subject_id,
                    # out_size=self.out_size,
                    # durations=batch["durations"],
                )
                if getattr(self, "use_face", False):
                    (diff_loss_body, diff_loss_r_hand, diff_loss_l_hand,
                     diff_loss_face) = streams_out
                else:
                    diff_loss_body, diff_loss_r_hand, diff_loss_l_hand = streams_out
                    diff_loss_face = None
                out = {
                    "diff_loss_body": diff_loss_body,
                    "diff_loss_r_hand": diff_loss_r_hand,
                    "diff_loss_l_hand": diff_loss_l_hand
                }
                if diff_loss_face is not None:
                    out["diff_loss_face"] = diff_loss_face
                # Set by forward() only on the steps where the term fires; kept out
                # of the dict otherwise so training_step falls back to the plain sum.
                if getattr(self, "_contact_term", None) is not None:
                    out["contact_loss"] = self._contact_term
                return out
            else:
                diff_loss_body, diff_loss_r_hand, diff_loss_l_hand = self(
                    # x=x,
                    # x_lengths=x_lengths,
                    y=y,
                    y_lengths=y_lengths,
                    cond=None
                    # spks=subject_id,
                    # out_size=self.out_size,
                    # durations=batch["durations"],
                )
                return {
                    "diff_loss_body": diff_loss_body,
                    "diff_loss_r_hand": diff_loss_r_hand,
                    "diff_loss_l_hand": diff_loss_l_hand
                }
    
    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def _mix_contact(self, diff, contact):
        """Blend the contact term with the diffusion loss at a real 50/50.

        The two have unrelated scales (diffusion ~2.2, contact BCE ~0.1-0.7), so a
        raw weight would leave diffusion dominant by an order of magnitude and the
        configured ratio would be fiction. Each is divided by an EMA of its own
        magnitude first; the result is multiplied back onto the diffusion scale so
        the total keeps its familiar size and the learning rate does not need
        retuning. The EMAs are updated under no_grad -- they are a measurement of
        the losses, not a path for gradient.
        """
        cfg = self.contact_cfg
        d, mix = float(cfg.ema_decay), float(cfg.mix)
        with torch.no_grad():
            dv, cv = diff.detach().abs(), contact.detach().abs()
            if float(self._ema_diff) == 0.0:
                self._ema_diff.copy_(dv)
            else:
                self._ema_diff.mul_(d).add_(dv, alpha=1.0 - d)
            if float(self._ema_contact) == 0.0:
                self._ema_contact.copy_(cv)
            else:
                self._ema_contact.mul_(d).add_(cv, alpha=1.0 - d)
        scale = (self._ema_diff / self._ema_contact.clamp_min(1e-8)).detach()
        self.log("contact/scale", scale, on_step=True, logger=True, sync_dist=True)
        return (1.0 - mix) * diff + mix * scale * contact

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        if self.streams == 1:
            self.log(
                "step",
                float(self.global_step),
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "sub_loss/train_diff_loss",
                loss_dict["diff_loss"],
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        
        elif self.streams == 3:
            use_dur = getattr(self, "use_duration", False)
            # zero_grad / manual_backward / clip / step are gone: this module runs
            # under automatic optimisation now, so Lightning does all four in the
            # right order (notably clipping AFTER the AMP unscale). See
            # configure_optimizers() and configure_gradient_clipping().
            self.log(
                "sub_loss/train_diff_loss_body",
                loss_dict["diff_loss_body"],
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "sub_loss/train_diff_loss_r_hand",
                loss_dict["diff_loss_r_hand"],
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "sub_loss/train_diff_loss_l_hand",
                loss_dict["diff_loss_l_hand"],
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            if "diff_loss_face" in loss_dict:
                self.log(
                    "sub_loss/train_diff_loss_face",
                    loss_dict["diff_loss_face"],
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
            if use_dur:
                for name in ("dur_loss", "prior_loss"):
                    self.log(
                        f"sub_loss/train_{name}",
                        loss_dict[name],
                        on_step=True,
                        on_epoch=True,
                        logger=True,
                        sync_dist=True,
                    )

        # Built from named terms rather than sum(loss_dict.values()). Under automatic
        # optimisation this return value IS what Lightning backpropagates, so a
        # diagnostic scalar added to loss_dict later must not be able to join the
        # gradient silently. src_mask summed the dict, which was harmless only
        # because manual optimisation ignored the return value.
        if self.streams == 1:
            total_loss = loss_dict["diff_loss"]
        else:
            total_loss = (loss_dict["diff_loss_body"]
                + loss_dict["diff_loss_r_hand"]
                + loss_dict["diff_loss_l_hand"])
            # Unweighted, like the three motion streams. Worth revisiting: the face
            # is 10 channels against 306, so an equal weight gives the expression
            # coefficients a much larger per-channel share of the gradient than
            # they have of the data.
            if "diff_loss_face" in loss_dict:
                total_loss = total_loss + loss_dict["diff_loss_face"]
            if getattr(self, "use_duration", False):
                # Without these two terms the encoder gets gradient only through the
                # packed mu, and the duration predictor (which is fed a detached
                # encoder output) gets none at all.
                total_loss = total_loss + loss_dict["dur_loss"] + loss_dict["prior_loss"]

        contact = loss_dict.get("contact_loss")
        if contact is not None:
            self.log("sub_loss/train_contact", contact, on_step=True, on_epoch=True,
                     logger=True, sync_dist=True)
            total_loss = self._mix_contact(total_loss, contact)

        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):
        # print("Validation step")
        # print(f"Batch keys: {batch.keys()}, batch shape: {batch['y'].shape}, batch lengths: {batch['y_lengths'].shape}")
        # dict_keys(['y', 'y_lengths']), batch shape: torch.Size([10, 42, 90]), batch lengths: torch.Size([10])
        loss_dict = self.get_losses(batch)
        if self.streams == 1:
            self.log(
                "sub_loss/val_diff_loss",
                loss_dict["diff_loss"],
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        elif self.streams == 3:
            self.log(
                "sub_loss/val_diff_loss_body",
                loss_dict["diff_loss_body"],
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "sub_loss/val_diff_loss_r_hand",
                loss_dict["diff_loss_r_hand"],
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "sub_loss/val_diff_loss_l_hand",
                loss_dict["diff_loss_l_hand"],
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            if "diff_loss_face" in loss_dict:
                self.log(
                    "sub_loss/val_diff_loss_face",
                    loss_dict["diff_loss_face"],
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
            if getattr(self, "use_duration", False):
                for name in ("dur_loss", "prior_loss"):
                    self.log(
                        f"sub_loss/val_{name}",
                        loss_dict[name],
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        sync_dist=True,
                    )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val_epoch",
            total_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss
    
    def test_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        if self.streams == 1:
            self.log(
                "sub_loss/test_diff_loss",
                loss_dict["diff_loss"],
                on_step=False,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        elif self.streams == 3:
            self.log(
                "sub_loss/test_diff_loss_body",
                loss_dict["diff_loss_body"],
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "sub_loss/test_diff_loss_r_hand",
                loss_dict["diff_loss_r_hand"],
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "sub_loss/test_diff_loss_l_hand",
                loss_dict["diff_loss_l_hand"],
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            if "diff_loss_face" in loss_dict:
                self.log(
                    "sub_loss/test_diff_loss_face",
                    loss_dict["diff_loss_face"],
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
        total_loss = sum(loss_dict.values())
        self.log(
            "loss/test",
            total_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
    
    def on_before_optimizer_step(self, optimizer):
        # self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
        pass

    def on_validation_end(self) -> None:
        """Keep the CURRENT weights on disk after every validation, as current.ckpt.

        This used to call trainer.save_checkpoint(checkpoint_callback.best_model_path),
        i.e. it wrote the current weights over the file ModelCheckpoint had picked as
        the BEST one, after every validation. Every run's best-named checkpoint (e.g.
        22148300's epoch=187-step=83758.ckpt, 22507765's epoch=21-step=9801.ckpt -- the
        latter really epoch 210) therefore held the final weights, and train.py's
        trainer.test(ckpt_path=best_model_path) tested those, not the best ones.

        Now the best/last files belong to ModelCheckpoint alone, and the latest weights
        go to their own file, written atomically (tmp + rename) so a reader never sees
        a half-written checkpoint. save_checkpoint is called on every rank -- Lightning
        writes from rank 0 only but the call is collective -- and the rename on rank 0.
        """
        callback = self.trainer.checkpoint_callback
        if self.trainer.sanity_checking or callback is None or not callback.dirpath:
            return
        path = os.path.join(callback.dirpath, "current.ckpt")
        tmp = path + ".tmp"
        self.trainer.save_checkpoint(tmp)
        if self.trainer.is_global_zero:
            os.replace(tmp, path)
            log.debug(f"Saved current weights (epoch {self.current_epoch}) to {path}")

"""    def on_validation_end(self) -> None:  # Use this for visualisation of generated samples
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                output = self.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=10, spks=spks)
                y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]
                attn = output["attn"]
                self.logger.experiment.add_image(
                    f"generated_enc/{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
"""
