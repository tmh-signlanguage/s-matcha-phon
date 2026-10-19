import datetime as dt
import math
import random
import torch
import torch.nn as nn

from src_mask_fast import utils
from src_mask_fast.models.baselightningmodule import BaseLightningClass
from bidirectional_cross_attention import BidirectionalCrossAttention
from scipy.signal import find_peaks
from einops import rearrange
import torch.nn.functional as F
from src_mask_fast.models.components.flow_matching import CFM
from src_mask_fast.models.components.text_encoder import SegmentEncoder
import src_mask_fast.utils.monotonic_align as monotonic_align  # pylint: disable=consider-using-from-import
from src_mask_fast.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)

log = utils.pylogger.get_pylogger(__name__)

#seed = 42
#torch.manual_seed(seed=seed)
# torch.cuda.manual_seed(seed)
#random.seed(seed)
#torch.use_deterministic_algorithms(True)


def get_joint_indices():
    # the json file is created in pymo.preprocessing
    joint_ids_mocap = {
        "0": "RightHandPinky3_alpha", "1": "RightHandPinky3_beta", "2": "RightHandPinky3_gamma", "3": "RightHandPinky2_alpha", "4": "RightHandPinky2_beta", "5": "RightHandPinky2_gamma", "6": "RightHandPinky1_alpha", "7": "RightHandPinky1_beta", "8": "RightHandPinky1_gamma", "9": "RightHandRing3_alpha", "10": "RightHandRing3_beta", "11": "RightHandRing3_gamma", "12": "RightHandRing2_alpha", "13": "RightHandRing2_beta", "14": "RightHandRing2_gamma", "15": "RightHandRing1_alpha", "16": "RightHandRing1_beta", "17": "RightHandRing1_gamma", "18": "RightHandMiddle3_alpha", "19": "RightHandMiddle3_beta", "20": "RightHandMiddle3_gamma", "21": "RightHandMiddle2_alpha", "22": "RightHandMiddle2_beta", "23": "RightHandMiddle2_gamma", "24": "RightHandMiddle1_alpha", "25": "RightHandMiddle1_beta", "26": "RightHandMiddle1_gamma", "27": "RightHandIndex3_alpha", "28": "RightHandIndex3_beta", "29": "RightHandIndex3_gamma", "30": "RightHandIndex2_alpha", "31": "RightHandIndex2_beta", "32": "RightHandIndex2_gamma", "33": "RightHandIndex1_alpha", "34": "RightHandIndex1_beta", "35": "RightHandIndex1_gamma", "36": "RightHandThumb3_alpha", "37": "RightHandThumb3_beta", "38": "RightHandThumb3_gamma", "39": "RightHandThumb2_alpha", "40": "RightHandThumb2_beta", "41": "RightHandThumb2_gamma", "42": "RightHandThumb1_alpha", "43": "RightHandThumb1_beta", "44": "RightHandThumb1_gamma",
        "45": "RightHand_alpha", "46": "RightHand_beta", "47": "RightHand_gamma", "48": "RightForeArm_alpha", "49": "RightForeArm_beta", "50": "RightForeArm_gamma", "51": "RightArm_alpha", "52": "RightArm_beta", "53": "RightArm_gamma", "54": "RightShoulder_alpha", "55": "RightShoulder_beta", "56": "RightShoulder_gamma",
        "57": "LeftHandPinky3_alpha", "58": "LeftHandPinky3_beta", "59": "LeftHandPinky3_gamma", "60": "LeftHandPinky2_alpha", "61": "LeftHandPinky2_beta", "62": "LeftHandPinky2_gamma", "63": "LeftHandPinky1_alpha", "64": "LeftHandPinky1_beta", "65": "LeftHandPinky1_gamma", "66": "LeftHandRing3_alpha", "67": "LeftHandRing3_beta", "68": "LeftHandRing3_gamma", "69": "LeftHandRing2_alpha", "70": "LeftHandRing2_beta", "71": "LeftHandRing2_gamma", "72": "LeftHandRing1_alpha", "73": "LeftHandRing1_beta", "74": "LeftHandRing1_gamma", "75": "LeftHandMiddle3_alpha", "76": "LeftHandMiddle3_beta", "77": "LeftHandMiddle3_gamma", "78": "LeftHandMiddle2_alpha", "79": "LeftHandMiddle2_beta", "80": "LeftHandMiddle2_gamma", "81": "LeftHandMiddle1_alpha", "82": "LeftHandMiddle1_beta", "83": "LeftHandMiddle1_gamma", "84": "LeftHandIndex3_alpha", "85": "LeftHandIndex3_beta", "86": "LeftHandIndex3_gamma", "87": "LeftHandIndex2_alpha", "88": "LeftHandIndex2_beta", "89": "LeftHandIndex2_gamma", "90": "LeftHandIndex1_alpha", "91": "LeftHandIndex1_beta", "92": "LeftHandIndex1_gamma", "93": "LeftHandThumb3_alpha", "94": "LeftHandThumb3_beta", "95": "LeftHandThumb3_gamma", "96": "LeftHandThumb2_alpha", "97": "LeftHandThumb2_beta", "98": "LeftHandThumb2_gamma", "99": "LeftHandThumb1_alpha", "100": "LeftHandThumb1_beta", "101": "LeftHandThumb1_gamma",
        "102": "LeftHand_alpha", "103": "LeftHand_beta", "104": "LeftHand_gamma", "105": "LeftForeArm_alpha", "106": "LeftForeArm_beta", "107": "LeftForeArm_gamma", "108": "LeftArm_alpha", "109": "LeftArm_beta", "110": "LeftArm_gamma", "111": "LeftShoulder_alpha", "112": "LeftShoulder_beta", "113": "LeftShoulder_gamma",
        "114": "Head_alpha", "115": "Head_beta", "116": "Head_gamma", "117": "Neck_alpha", "118": "Neck_beta", "119": "Neck_gamma", "120": "Spine1_alpha", "121": "Spine1_beta", "122": "Spine1_gamma", "123": "Spine_alpha", "124": "Spine_beta", "125": "Spine_gamma",
        "126": "Hips_alpha", "127": "Hips_beta", "128": "Hips_gamma"
        }
        # "129": "Hips_Xposition", "130": "Hips_Yposition", "131": "Hips_Zposition", "132": "Hips_Xposition", "133": "Hips_Yposition", "134": "Hips_Zposition"}

    r_hand_indices, l_hand_indices, body_indices = [], [], []
    for k, val in joint_ids_mocap.items():
        if "RightHand" in val:
            r_hand_indices.append(int(k))  # 48
        elif "LeftHand" in val:
            l_hand_indices.append(int(k))  # 48
        else:
            # print(f"Adding to body indices: {k}, {val}")
            body_indices.append(int(k))
    wrists = [int(key) for key, value in joint_ids_mocap.items() if value in ["RightHand_alpha", "RightHand_beta", "RightHand_gamma", "LeftHand_alpha", "LeftHand_beta", "LeftHand_gamma"]]
    print(f"Wrists indices: {wrists}")
    body_indices.extend(wrists)  # 39 

    # print(f"mocap r hand indices: {len(r_hand_indices)},\n body indices: {len(body_indices)}, l hand indices: {len(l_hand_indices)}")
    out_size_body = len(body_indices) 
    out_size_r_hand = len(r_hand_indices)
    out_size_l_hand = len(l_hand_indices)

    assert out_size_l_hand == out_size_r_hand, "Left and right hand should have the same number of features."

    return body_indices, r_hand_indices, l_hand_indices 


def separate_body_hands(y, body_indices, r_hand_indices, l_hand_indices):
    """
    Split body and hand features in the input tensor.
    Args:
        y (torch.Tensor): Input tensor of shape (batch_size, n_feats, sequence_length).
        out_size (int): Total number of features (n_feats).
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: body features and hand features tensors.
    """
    # torch.Size([1, 56, 90]), out_
    # size: 56 // torch.Size([1, 129, 90]), out_size: 129
    # [B, n_feats, seq_len]
    
    y_body = y[:, body_indices, :]
    y_r_hand = y[:, r_hand_indices, :]
    y_l_hand = y[:, l_hand_indices, :]
    return y_body, y_r_hand, y_l_hand


def concatenate_body_hands(y_body, y_r_hand, y_l_hand, body_indices, r_hand_indices, l_hand_indices):
    """
    Concatenate body and hand features into a single tensor.
    # Inference: Shape of decoder_outputs_body: torch.Size([1, 42, 90]), decoder_outputs_hands: torch.Size([1, 94, 90])
    Args:
        y_body (torch.Tensor): Body features tensor of shape (batch_size, body_feats, sequence_length).
        y_hands (torch.Tensor): Hand features tensor of shape (batch_size, hand_feats, sequence_length).
    Returns:
        torch.Tensor: Concatenated tensor of shape (batch_size, total_feats, sequence_length).
    """
    b = y_body.shape[0]
    seq_len = y_body.shape[2]
    out_size = y_body.shape[1] + y_r_hand.shape[1] + y_l_hand.shape[1] - 2*3  # -2 because wrists are counted in body and hands both
    print(f"Concatenate body and hands: batch_size: {b}, out_size: {out_size}, seq_len: {seq_len}")
    
    concatenated_output = torch.zeros(b, out_size, seq_len, device=y_body.device)

    concatenated_output[:, r_hand_indices, :] = y_r_hand
    concatenated_output[:, l_hand_indices, :] = y_l_hand
    concatenated_output[:, body_indices, :] = y_body  # take wrists values from the body

    return concatenated_output


class ConditionProjector(nn.Module):
    """Down-projects conditioning embeddings from STS-Net before concatenation with motion data."""
    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        mid_dim = (in_dim + out_dim) // 2  # e.g. 288 for 512 -> 64
        
        self.proj = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim),  # stabilises scale before concat
        )
    
    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, n_frames, 512)  ->  (B, n_frames, out_dim)
        return self.proj(cond)


class UnconditionalBodyGeneration(BaseLightningClass):
    def __init__(
        self,
        # n_vocab,
        #n_spks,
        #spk_emb_dim,
        n_feats,
        # encoder,
        decoder,
        cfm,
        # data_statistics,
        out_size,
        streams=None,
        cascaded=None,
        bicrossattention=None,
        kmeans=None,
        conditional=None,
        kmeans_three_d=None,
        codes=None,
        smplx=None,
        latents=None,
        label_dim=256,
        optimizer=None,
        scheduler=None,
        # --- Matcha duration module (off by default) ---
        use_duration=False,
        use_contact_loss=False,
        contact_loss=None,
        encoder=None,
        prior_loss=True,
        use_precomputed_durations=True,
        # --- face expression stream (off by default) ---
        use_face=False,
        face_dim=10,
        # Width of an optional face-specific tail on every segment token, after
        # body|dom|nondom; comes from ${data.face_label_dim}. 0 -> no tail (every
        # run before the pose-token experiment), and checkpoints saved without this
        # hparam load with the default.
        face_label_dim=0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # self.n_vocab = n_vocab
        #self.n_spks = n_spks
        #self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.streams = streams
        self.cascaded = cascaded
        self.bicrossattention = bicrossattention
        self.kmeans = kmeans
        self.conditional = conditional
        self.codes = codes
        self.latents = latents
        self.smplx = smplx
        # Width of the STS-Net conditioning embeddings; comes from ${data.label_dim}.
        self.label_dim = label_dim
        # self.three_d = three_d

        self.kmeans_three_d = kmeans_three_d
        # Face expression stream: a fourth CFM decoder over the 10 FLAME
        # expression coefficients, conditioned off the same fused label vector the
        # manual streams read. Orthogonal to `streams`, which stays 3 -- it counts
        # the MANUAL streams (body / dom / nondom) and every branch below keys off
        # it. Face is an add-on to the conditional-smplx branch, not a new value of
        # `streams`, so the legacy branches (cascaded / bicrossattention / kmeans /
        # codes) are untouched and old checkpoints still load with use_face=False.
        self.use_face = bool(use_face)
        self.face_dim = int(face_dim)
        # The face reads the WHOLE fused segment vector -- all three thirds
        # concatenated, 3 * label_dim. The fused labels carry no face-specific
        # third, and a non-manual is not a function of the body alone: mouthings
        # and brow movement co-articulate with what both hands are doing, so the
        # face decoder gets everything the annotation has rather than one slice.
        #
        # face_label_dim > 0 appends a face-specific vector to every token (the
        # pose-token run: the segment's median [expression | jaw 6D]). It is read by
        # the segment encoder with the rest of the token and added to the FACE
        # decoder's conditioning only -- the manual streams still see their own
        # label_dim slice. It exists at segment resolution only, so it needs the
        # duration path, which expands tokens to frames.
        self.face_label_dim = int(face_label_dim)
        if self.face_label_dim and not (use_duration and use_face):
            raise ValueError(
                f"face_label_dim={self.face_label_dim} needs use_duration=True and "
                f"use_face=True (got use_duration={use_duration}, use_face={use_face}): "
                "the face token is segment-level and only the face decoder reads it.")
        self.face_cond_width = 3 * self.label_dim + self.face_label_dim
        # print(f"Kmeans in init {self.kmeans}")
        # Duration module: predicts one duration per annotated segment, expands the
        # segment tokens to frame rate through an alignment map, and feeds the
        # decoders an extra prior `mu` on top of the usual conditioning.
        # Geometric self-contact term. Everything it needs is built once here so the
        # training step only does the forward pass. Off by default: with the flag
        # False nothing below runs and the module is the setup that produced 21824369.
        self.use_contact_loss = bool(use_contact_loss)
        self.contact_cfg = contact_loss
        self._contact_term = None
        if self.use_contact_loss:
            if contact_loss is None:
                raise ValueError("use_contact_loss=True needs a contact_loss config block")
            from src_mask_fast.models.components.contact_loss import SparseContactHead
            c = contact_loss
            self.contact_head = SparseContactHead(
                model_path=c.smplx_model_path,
                contacts_h5=c.contacts_h5,
                keep_pairs=list(c.keep_pairs) if c.keep_pairs is not None else None,
                n_anchors=int(c.n_anchors),
                softmin_tau=float(c.softmin_tau),
                d_on=float(c.d_on),
                d_softness=float(c.d_softness),
            )
            # x1_hat is z-scored; SMPL-X needs metres. Same stats file the dataset uses.
            import json as _json
            with open(c.norm_stats_path) as _f:
                _st = _json.load(_f)
            for _key, _name in (("smplx_mesh_body_pose", "body"),
                                ("smplx_mesh_right_hand_pose", "rhand"),
                                ("smplx_mesh_left_hand_pose", "lhand")):
                self.register_buffer(f"_nm_{_name}",
                                     torch.tensor(_st[_key]["mean"], dtype=torch.float32))
                self.register_buffer(f"_ns_{_name}",
                                     torch.tensor(_st[_key]["std"], dtype=torch.float32).clamp_min(1e-8))
            # Column selector into the full 61-pair targets emitted by the dataset.
            self.register_buffer("_keep_idx",
                                 torch.as_tensor(self.contact_head.keep_pairs, dtype=torch.long))
            # Running magnitude of each loss, so the configured mix is a real ratio
            # of contributions rather than a ratio of incomparable numbers.
            self.register_buffer("_ema_diff", torch.tensor(0.0))
            self.register_buffer("_ema_contact", torch.tensor(0.0))

        if self.use_face and not (self.streams == 3 and self.conditional and self.smplx
                                  and not self.cascaded and not self.bicrossattention
                                  and not self.kmeans and not self.codes):
            raise ValueError(
                "use_face=True is only wired for the conditional 3-stream smplx "
                "configuration (streams=3, conditional=True, smplx=True, and "
                "cascaded/bicrossattention/kmeans/codes all False). Got "
                f"streams={self.streams}, conditional={self.conditional}, "
                f"smplx={self.smplx}, cascaded={self.cascaded}, "
                f"bicrossattention={self.bicrossattention}, kmeans={self.kmeans}, "
                f"codes={self.codes}."
            )

        self.use_duration = bool(use_duration)
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations
        if self.use_duration and not (self.streams == 3 and self.conditional and self.smplx
                                      and not self.cascaded and not self.bicrossattention
                                      and not self.kmeans and not self.codes):
            raise ValueError(
                "use_duration=True is only wired for the conditional 3-stream smplx "
                "configuration (streams=3, conditional=True, smplx=True, and "
                "cascaded/bicrossattention/kmeans/codes all False). Got "
                f"streams={self.streams}, conditional={self.conditional}, "
                f"smplx={self.smplx}, cascaded={self.cascaded}, "
                f"bicrossattention={self.bicrossattention}, kmeans={self.kmeans}, "
                f"codes={self.codes}."
            )

        # if n_spks > 1:
        #     self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        #self.encoder = TextEncoder(
        #    encoder.encoder_type,
        #    encoder.encoder_params,
        #    encoder.duration_predictor_params,
        #    n_vocab,
        #    n_spks,
        #    spk_emb_dim,
        #)
        
        # TSG2 body: batch_size, sequence_length, features - ([28, 120, 42])

        if self.streams == 1:
            self.decoder = CFM(
                in_channels=self.n_feats, # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=self.out_size,
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
        elif self.streams == 3 and not self.cascaded and not self.bicrossattention and not self.kmeans and not self.conditional:
            print("===Unconditional===")
            if self.smplx:
                # Shapes: smplx_mesh_body_pose torch.Size([8, 90, 63]), smplx_mesh_left_hand_pose torch.Size([8, 90, 45]),
                # smplx_mesh_global_orient torch.Size([8, 90, 3]), smplx_mesh_transl torch.Size([8, 90, 3])
                out_channels_body, out_channels_hand = 126, 90 # 63, 45
                in_channels_body, in_channels_hand = 126, 90 # 63, 45
            else:
                self.body_indices, self.r_hand_indices, self.l_hand_indices = get_joint_indices()
                out_channels_body, out_channels_hand = len(self.body_indices), len(self.r_hand_indices)
                in_channels_body, in_channels_hand = 39, 48
            self.decoder_body = CFM(
                in_channels=in_channels_body,  # len(self.body_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=out_channels_body,
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_r_hand = CFM(
                in_channels=in_channels_hand, # 80 <- This value works for non-cascaded setting,  # len(self.r_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=out_channels_hand,
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_l_hand = CFM(
                in_channels=in_channels_hand, # 80<- This value works for non-cascaded setting,   # len(self.l_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=out_channels_hand,
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            # Automatic optimization (src_mask set automatic_optimization=False here).
            # One optimizer over all params replaces the per-decoder split; see
            # configure_optimizers + configure_gradient_clipping in baselightningmodule.py.
        elif self.streams == 3 and not self.cascaded and not self.bicrossattention and not self.kmeans and self.conditional and not self.codes:
            print("Not cascaded. Conditioned on 2 betas")
            if self.smplx:
                # that's concatenation not addition clearly
                # Pose is concatenated with the conditioning along the channel dim:
                #   label_dim 256 -> 382 / 346    label_dim 512 -> 638 / 602
                out_channels_body, out_channels_hand = 126, 90  # 63, 45  # 126, 90 on 6D # 363, 345 concat # 63, 45
                in_channels_body = out_channels_body + self.label_dim
                in_channels_hand = out_channels_hand + self.label_dim
                # Face: 10 FLAME expression coefficients, conditioned off the
                # fused label vector exactly the way the manual streams are.
                out_channels_face = self.face_dim if self.use_face else None
                in_channels_face = (
                    out_channels_face + self.face_cond_width) if self.use_face else None
                if self.use_duration:
                    # The aligned prior mu is packed next to x alongside cond, so
                    # each stream grows by its own feature width. NOTE: this changes
                    # in_channels, so checkpoints trained with use_duration=False
                    # cannot be loaded with it on (and vice versa).
                    in_channels_body += out_channels_body
                    in_channels_hand += out_channels_hand
                    if self.use_face:
                        in_channels_face += out_channels_face
                print(f"Conditional smplx: label_dim={self.label_dim} -> "
                      f"in_channels body/hand = {in_channels_body}/{in_channels_hand} "
                      f"(use_duration={self.use_duration})")
                if self.use_face:
                    print(f"Face stream ON: {out_channels_face} expression coeffs, "
                          f"cond = 3 x {self.label_dim} fused + {self.face_label_dim} face token "
                          f"({self.face_cond_width}) -> "
                          f"in_channels_face = {in_channels_face}")
                if self.use_duration:
                    assert encoder is not None, (
                        "use_duration=True needs an `encoder` config "
                        "(configs/model/encoder/default.yaml)"
                    )
                    self.encoder = SegmentEncoder(
                        encoder_params=encoder.encoder_params,
                        duration_predictor_params=encoder.duration_predictor_params,
                        label_dim=self.label_dim,
                        out_channels_body=out_channels_body,
                        out_channels_hand=out_channels_hand,
                        # None when the face stream is off -> no face prior head is
                        # built and the state dict is unchanged.
                        out_channels_face=out_channels_face,
                        # The switch. Defaults True for checkpoints saved before the
                        # flag existed, whose hparams have no `use_is_sign` and whose
                        # emb is the wide one.
                        use_is_sign=bool(getattr(encoder, "use_is_sign", True)),
                        face_label_dim=self.face_label_dim,
                    )
                # self.cond_proj = ConditionProjector(in_dim=512, out_dim=64)
            else:
                self.body_indices, self.r_hand_indices, self.l_hand_indices = get_joint_indices()
                out_channels_body, out_channels_hand = len(self.body_indices), len(self.r_hand_indices)
                in_channels_body, in_channels_hand = 65, 80
                # Face is smplx-only (the expression coefficients are a FLAME
                # parameter); the guard in __init__ already requires smplx=True.
                out_channels_face = in_channels_face = None
            self.decoder_body = CFM(
                in_channels=in_channels_body,  # len(self.body_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=out_channels_body,
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_r_hand = CFM(
                in_channels=in_channels_hand, # 80 <- This value works for non-cascaded setting,  # len(self.r_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=out_channels_hand,
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_l_hand = CFM(
                in_channels=in_channels_hand, # 80<- This value works for non-cascaded setting,   # len(self.l_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=out_channels_hand,
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            if self.use_face:
                # Same CFM/decoder hyperparameters as the motion streams. The
                # decoder is a 1D U-Net over time, so a 10-channel stream is fine
                # as long as the width survives the downsampling -- it is the TIME
                # axis that gets halved per level, not the channel axis.
                self.decoder_face = CFM(
                    in_channels=in_channels_face,
                    out_channels=out_channels_face,
                    cfm_params=cfm,
                    decoder_params=decoder,
                )
            # Automatic optimization (src_mask set automatic_optimization=False here).
            # One optimizer over all params replaces the per-decoder split; see
            # configure_optimizers + configure_gradient_clipping in baselightningmodule.py.
        elif self.streams == 3 and self.cascaded and not self.bicrossattention:
            self.body_indices, self.r_hand_indices, self.l_hand_indices = get_joint_indices()
            self.decoder_body = CFM(
                in_channels=65,  # len(self.body_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.body_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_r_hand = CFM(
                in_channels=106, # 80 <- This value works for non-cascaded setting,  # len(self.r_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.r_hand_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_l_hand = CFM(
                in_channels=138, # 80<- This value works for non-cascaded setting,   # len(self.l_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.l_hand_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            # Automatic optimization (src_mask set automatic_optimization=False here).
            # One optimizer over all params replaces the per-decoder split; see
            # configure_optimizers + configure_gradient_clipping in baselightningmodule.py.
        elif self.streams == 3 and self.bicrossattention and not self.cascaded:
            self.body_indices, self.r_hand_indices, self.l_hand_indices = get_joint_indices()
            # weight, bias: mat1 and mat2 shapes cannot be multiplied (624x28 and 512x512)
            self.joint_cross_attn = BidirectionalCrossAttention(
                dim         = 39,   # body feature dim (28)
                heads       = 2,
                dim_head    = 32,   # so inner_dim = 64 etc (14)
                context_dim = 96,   # both hands feature dim (28)
            )
            self.decoder_body = CFM(
                in_channels=65,  # len(self.body_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.body_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_r_hand = CFM(
                in_channels=80, # 80 <- This value works for non-cascaded setting,  # len(self.r_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.r_hand_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_l_hand = CFM(
                in_channels=80, # 80<- This value works for non-cascaded setting,   # len(self.l_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.l_hand_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            # Automatic optimization (src_mask set automatic_optimization=False here).
            # One optimizer over all params replaces the per-decoder split; see
            # configure_optimizers + configure_gradient_clipping in baselightningmodule.py.
        elif self.streams == 3 and not self.bicrossattention and not self.cascaded and self.kmeans and not self.kmeans_three_d and not self.codes:
            print("K-Means")
            self.body_indices, self.r_hand_indices, self.l_hand_indices = get_joint_indices()
            self.decoder_body = CFM(
                in_channels=65, # 40,  # len(self.body_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.body_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_r_hand = CFM(
                in_channels=80, # 80 <- This value works for non-cascaded setting,  # len(self.r_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.r_hand_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_l_hand = CFM(
                in_channels=80, # 80<- This value works for non-cascaded setting,   # len(self.l_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.l_hand_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            # Automatic optimization (src_mask set automatic_optimization=False here).
            # One optimizer over all params replaces the per-decoder split; see
            # configure_optimizers + configure_gradient_clipping in baselightningmodule.py.
        elif self.streams == 3 and not self.bicrossattention and not self.cascaded and self.kmeans and self.kmeans_three_d and not self.codes:
            print("K-Means trained on 3D")
            self.body_indices, self.r_hand_indices, self.l_hand_indices = get_joint_indices()
            self.decoder_body = CFM(
                in_channels=78, # 40,  # len(self.body_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.body_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_r_hand = CFM(
                in_channels=96, # 80 <- This value works for non-cascaded setting,  # len(self.r_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.r_hand_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_l_hand = CFM(
                in_channels=96, # 80<- This value works for non-cascaded setting,   # len(self.l_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.l_hand_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            # Automatic optimization (src_mask set automatic_optimization=False here).
            # One optimizer over all params replaces the per-decoder split; see
            # configure_optimizers + configure_gradient_clipping in baselightningmodule.py.
        # self.update_data_statistics(data_statistics)
        elif self.streams == 3 and not self.bicrossattention and not self.cascaded and not self.kmeans and not self.kmeans_three_d and self.codes:
            in_channels_body, in_channels_hands = 43, 52
            if self.latents:
                in_channels_body, in_channels_hands = 295, 304
            print("Model conditioned on RVQ simple codes")
            self.body_indices, self.r_hand_indices, self.l_hand_indices = get_joint_indices()
            self.decoder_body = CFM(
                in_channels=in_channels_body, # 40,  # len(self.body_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.body_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_r_hand = CFM(
                in_channels=in_channels_hands, # 80 <- This value works for non-cascaded setting,  # len(self.r_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.r_hand_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            self.decoder_l_hand = CFM(
                in_channels=in_channels_hands, # 80<- This value works for non-cascaded setting,   # len(self.l_hand_indices), # 3rd dimension of the batch    # 2 * encoder.encoder_params.n_feats,
                out_channels=len(self.l_hand_indices),
                cfm_params=cfm,
                decoder_params=decoder,
                #n_spks=n_spks,
                #spk_emb_dim=spk_emb_dim,
            )
            # Automatic optimization (src_mask set automatic_optimization=False here).
            # One optimizer over all params replaces the per-decoder split; see
            # configure_optimizers + configure_gradient_clipping in baselightningmodule.py.

    @torch.inference_mode()
    # ------------------------------------------------------------------ #
    # Duration module (Matcha / Grad-TTS), active only when use_duration=True
    # ------------------------------------------------------------------ #

    def _split_tokens(self, x):
        """(B, n_seg, 3*label_dim [+ face_label_dim]) -> the body / dom / nondom thirds.

        Bounded at 3*label_dim so an optional face tail never leaks into nondom.
        """
        d = self.label_dim
        if x.shape[-1] != 3 * d + self.face_label_dim:
            raise ValueError(
                f"segment tokens are {x.shape[-1]}-d, expected 3 x label_dim {d} + "
                f"face_label_dim {self.face_label_dim} = {3 * d + self.face_label_dim}; "
                "data.label_dim / data.face_label_dim do not match the labels file.")
        return x[..., :d], x[..., d:2 * d], x[..., 2 * d:3 * d]

    def _face_token(self, x):
        """(B, n_seg, face_label_dim) face tail of the tokens, or None without one."""
        return x[..., 3 * self.label_dim:] if self.face_label_dim else None

    @staticmethod
    def _face_cond(tok_body, tok_dom, tok_nondom, tok_face=None):
        """Conditioning for the face stream: the whole fused label vector.

        The fused segment token is (body | dom | nondom) and carries no
        face-specific third, so the face decoder reads all three concatenated
        (3 * label_dim) rather than one slice of them, plus the face tail when the
        tokens have one (face_label_dim). Inputs and output are all
        (..., C)-last, so this works at both segment and frame resolution.
        """
        parts = (tok_body, tok_dom, tok_nondom) + (() if tok_face is None else (tok_face,))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _expand(attn, seq):
        """Segment rate -> frame rate through the alignment map.

        attn: (B, n_seg, T), seq: (B, n_seg, C)  ->  (B, T, C)
        """
        return torch.matmul(attn.transpose(1, 2), seq)

    def _mas_attn(self, mu_x_all, y_all, attn_mask):
        """Monotonic Alignment Search over the three streams jointly."""
        with torch.no_grad():
            n_feats = mu_x_all.shape[1]
            const = -0.5 * math.log(2 * math.pi) * n_feats
            factor = -0.5 * torch.ones(mu_x_all.shape, dtype=mu_x_all.dtype, device=mu_x_all.device)
            y_square = torch.matmul(factor.transpose(1, 2), y_all**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x_all).transpose(1, 2), y_all)
            mu_square = torch.sum(factor * (mu_x_all**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask)
        return attn.detach()

    def _duration_forward(self, x, x_lengths, durations, y_mask, y_streams, is_sign=None):
        """Encode segments, align them to frames, and score the durations.

        Args:
            x:         (B, n_seg, 3*label_dim) pooled segment tokens
            x_lengths: (B,) real segments per item
            durations: (B, n_seg) annotated frame counts, or None with MAS
            y_mask:    (B, 1, T) float
            y_streams: (y_body, y_r_hand, y_l_hand), each (B, feat, T)

        Returns:
            cond_frames: (body, dom, nondom, face) each (B, T, label_dim), except
                face which is (B, T, 3*label_dim) -- the three thirds concatenated.
                None unless use_face.
            mu_y:        (body, r_hand, l_hand, face) each (B, feat, T); face None
                unless use_face
            dur_loss:    scalar
            attn:        (B, n_seg, T)
        """
        (mu_x_body, mu_x_r, mu_x_l, mu_x_face), logw, x_mask = self.encoder(x, x_lengths, is_sign)

        attn_mask = (x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)).squeeze(1)  # (B, n_seg, T)

        if self.use_precomputed_durations:
            assert durations is not None, (
                "use_precomputed_durations=True but the batch carries no `durations`; "
                "set data.return_tokens=True"
            )
            attn = generate_path(durations.to(attn_mask.dtype), attn_mask)
        else:
            attn = self._mas_attn(
                torch.cat((mu_x_body, mu_x_r, mu_x_l), dim=1),
                torch.cat(y_streams, dim=1),
                attn_mask,
            )

        # Score the predicted log-durations against the alignment actually used.
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        tok_body, tok_dom, tok_nondom = self._split_tokens(x)
        # Expanded at segment resolution then aligned, exactly like the other three,
        # so the face conditioning is frame-aligned by the same attn map.
        tok_face = (self._face_cond(tok_body, tok_dom, tok_nondom, self._face_token(x))
                    if self.use_face else None)
        cond_frames = (
            self._expand(attn, tok_body),
            self._expand(attn, tok_dom),
            self._expand(attn, tok_nondom),
            self._expand(attn, tok_face) if tok_face is not None else None,
        )
        mu_y = (
            self._expand(attn, mu_x_body.transpose(1, 2)).transpose(1, 2),
            self._expand(attn, mu_x_r.transpose(1, 2)).transpose(1, 2),
            self._expand(attn, mu_x_l.transpose(1, 2)).transpose(1, 2),
            self._expand(attn, mu_x_face.transpose(1, 2)).transpose(1, 2)
            if mu_x_face is not None else None,
        )
        return cond_frames, mu_y, dur_loss, attn

    @staticmethod
    def _prior_loss(y_stream, mu_stream, y_mask):
        """Negative log-likelihood of y under the aligned prior (Matcha eq. 5)."""
        n_feats = y_stream.shape[1]
        loss = torch.sum(0.5 * ((y_stream - mu_stream) ** 2 + math.log(2 * math.pi)) * y_mask)
        return loss / (torch.sum(y_mask) * n_feats)

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0, length_scale=1.0,
                   device="cuda", x0=None, is_sign=None):
        """Generate motion from a sequence of segment tokens, timing included.

        This is the duration-module counterpart to `inference`: instead of being
        told how many frames to produce, the model predicts a duration per segment,
        builds the alignment map from those durations, and expands the tokens to
        frame rate through it.

        Args:
            x:            (B, n_seg, 3*label_dim) pooled segment tokens
            x_lengths:    (B,) real segments per item
            n_timesteps:  reverse-diffusion steps
            length_scale: >1 slows the signing down, <1 speeds it up
            x0:           optional initial noise, forwarded to the decoders

        Returns:
            (body, r_hand, l_hand, concatenated) with body/hands as (B, T, feat),
            matching `inference`, plus `attn` and `y_lengths` on the returned dict.
        """
        if not self.use_duration:
            raise RuntimeError(
                "synthesise() requires use_duration=True; use inference() for the "
                "fixed/masked-length path."
            )
        t = dt.datetime.now()

        x = x.to(device)
        x_lengths = x_lengths.to(device)

        (mu_x_body, mu_x_r, mu_x_l, mu_x_face), logw, x_mask = self.encoder(
            x, x_lengths, is_sign.to(device) if is_sign is not None else None)

        # Predicted durations -> total length
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = (x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)).squeeze(1)  # (B, n_seg, T)
        attn = generate_path(w_ceil.squeeze(1), attn_mask)

        tok_body, tok_dom, tok_nondom = self._split_tokens(x)
        cond_body = self._expand(attn, tok_body).permute(0, 2, 1)      # (B, label_dim, T)
        cond_r_hand = self._expand(attn, tok_dom).permute(0, 2, 1)
        cond_l_hand = self._expand(attn, tok_nondom).permute(0, 2, 1)

        mu_body = self._expand(attn, mu_x_body.transpose(1, 2)).transpose(1, 2)
        mu_r_hand = self._expand(attn, mu_x_r.transpose(1, 2)).transpose(1, 2)
        mu_l_hand = self._expand(attn, mu_x_l.transpose(1, 2)).transpose(1, 2)

        if self.use_face:
            cond_face = self._face_cond(tok_body, tok_dom, tok_nondom, self._face_token(x))
            cond_face = self._expand(attn, cond_face).permute(0, 2, 1)
            mu_face = self._expand(attn, mu_x_face.transpose(1, 2)).transpose(1, 2)

        batch_size = x.shape[0]
        common = dict(
            batch_size=batch_size,
            sequence_length=y_max_length_,
            device=device,
            n_timesteps=n_timesteps,
            x0=x0,
            mask=y_mask,
            temperature=temperature,
        )
        decoder_outputs_body = self.decoder_body(cond=cond_body, mu=mu_body, **common).cpu()
        decoder_outputs_r_hand = self.decoder_r_hand(cond=cond_r_hand, mu=mu_r_hand, **common).cpu()
        decoder_outputs_l_hand = self.decoder_l_hand(cond=cond_l_hand, mu=mu_l_hand, **common).cpu()
        decoder_outputs_face = (
            self.decoder_face(cond=cond_face, mu=mu_face, **common).cpu()
            if self.use_face else None
        )

        # (B, feat, T) -> (B, T, feat), trimmed to the predicted length and masked,
        # matching what inference() returns for the smplx conditional branch.
        mask_cpu = y_mask.permute(0, 2, 1).cpu()[:, :y_max_length]
        body = decoder_outputs_body.permute(0, 2, 1)[:, :y_max_length] * mask_cpu
        r_hand = decoder_outputs_r_hand.permute(0, 2, 1)[:, :y_max_length] * mask_cpu
        l_hand = decoder_outputs_l_hand.permute(0, 2, 1)[:, :y_max_length] * mask_cpu
        # Face is appended LAST, matching the layout get_losses builds for y, and
        # only when the stream is on -- so with use_face=False the returned width
        # is the unchanged 306 and existing synthesis scripts keep working.
        face = None
        if self.use_face:
            face = decoder_outputs_face.permute(0, 2, 1)[:, :y_max_length] * mask_cpu
            decoder_outputs = torch.cat((body, r_hand, l_hand, face), dim=2)
        else:
            decoder_outputs = torch.cat((body, r_hand, l_hand), dim=2)

        rtf = (dt.datetime.now() - t).total_seconds()
        print(f"synthesise: predicted lengths {y_lengths.tolist()} "
              f"(padded to {y_max_length_}) in {rtf:.2f}s")

        # `face` is None when the stream is off; the 6-tuple shape is unchanged for
        # every existing caller, which unpacks the first four positionally.
        self.last_face = face
        return body, r_hand, l_hand, decoder_outputs, attn[:, :, :y_max_length], y_lengths

    # ---- geometric self-contact loss -------------------------------------
    def _sample_frames(self, soft_dist, y_mask, n):
        """(B, n) frame indices per item.

        stratified -> half from frames where any kept pair is active, half from
        frames where none is. Contacts are ~1.6% of (frame, pair) cells, so a
        uniform draw is nearly all negatives and the cheapest way to cut the loss
        would be to never predict contact. Shortfall in either pool is topped up
        from the other so the count per item stays fixed at n.
        """
        cfg = self.contact_cfg
        B, T, _ = soft_dist.shape
        dev = soft_dist.device
        valid = y_mask.squeeze(1).bool()
        active = (soft_dist.index_select(-1, self._keep_idx)
                  > float(cfg.contact_active_threshold)).any(-1) & valid
        idle = (~active) & valid
        half = n // 2
        out = []
        for b in range(B):
            pool = torch.nonzero(valid[b], as_tuple=False).flatten()
            if pool.numel() == 0:                      # fully padded item
                out.append(torch.zeros(n, dtype=torch.long, device=dev)); continue
            if str(cfg.frame_sampling) != "stratified":
                sel = pool[torch.randint(pool.numel(), (n,), device=dev)]
            else:
                a = torch.nonzero(active[b], as_tuple=False).flatten()
                i = torch.nonzero(idle[b], as_tuple=False).flatten()
                na = min(half, a.numel())
                ni = min(n - na, i.numel())
                na = min(n - ni, a.numel())            # re-balance if idle was short
                parts = []
                if na: parts.append(a[torch.randperm(a.numel(), device=dev)[:na]])
                if ni: parts.append(i[torch.randperm(i.numel(), device=dev)[:ni]])
                sel = torch.cat(parts) if parts else pool[:0]
                if sel.numel() < n:                    # top up from anything valid
                    sel = torch.cat([sel, pool[torch.randint(pool.numel(), (n - sel.numel(),), device=dev)]])
            out.append(sel[:n])
        return torch.stack(out)

    def _contact_loss(self, extras, betas, soft_dist, y_mask):
        """BCE between predicted and target per-pair soft-distance scores.

        extras[stream] = (y_t, v_hat) from compute_loss(return_extras=True);
        extras["t"] is the shared (B,1,1) timestep.
        """
        cfg = self.contact_cfg

        # The batch's per-frame tensors are padded by the collate to max(lengths),
        # but forward() rounds y up to a U-Net-friendly multiple (fix_len_compatibility)
        # AFTER that, so y_mask can be a couple of frames longer than soft_dist and
        # betas -- measured 288 vs 286. Right-pad them to agree; the extra frames are
        # masked out, so _sample_frames never draws them and the zeros are never read.
        T = y_mask.shape[-1]

        def _fit(x):
            if x.shape[1] == T:
                return x
            if x.shape[1] > T:
                return x[:, :T]
            pad = x.new_zeros(x.shape[0], T - x.shape[1], *x.shape[2:])
            return torch.cat([x, pad], dim=1)

        soft_dist = _fit(soft_dist)
        betas = _fit(betas)

        t = extras["t"]                                  # (B,1,1)
        tf = t.view(-1)
        keep = tf > float(cfg.t_gate)
        if not bool(keep.any()):
            # x1_hat is mostly noise at small t, so a batch that gates out entirely
            # contributes nothing rather than contributing garbage.
            return None
        w = (tf if bool(cfg.t_weight) else torch.ones_like(tf)) * keep.to(tf.dtype)

        n = int(cfg.n_frames)
        idx = self._sample_frames(soft_dist, y_mask, n)   # (B, n)

        def stream(name):
            y_t, v = extras[name]                         # (B, C, T)
            x1 = y_t + (1.0 - t) * v                      # exact as sigma_min -> 0
            mean = getattr(self, f"_nm_{name}").view(1, -1, 1)
            std = getattr(self, f"_ns_{name}").view(1, -1, 1)
            x1 = x1 * std + mean                          # z-scored -> metres/6D
            g = torch.gather(x1, 2, idx.unsqueeze(1).expand(-1, x1.shape[1], -1))
            return g.permute(0, 2, 1).reshape(-1, x1.shape[1])   # (B*n, C)

        body = stream("body").view(-1, 21, 6)
        rh = stream("rhand").view(-1, 15, 6)
        lh = stream("lhand").view(-1, 15, 6)
        b_sel = torch.gather(betas, 1, idx.unsqueeze(-1).expand(-1, -1, betas.shape[-1]))
        pred = self.contact_head(body, rh, lh, b_sel.reshape(-1, betas.shape[-1]))

        tgt = torch.gather(soft_dist, 1,
                           idx.unsqueeze(-1).expand(-1, -1, soft_dist.shape[-1]))
        tgt = tgt.index_select(-1, self._keep_idx).reshape(-1, pred.shape[1])

        per = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), tgt.clamp(0, 1),
                                     reduction="none").mean(dim=1)      # (B*n,)
        wf = w.unsqueeze(1).expand(-1, n).reshape(-1)
        denom = wf.sum().clamp_min(1e-8)
        return (per * wf).sum() / denom

    def forward(self, y, y_lengths, cond=None, x=None, x_lengths=None, durations=None,
                betas=None, soft_dist=None, is_sign=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        # if self.n_spks > 1:
            # Get speaker embedding
        #    spks = self.spk_emb(spks)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        # mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        # y_max_length = y.shape[-1]
        # temporary fix for fixed sequence length
        # y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(device=y.device)  #.to(x_mask)
        # attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # if self.use_precomputed_durations:
        #    attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        # else:
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            #with torch.no_grad():
            #    const = -0.5 * math.log(2 * math.pi) * self.n_feats
            #    factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            #    y_square = torch.matmul(factor.transpose(1, 2), y**2)
            #    y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            #    mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            #    log_prior = y_square - y_mu_double + mu_square + const

            #    attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            #    attn = attn.detach()  # b, t_text, T_mel

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # refered to as prior loss in the paper
        # logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        # dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
        #   - Do not need this hack for Matcha-TTS, but it works with it as well
        """if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask"""

        # Align encoded text with mel-spectrogram and get mu_y segment
        # mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        # mu_y = mu_y.transpose(1, 2)
        
        # print(f"Shape of y_mask: {y_mask.shape}") # Shape of y_mask: torch.Size([10, 1, 90]), all True

        y_max_length = y.shape[1]                                          # T
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1)      # (B, 1, T)
        y_mask = y_mask.to(device=y.device)

        # ONE flow-matching timestep per sample, shared by every stream below.
        #
        # Deviates from src_mask, where each decoder_*.compute_loss() ran its own
        # torch.rand([b,1,1]) and the three streams therefore sat at different
        # points on the noise path within a single step (measured: body t=0.644,
        # r_hand t=0.776, l_hand t=0.818 for the same sample).
        #
        # That was not a bug for the diffusion loss -- the three decoders are
        # independent flow models coupled only through the conditioning, and t
        # broadcasts over time, so each stream always saw a uniformly noised,
        # temporally coherent sequence. But it makes the streams non-composable
        # mid-denoising: a body at t=0.9 and hands at t=0.2 do not describe one
        # body, so nothing geometric can be computed across them. Inference has
        # always shared t (solve_euler walks one t_span per stream, built
        # identically), so sharing it here also removes a train/inference
        # asymmetry.
        #
        # Only the LEVEL is shared. Each stream still draws its own z inside
        # compute_loss, which it must: the streams have 126/90/90 channels.
        t_shared = torch.rand([y.shape[0], 1, 1], device=y.device, dtype=y.dtype)

        # Compute loss of the decoder
        if self.streams == 1:
            diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask)  # mask=y_mask, mu=mu_y , spks=spks, cond=cond)
            # print(f"Diff loss in UnconditionalBodyGeneration: {diff_loss.item()}")
            """if self.prior_loss:
                prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
                prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
            else:
                prior_loss = 0"""
            return diff_loss  # , attn
        
        # Unconditional model
        elif self.streams == 3 and cond is None:
            ### 3 streams no condition
            # assert y.shape[2] == 90, "y shape[2] should be equal sequence length"
            if self.smplx:
                # Shapes: smplx_mesh_body_pose torch.Size([8, 90, 63]), smplx_mesh_left_hand_pose torch.Size([8, 90, 45]),
                # smplx_mesh_global_orient torch.Size([8, 90, 3]), smplx_mesh_transl torch.Size([8, 90, 3])
                # y_body = torch.cat((y[:, :, :63], y[:,:,63+45+45:]), dim=2).permute(0, 2, 1)
                # b, h = 63, 45
                # print("Training on SMPLX parameters. No conditioning.")
                b, h = 126, 90
                y_body = y[:, :, :b].permute(0, 2, 1)
                # print(f"y_body shape before compute loss: {y_body.shape}")
                # y_body shape before compute loss: torch.Size([16, 69, 90])
                y_r_hand = y[:, :, b:b+h].permute(0, 2, 1)
                y_l_hand = y[:, :, b+h:b+h+h].permute(0, 2, 1)
                # print(f"y_r_hand shape before compute loss: {y_r_hand.shape}, left hand {y_l_hand.shape}")
                # y_r_hand shape before compute loss: torch.Size([16, 45, 90]), left hand torch.Size([16, 45, 90])
                diff_loss_body, _ = self.decoder_body.compute_loss(t=t_shared, x1=y_body, mask=y_mask)
                diff_loss_r_hand, _ = self.decoder_r_hand.compute_loss(t=t_shared, x1=y_r_hand, mask=y_mask)
                diff_loss_l_hand, _ = self.decoder_l_hand.compute_loss(t=t_shared, x1=y_l_hand, mask=y_mask)
            else:
                y_body, y_r_hand, y_l_hand = separate_body_hands(y, self.body_indices, self.r_hand_indices, self.l_hand_indices)
                # print(f"y_body shape before compute loss: {y_body.shape}") [2, 39, 90]

                diff_loss_body, _ = self.decoder_body.compute_loss(t=t_shared, x1=y_body, mask=y_mask)
                diff_loss_r_hand, _ = self.decoder_r_hand.compute_loss(t=t_shared, x1=y_r_hand, mask=y_mask)
                diff_loss_l_hand, _ = self.decoder_l_hand.compute_loss(t=t_shared, x1=y_l_hand, mask=y_mask)
            return diff_loss_body, diff_loss_r_hand, diff_loss_l_hand

        # Conditional model
        elif self.streams == 3 and (cond is not None) and (not self.cascaded) and (not self.bicrossattention):
            ### 3 streams and conditioned on 2D pose
            if self.smplx:
                # Shapes: smplx_mesh_body_pose torch.Size([8, 90, 63]), smplx_mesh_left_hand_pose torch.Size([8, 90, 45]),
                # smplx_mesh_global_orient torch.Size([8, 90, 3]), smplx_mesh_transl torch.Size([8, 90, 3])
                # print(f"y shape before selecting ids: {y.shape}") [32, 90, 153]
                # print(f"cond shape: {cond.shape}") [32, 300, 90]
                b, h = 126, 90
                #cond = cond[:, :2, :]  # on 10 it overfits
                cond_body, cond_r_hand, cond_l_hand = cond
                #cond_body_proj = self.cond_proj(cond_body).permute(0, 2, 1) # [32, 64, 30]
                # print("self.cond_body_proj.shape in forward: ", cond_body_proj.shape) #  torch.Size([32, 30, 64])
                #cond_r_hand_proj = self.cond_proj(cond_r_hand).permute(0, 2, 1)
                #cond_l_hand_proj = self.cond_proj(cond_l_hand).permute(0, 2, 1)
                y_body = y[:, :, :b].permute(0, 2, 1)  # :63
                # print(f"y_body shape before compute loss: {y_body.shape}") # [32, 63, 90]
                y_r_hand = y[:, :, b:b+h].permute(0, 2, 1)  # 63:63+45
                y_l_hand = y[:, :, b+h:b+h+h].permute(0, 2, 1)  # 63+45:63+45+45
                # print(f"y_r_hand shape before compute loss: {y_r_hand.shape}, left hand {y_l_hand.shape}") torch.Size([32, 45, 90])

                # Face rides on the TAIL of y: get_losses concatenates the
                # expression coefficients after the two hands, so the manual
                # slicing above is byte-for-byte unchanged when the face is off.
                y_face = None
                if self.use_face:
                    f = self.face_dim
                    assert y.shape[2] >= b + h + h + f, (
                        f"use_face=True expects {b + h + h + f} feature columns in y "
                        f"(126 body + 90 + 90 hands + {f} expression), got {y.shape[2]}. "
                        "The dataset must emit the expression coefficients; see "
                        "get_losses in baselightningmodule.py."
                    )
                    y_face = y[:, :, b+h+h:b+h+h+f].permute(0, 2, 1)

                mu_body = mu_r_hand = mu_l_hand = mu_face = None
                cond_face = None
                if self.use_duration:
                    # The frame-level conditioning is rebuilt from the segment tokens
                    # through the alignment map, so it replaces whatever `cond` the
                    # dataloader handed in. With precomputed durations the two are
                    # equivalent -- segment vectors are pooled and constant over the
                    # segment -- which is what makes the flag switchable mid-project.
                    y_mask_f = y_mask.to(y_body.dtype)
                    ((cond_body, cond_r_hand, cond_l_hand, cond_face),
                     (mu_body, mu_r_hand, mu_l_hand, mu_face), dur_loss, _) = (
                        self._duration_forward(
                            x, x_lengths, durations, y_mask_f,
                            # MAS (use_precomputed_durations=False) scores the
                            # alignment over the motion streams only -- the face is
                            # deliberately kept out of the timing decision.
                            (y_body, y_r_hand, y_l_hand), is_sign,
                        )
                    )
                elif self.use_face:
                    # Frame-level path: the fused conditioning arrives already at
                    # frame rate, so the face third is taken straight off it.
                    cond_face = self._face_cond(cond_body, cond_r_hand, cond_l_hand)

                # The contact term needs y_t and v_hat back to rebuild x1_hat, but
                # only on the steps where it actually fires -- every other step
                # takes the plain 2-tuple path and costs nothing extra.
                want_contact = (
                    self.use_contact_loss and self.training
                    and betas is not None and soft_dist is not None
                    and int(self.global_step) % int(self.contact_cfg.stride) == 0
                )
                self._contact_term = None
                if want_contact:
                    diff_loss_body, yb_t, _tb, vb = self.decoder_body.compute_loss(
                        t=t_shared, x1=y_body, cond=cond_body.permute(0, 2, 1), mask=y_mask,
                        mu=mu_body, return_extras=True)
                    diff_loss_r_hand, yr_t, _tr, vr = self.decoder_r_hand.compute_loss(
                        t=t_shared, x1=y_r_hand, cond=cond_r_hand.permute(0, 2, 1), mask=y_mask,
                        mu=mu_r_hand, return_extras=True)
                    diff_loss_l_hand, yl_t, _tl, vl = self.decoder_l_hand.compute_loss(
                        t=t_shared, x1=y_l_hand, cond=cond_l_hand.permute(0, 2, 1), mask=y_mask,
                        mu=mu_l_hand, return_extras=True)
                    self._contact_term = self._contact_loss(
                        {"t": t_shared, "body": (yb_t, vb),
                         "rhand": (yr_t, vr), "lhand": (yl_t, vl)},
                        betas, soft_dist, y_mask)
                else:
                    diff_loss_body, _ = self.decoder_body.compute_loss(t=t_shared, x1=y_body, cond=cond_body.permute(0, 2, 1), mask=y_mask, mu=mu_body)
                    diff_loss_r_hand, _ = self.decoder_r_hand.compute_loss(t=t_shared, x1=y_r_hand, cond=cond_r_hand.permute(0, 2, 1), mask=y_mask, mu=mu_r_hand)
                    diff_loss_l_hand, _ = self.decoder_l_hand.compute_loss(t=t_shared, x1=y_l_hand, cond=cond_l_hand.permute(0, 2, 1), mask=y_mask, mu=mu_l_hand)

                # The face never takes the contact path: the contact term is
                # hand/body geometry and the expression coefficients are not part
                # of it, so this one call covers both branches above.
                diff_loss_face = None
                if self.use_face:
                    diff_loss_face, _ = self.decoder_face.compute_loss(
                        t=t_shared, x1=y_face, cond=cond_face.permute(0, 2, 1),
                        mask=y_mask, mu=mu_face)

                if self.use_duration:
                    if self.prior_loss:
                        prior = (
                            self._prior_loss(y_body, mu_body, y_mask_f)
                            + self._prior_loss(y_r_hand, mu_r_hand, y_mask_f)
                            + self._prior_loss(y_l_hand, mu_l_hand, y_mask_f)
                        )
                        if self.use_face:
                            # Summed into the same scalar as the other streams, so
                            # the prior term keeps one weight rather than four.
                            prior = prior + self._prior_loss(y_face, mu_face, y_mask_f)
                    else:
                        prior = torch.zeros((), device=y.device, dtype=y_body.dtype)
                    if self.use_face:
                        return (diff_loss_body, diff_loss_r_hand, diff_loss_l_hand,
                                dur_loss, prior, diff_loss_face)
                    return diff_loss_body, diff_loss_r_hand, diff_loss_l_hand, dur_loss, prior
                if self.use_face:
                    return (diff_loss_body, diff_loss_r_hand, diff_loss_l_hand,
                            diff_loss_face)
            else:
                condition_body, condition_r_hand, condition_l_hand =  cond[0], cond[1], cond[2]
                # assert y.shape[2] == 90, "y shape[2] should be equal sequence length"

                y_body, y_r_hand, y_l_hand = separate_body_hands(y, self.body_indices, self.r_hand_indices, self.l_hand_indices)
                # print(f"y_body shape before compute loss: {y_body.shape}") [2, 39, 90]

                diff_loss_body, _ = self.decoder_body.compute_loss(t=t_shared, x1=y_body, cond=condition_body, mask=y_mask)
                diff_loss_r_hand, _ = self.decoder_r_hand.compute_loss(t=t_shared, x1=y_r_hand, cond=condition_r_hand, mask=y_mask)
                diff_loss_l_hand, _ = self.decoder_l_hand.compute_loss(t=t_shared, x1=y_l_hand, cond=condition_l_hand, mask=y_mask)
            return diff_loss_body, diff_loss_r_hand, diff_loss_l_hand

        # Bidirectional cross attention between the streams
        elif self.streams == 3 and (cond is not None) and (self.bicrossattention) and (not self.cascaded):
            condition_body, condition_r_hand, condition_l_hand =  cond[0], cond[1], cond[2]
            y_body, y_r_hand, y_l_hand = separate_body_hands(y, self.body_indices, self.r_hand_indices, self.l_hand_indices)
            
            output_body, u_body = self.decoder_body.compute_loss(t=t_shared, x1=y_body, cond=condition_body, bicrossattention=True)
            output_r_hand, u_r_hand = self.decoder_r_hand.compute_loss(t=t_shared, x1=y_r_hand, cond=condition_r_hand, bicrossattention=True)
            output_l_hand, u_l_hand = self.decoder_l_hand.compute_loss(t=t_shared, x1=y_l_hand, cond=condition_l_hand, bicrossattention=True)
            # print(f"shapes {output_body.shape}, {output_r_hand.shape}, {output_l_hand.shape}")
            # shapes torch.Size([16, 39, 28]), torch.Size([16, 48, 28]), torch.Size([16, 48, 28])  B, Feats, Seq_length
            """body_in  = rearrange(output_body, "b c t -> b t c")
            hands_in = rearrange(
                torch.cat([output_r_hand, output_l_hand], dim=1),  # [B, C_r+C_l, T]
                "b c t -> b t c"                               # [B, T, C_r+C_l]
                )
            body_with_attn, hands_with_attn = self.joint_cross_attn(body_in, hands_in)

            body_with_attn = rearrange(body_with_attn, "b t c -> b c t")
            hands_with_attn = rearrange(hands_with_attn, "b t c -> b c t")
            r_hand, l_hand = torch.split(hands_with_attn, hands_with_attn.shape[1] // 2, dim=1)"""
            body_with_attn = output_body.clone()
            r_hand = output_r_hand.clone()
            l_hand = output_l_hand.clone()

            # print(f"shapes {body_with_attn.shape}, {r_hand.shape}, {l_hand.shape}")

            diff_loss_body = F.mse_loss(body_with_attn, u_body, reduction="sum") / u_body.shape[1]
            diff_loss_r_hand = F.mse_loss(r_hand, u_r_hand, reduction="sum") / u_r_hand.shape[1]
            diff_loss_l_hand = F.mse_loss(l_hand, u_l_hand, reduction="sum") / u_l_hand.shape[1]
            return diff_loss_body, diff_loss_r_hand, diff_loss_l_hand

        # Cascaded model without inference (#TODO)
        elif self.streams == 3 and (cond is not None) and (self.cascaded):
            condition_body, condition_r_hand, condition_l_hand =  cond[0], cond[1], cond[2]
            # print("shapes of condition_body, condition_r_hand, condition_l_hand: ", condition_body.shape, condition_r_hand.shape, condition_l_hand.shape)
            # torch.Size([16, 26, 28]) torch.Size([16, 32, 28]) torch.Size([16, 32, 28])
            # assert y.shape[2] == 90, "y shape[2] should be equal sequence length"
            y_body, y_r_hand, y_l_hand = separate_body_hands(y, self.body_indices, self.r_hand_indices, self.l_hand_indices)
            # print(f"y_body shape before compute loss: {y_body.shape}") [2, 39, 90]

            diff_loss_body, _ = self.decoder_body.compute_loss(t=t_shared, x1=y_body, cond=condition_body)
            #print("body")
            diff_loss_r_hand, _ = self.decoder_r_hand.compute_loss(t=t_shared, x1=y_r_hand, cond=torch.cat([condition_body, condition_r_hand], dim=1))
            #print("r hand")
            diff_loss_l_hand, _ = self.decoder_l_hand.compute_loss(t=t_shared, x1=y_l_hand, cond=torch.cat([condition_body, condition_r_hand, condition_l_hand], dim=1))
            #print("l hand")
            return diff_loss_body, diff_loss_r_hand, diff_loss_l_hand
        
    @torch.inference_mode()
    def inference(self, sequence_length, n_timesteps, batch_size, cond=None, device="cuda", x0=None, temperature=1.0, y_lengths=None):
        # self.kmeans = True
        """
        Generates a sample from the model using the decoder.
        Args:
            sequence_length (int): Length of the generated sequence.
            n_timesteps (int): Number of timesteps for reverse diffusion in decoder.
            device (str): Device to use for computation.
        """
        if y_lengths is None:
            y_lengths = torch.full(
                (batch_size,), sequence_length,
                dtype=torch.long, device=device
            )
        mask = sequence_mask(y_lengths, sequence_length).unsqueeze(1).to(device)  # (B, 1, T)
        
        # Generate sample tracing the probability flow
        # n_timesteps - number of steps to use for reverse diffusion in decoder.
        # mask, n_timesteps, batch_size=8, n_feats=42, sequence_length=90, cond=None, temperature=1.0, device='cuda')
        # print(f"Shape of y_lengths: {y_lengths.shape}")  # 8
        # print(f"Shape of mask: {mask.shape}")  # 8, 1, 90
        # self.decoder(mu_y, y_mask, n_timesteps, temperature, spks)
        # (x, mask, t)

        if self.streams == 1:
            decoder_outputs = self.decoder(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                x0=x0,
                mask=mask,
                temperature=temperature
                )
            
            decoder_outputs = decoder_outputs[:, :, :sequence_length]
            return decoder_outputs
        elif self.streams == 3 and cond is None:
            decoder_outputs_body = self.decoder_body(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            
            print(decoder_outputs_body.shape)
            decoder_outputs_r_hand = self.decoder_r_hand(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            
            print(decoder_outputs_r_hand.shape)   # torch.Size([1, 48, 90])
            decoder_outputs_l_hand = self.decoder_l_hand(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            print(f"Inference: Shape of decoder_outputs_body: {decoder_outputs_body.shape}, decoder_outputs_hands: {decoder_outputs_r_hand.shape}, mean: {decoder_outputs_r_hand.mean().item()}, std: {decoder_outputs_r_hand.std().item()}")
            if self.smplx:
                # Unpermute each part back to (batch, time, features)
                decoder_outputs_body_unperm = decoder_outputs_body.permute(0, 2, 1)    # (16, 90, 69)
                decoder_outputs_r_hand_unperm  = decoder_outputs_r_hand.permute(0, 2, 1)  # (16, 90, 45)
                decoder_outputs_l_hand_unperm  = decoder_outputs_l_hand.permute(0, 2, 1)  # (16, 90, 45)
                # Split y_body back into its two original segments
                # decoder_outputs_body = decoder_outputs_body_unperm[:, :, :63]        # original y[:,:,:63]
                # decoder_outputs_body_last  = decoder_outputs_body_unperm[:, :, 63:]        # original y[:,:,63+45+45:]
                # Concatenate in original order: body_first | r_hand | l_hand | body_last
                # mask is (B, 1, T); the *_unperm tensors are (B, T, feat),
                # so transpose it to (B, T, 1) to broadcast over the feature axis.
                mask_cpu = mask.permute(0, 2, 1).cpu().to(decoder_outputs_body_unperm.dtype)
                decoder_outputs_body_unperm = decoder_outputs_body_unperm * mask_cpu
                decoder_outputs_r_hand_unperm = decoder_outputs_r_hand_unperm * mask_cpu
                decoder_outputs_l_hand_unperm = decoder_outputs_l_hand_unperm * mask_cpu

                decoder_outputs = torch.cat((decoder_outputs_body_unperm, decoder_outputs_r_hand_unperm, decoder_outputs_l_hand_unperm), dim=2)
                print(f"Inference: Shape of concatenated decoder_outputs_body: {decoder_outputs_body.shape}")
                print(f"Inference: Shape of concatenated decoder_outputs_r_hand_unperm: {decoder_outputs_r_hand_unperm.shape}")
                return decoder_outputs_body, decoder_outputs_r_hand, decoder_outputs_l_hand, decoder_outputs
            else:
                decoder_outputs = concatenate_body_hands(decoder_outputs_body, decoder_outputs_r_hand, decoder_outputs_l_hand, self.body_indices, self.r_hand_indices, self.l_hand_indices)
                print(f"Inference: Shape of concatenated decoder_outputs: {decoder_outputs.shape}")
                return decoder_outputs
        # Same for normal split conditioning model and the one with bidrectional cross-attention
        elif self.streams == 3 and cond is not None and (not self.cascaded) or (not self.bicrossattention) and (not self.kmeans):
            if not self.smplx:
                cond_body, cond_r_hand, cond_l_hand = cond[0][:, :, :sequence_length], cond[1][:, :, :sequence_length], cond[2][:, :, :sequence_length] # last dim is time [:, :, :90]
                cond_face = None  # face is smplx-only; see the guard in __init__
            else:
                cond_body, cond_r_hand, cond_l_hand = cond
                #cond_body = self.cond_proj(cond_body).permute(0, 2, 1).to(device)
                #cond_r_hand = self.cond_proj(cond_r_hand).permute(0, 2, 1).to(device)
                #cond_l_hand = self.cond_proj(cond_l_hand).permute(0, 2, 1).to(device)
                # The face reads all three thirds concatenated, so it is built
                # BEFORE the permute, while the label axis is still last.
                cond_face = (
                    self._face_cond(cond_body, cond_r_hand, cond_l_hand)
                    .permute(0, 2, 1).to(device) if self.use_face else None
                )
                cond_body = cond_body.permute(0, 2, 1).to(device)
                cond_r_hand = cond_r_hand.permute(0, 2, 1).to(device)
                cond_l_hand = cond_l_hand.permute(0, 2, 1).to(device)
                # cond_body, cond_r_hand, cond_l_hand = cond, cond, cond
            print(f"Inference shapes for conditions cond_body.shape: {cond_body.shape}, cond_r_hand.shape: {cond_r_hand.shape}")
            # cond_body.shape: torch.Size([4, 26, 90]), cond_r_hand.shape: torch.Size([4, 32, 90])
            decoder_outputs_body = self.decoder_body(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                cond=cond_body,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            
            print(decoder_outputs_body.shape)
            decoder_outputs_r_hand = self.decoder_r_hand(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                cond=cond_r_hand,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            
            print(decoder_outputs_r_hand.shape)   # torch.Size([1, 48, 90])
            decoder_outputs_l_hand = self.decoder_l_hand(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                cond=cond_l_hand,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            decoder_outputs_face = self.decoder_face(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                cond=cond_face,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu() if self.use_face else None
            # put the outputs back together in the right order
            # Inference: Shape of decoder_outputs_body: torch.Size([1, 42, 90]), decoder_outputs_hands: torch.Size([1, 94, 90])
            print(f"Inference: Shape of decoder_outputs_body: {decoder_outputs_body.shape}, decoder_outputs_hands: {decoder_outputs_r_hand.shape}, mean: {decoder_outputs_r_hand.mean().item()}, std: {decoder_outputs_r_hand.std().item()}")
            if self.smplx:
                # Unpermute each part back to (batch, time, features)
                decoder_outputs_body_unperm = decoder_outputs_body.permute(0, 2, 1)    # (16, 90, 69)
                decoder_outputs_r_hand_unperm  = decoder_outputs_r_hand.permute(0, 2, 1)  # (16, 90, 45)
                decoder_outputs_l_hand_unperm  = decoder_outputs_l_hand.permute(0, 2, 1)  # (16, 90, 45)
                # Split y_body back into its two original segments
                # decoder_outputs_body = decoder_outputs_body_unperm[:, :, :63]        # original y[:,:,:63]
                # decoder_outputs_body_last  = decoder_outputs_body_unperm[:, :, 63:]        # original y[:,:,63+45+45:]
                # Concatenate in original order: body_first | r_hand | l_hand | body_last
                # mask is (B, 1, T); the *_unperm tensors are (B, T, feat),
                # so transpose it to (B, T, 1) to broadcast over the feature axis.
                mask_cpu = mask.permute(0, 2, 1).cpu().to(decoder_outputs_body_unperm.dtype)
                decoder_outputs_body_unperm   = decoder_outputs_body_unperm * mask_cpu
                decoder_outputs_r_hand_unperm = decoder_outputs_r_hand_unperm * mask_cpu
                decoder_outputs_l_hand_unperm = decoder_outputs_l_hand_unperm * mask_cpu

                # Face last, matching the y layout get_losses builds. With the
                # stream off the width is the unchanged 306.
                face_unperm = None
                if self.use_face:
                    face_unperm = decoder_outputs_face.permute(0, 2, 1) * mask_cpu
                    decoder_outputs = torch.cat((decoder_outputs_body_unperm, decoder_outputs_r_hand_unperm, decoder_outputs_l_hand_unperm, face_unperm), dim=2)
                else:
                    decoder_outputs = torch.cat((decoder_outputs_body_unperm, decoder_outputs_r_hand_unperm, decoder_outputs_l_hand_unperm), dim=2)
                self.last_face = face_unperm
                print(f"Inference: Shape of concatenated decoder_outputs: {decoder_outputs.shape}")
                print(f"Inference: Shape of concatenated decoder_outputs_body: {decoder_outputs_body.shape}")
                print(f"Inference: Shape of concatenated decoder_outputs_r_hand_unperm: {decoder_outputs_r_hand_unperm.shape}")
                return decoder_outputs_body_unperm, decoder_outputs_r_hand_unperm, decoder_outputs_l_hand_unperm, decoder_outputs
            else:
                decoder_outputs = concatenate_body_hands(decoder_outputs_body, decoder_outputs_r_hand, decoder_outputs_l_hand, self.body_indices, self.r_hand_indices, self.l_hand_indices)
                print(f"Inference: Shape of concatenated decoder_outputs: {decoder_outputs.shape}")
                # Shape of concatenated decoder_outputs: torch.Size([1, 129, 90])
                return decoder_outputs
        elif self.streams == 3 and cond and (self.cascaded) and (not self.bicrossattention) and (not self.kmeans):
            cond_body, cond_r_hand, cond_l_hand = cond[0][:, :, :sequence_length], cond[1][:, :, :sequence_length], cond[2][:, :, :sequence_length] # last dim is time [:, :, :90]
            print(f"Inference shapes for conditions cond_body.shape: {cond_body.shape}, cond_r_hand.shape: {cond_r_hand.shape}")
            # Inference shapes for conditions cond_body.shape: torch.Size([2, 26, 234]), cond_r_hand.shape: torch.Size([2, 32, 234])
            # cond_r_hand, cond_l_hand replace with model outputs projections
            decoder_outputs_body = self.decoder_body(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                cond=cond_body,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            print("body")
            print("decoder_outputs_body.shape: ", decoder_outputs_body.shape)  #  torch.Size([2, 39, 234])
            decoder_outputs_r_hand = self.decoder_r_hand(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                cond=torch.cat([cond_body, cond_r_hand], dim=1),
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            #print("r hand")
            print(decoder_outputs_r_hand.shape)   # torch.Size([1, 48, 90])
            decoder_outputs_l_hand = self.decoder_l_hand(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                cond=torch.cat([cond_body, cond_r_hand, cond_l_hand], dim=1),
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            #print("l hand")
            # put the outputs back together in the right order
            # Inference: Shape of decoder_outputs_body: torch.Size([1, 42, 90]), decoder_outputs_hands: torch.Size([1, 94, 90])
            print(f"Inference: Shape of decoder_outputs_body: {decoder_outputs_body.shape}, decoder_outputs_hands: {decoder_outputs_r_hand.shape}, mean: {decoder_outputs_r_hand.mean().item()}, std: {decoder_outputs_r_hand.std().item()}")
            decoder_outputs = concatenate_body_hands(decoder_outputs_body, decoder_outputs_r_hand, decoder_outputs_l_hand, self.body_indices, self.r_hand_indices, self.l_hand_indices)
            print(f"Inference: Shape of concatenated decoder_outputs: {decoder_outputs.shape}")
            # Shape of concatenated decoder_outputs: torch.Size([1, 129, 90])
            return decoder_outputs
        elif self.streams == 3 and (not self.bicrossattention) and (not self.cascaded) and (self.kmeans):
            # [256, 40, 3], expected input[2, 65, 234] to have 40 channels, but got 65 channels
            cond_body, cond_r_hand, cond_l_hand = cond[0][:, :, :sequence_length], cond[1][:, :, :sequence_length], cond[2][:, :, :sequence_length] # last dim is time [:, :, :90]
            print(f"Inference shapes for conditions cond_body.shape: {cond_body.shape}, cond_r_hand.shape: {cond_r_hand.shape}")
            # Inference shapes for conditions cond_body.shape: torch.Size([2, 26, 234]), cond_r_hand.shape: torch.Size([2, 32, 234])
            # Inference shapes for conditions cond_body.shape: torch.Size([2, 26, 234]), cond_r_hand.shape: torch.Size([2, 32, 234])
            decoder_outputs_body = self.decoder_body(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                cond=cond_body,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            decoder_outputs_r_hand = self.decoder_r_hand(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                cond=cond_r_hand,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            decoder_outputs_l_hand = self.decoder_l_hand(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                n_timesteps=n_timesteps,
                cond=cond_l_hand,
                x0=x0,
                mask=mask,
                temperature=temperature
                ).cpu()
            decoder_outputs = concatenate_body_hands(decoder_outputs_body, decoder_outputs_r_hand, decoder_outputs_l_hand, self.body_indices, self.r_hand_indices, self.l_hand_indices)
            print(f"Inference: Shape of concatenated decoder_outputs: {decoder_outputs.shape}")