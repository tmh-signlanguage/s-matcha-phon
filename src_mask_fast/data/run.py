"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os, sys
from pathlib import Path
import posixpath

import ffmpeg
import smplx
import torch
import torchaudio
from src_mask_fast.data.smplx_loader import BaseDataset, SmplxDataloader, MocapSmplxDataset
from src_mask_fast.data.enums import FeatName
from torch.utils.data import DataLoader
from torchvision.io import write_video
from src_mask_fast.data.visualize import PyrenderRenderer



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


def smplx_to_pelvis_relative(batch, smplx_model):
    transl = batch["smplx_mesh_transl"].to(args.device)         # (T,3)  -> (1, T, 3)
    orient = batch["smplx_mesh_global_orient"].to(args.device)     # (T,3) -> (1, T, 3)
    body_pose = batch["smplx_mesh_body_pose"].to(args.device)     # (T,63)
    betas = batch["smplx_mesh_betas"].to(args.device)              # (10)  -> torch.Size([1, 600, 300])
    lhand = batch["smplx_mesh_left_hand_pose"].to(args.device)     # (T,45)
    rhand = batch["smplx_mesh_right_hand_pose"].to(args.device)    # (T,45)

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
    
    # 3 compute joints from SMPL-X
    """output = smplx_model(
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
    joints_rel = joints - pelvis """

    batch["smplx_mesh_transl"] = transl.unsqueeze(0)
    batch["smplx_mesh_global_orient"] = orient.unsqueeze(0)
    # batch["smlpx_mesh_betas"] = betas
    # batch["joints_rel"] = joints_rel

    return batch


def compute_lbs_from_batch(args, smplx_model, batch, smplx_keys, B=None):
    smplx_input = {}
    for key in batch.keys():
        if key in smplx_keys:
            smplx_input[key.replace("smplx_mesh_", "")] = batch[key][0].to(args.device)
    # compute the vertices and render the video
    print("betas shape:", smplx_input["betas"].shape)  # [300]
    # smplx_input["betas"] = smplx_input["betas"].unsqueeze(0).expand(B, -1)[:, :10]
    betas = smplx_input["betas"]
    B = smplx_input["body_pose"].shape[0]
    print(f"B: {B}")
    #if betas.ndim == 1:
    #    betas = betas.unsqueeze(0)
    smplx_input["betas"] = betas.unsqueeze(0).expand(B, -1)[:, :10].unsqueeze(0)  # betas.expand(B, -1).unsqueeze(0)
    print(f"New betas shape {smplx_input['betas'].shape}")   # New betas shape torch.Size([1, 90, 10])
    # Sizes of tensors must match except in dimension 1. Expected size 1 but got size 90 for tensor number 1 in the list.
    print(f"Global orient shape {smplx_input['global_orient'].shape}")
    smplx_input["global_orient"] = smplx_input["global_orient"].unsqueeze(0)
    smplx_input["transl"] = smplx_input["transl"].unsqueeze(0)
    result_verts = smplx_model(**smplx_input).vertices[None, ...]  
    return result_verts

def load_and_visualize_data(args):
    ### set up the dataset
    smplx_keys = [
        FeatName.BODY.value,
        FeatName.ROT.value,
        FeatName.TRANS.value,
        FeatName.SHAPE.value,
        FeatName.LEFT_HAND.value,
        FeatName.RIGHT_HAND.value,
    ]
    features_to_load = smplx_keys + args.anno

    """dataset = BaseDataset(
        split="train",
        data_dir=args.data_path[0],
        seq_len=args.seq_len,
        features_to_load=features_to_load,
        load_multiperson=args.load_multiperson,
    )"""
    dataset = MocapSmplxDataset(
        split="train",
        data_dir=args.data_path[0],
        seq_len=args.seq_len,
        features_to_load=features_to_load
    )

    ### set up the smplx function
    smplx_model = smplx.create(
        args.smplx_model_path,
        model_type="smplx",
        gender="neutral",
        flat_hand_mean=True,
        num_betas=300,
        num_expression_coeffs=100,
        use_pca=False,
        batch_size=args.seq_len,
    ).to(args.device)

    ### set up the renderer for smplx
    r = PyrenderRenderer(args.smplx_topology_path).to(args.device)
    r.faces = r.faces[:, [0, 2, 1]]

    ### set up to the dataloader
    dl_kwargs = {"num_workers": 0, "shuffle": True, "batch_size": 1}
    dataloader = DataLoader(dataset, **dl_kwargs)

    ### iterate through the dataloader
    for i, batch in enumerate(dataloader):
        print(f".....[batch {i}]:.....")
        print(f"subject_id: {batch['subject_id'][0]}")
        print(f"sequence_id: {batch['sequence_id'][0]}")
        print(f"loaded batch keys: {batch.keys()}")
        # batch = normalize_smplx_yaw(batch) 
        # rename the smplx keys so that it matches the smplx function
        print(f"rendering smplx mesh for {batch['subject_id'][0]}")
        
        verts_list = []
        B = len(batch)
        T = batch["smplx_mesh_body_pose"].shape[0]
        # batch = smplx_to_pelvis_relative(batch, smplx_model)
        batch["smplx_mesh_global_orient"] = torch.zeros(T, 3, device=args.device)   # (T, 3) axis-angle
        batch["smplx_mesh_transl"] = torch.zeros(T, 3, device=args.device)    # (T, 3)

        verts_list.append(compute_lbs_from_batch(args, smplx_model, batch, smplx_keys, B))
        assert verts_list[0].shape == torch.Size([1, args.seq_len, 10475, 3])
        if FeatName.MULTIPERSON.value in batch.keys():
            other_ids = batch[FeatName.MULTIPERSON.value]
            for other_id in other_ids.keys():
                print(f"multiperson: rendering smplx mesh for {other_id}")
                other_batch = other_ids[other_id]
                verts_list.append(compute_lbs_from_batch(args, smplx_model, other_batch, smplx_keys, B))
        video_frames = r(verts_list)[0]  # T x 3 x H x W
        video_frames = video_frames.transpose(0, 2, 3, 1)  # T x H x W x 3
        #save_path = (
        #    f"{args.output_dir}/{batch['subject_id'][0]}--{batch['sequence_id'][0]}-{batch['start_frame'][0]}.mp4"
        #)
        save_path = Path(args.output_dir) / f"{batch['subject_id'][0]}--{batch['sequence_id'][0]}-{batch['start_frame'][0]}.mp4"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        write_video(str(save_path), video_frames, fps=30)

        # save_path_path = Path(save_path[0])
        # save_path_path.parent.mkdir(parents=True, exist_ok=True)
        #save_path.parent.mkdir(parents=True, exist_ok=True)
        #write_video(save_path, video_frames, fps=30)
        if FeatName.TEXT.value in batch.keys():
            print(f"text anno: {batch[FeatName.TEXT.value][0]}")
            if "text_offset_in_seconds" in batch.keys():
                print(f"text anno offset in seconds: {batch['text_offset_in_seconds'][0]:02f}")
        if FeatName.TEXT_HOLISTIC.value in batch.keys():
            print(f"text anno holistic: {batch[FeatName.TEXT_HOLISTIC.value][0]}")
        if (
            FeatName.AUDIO_SEPARATED.value in batch.keys() and torch.any(batch[FeatName.AUDIO_SEPARATED.value][0] != 0)
        ) or (FeatName.AUDIO_RAW.value in batch.keys() and torch.any(batch[FeatName.AUDIO_RAW.value][0] != 0)):
            key = FeatName.AUDIO_RAW if FeatName.AUDIO_RAW.value in batch.keys() else FeatName.AUDIO_SEPARATED
            print(f"found {key.value}, if trying to load separated and mono, mono takes precedence")
            audio_path = f"{os.path.splitext(save_path)[0]}_audio.wav"
            video_audio_path = f"{os.path.splitext(save_path)[0]}_audio.mp4"
            torchaudio.save(audio_path, batch[key.value][0], 48_000)  # sr = 48_000 from the file
            video_input = ffmpeg.input(save_path)
            audio_input = ffmpeg.input(audio_path)
            ffmpeg_out = ffmpeg.output(video_input, audio_input, video_audio_path, vcodec="copy", acodec="aac")
            ffmpeg_out = ffmpeg_out.global_args("-hide_banner", "-loglevel", "error")
            result = ffmpeg_out.run(overwrite_output=True)
            print(f"ffmpeg result: {result}")
            print(f"saved video with audio {video_audio_path}\n")
        else:
            print(f"saved video without audio {save_path}\n")

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_model_path", type=str, required=True, help="path to the smplx model")
    parser.add_argument("--smplx_topology_path", type=str, required=True, help="path to the smplx obj topology")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory to save the rendered videos")
    parser.add_argument("--load_multiperson", action="store_true", help="whether to load the multiperson dataset")
    parser.add_argument("--data_path", nargs="+", type=str, required=True, help="A list of data directories to load")
    parser.add_argument("--anno", nargs="+", type=str, default=[], help="additional annotations to load, e.g. audio")
    parser.add_argument("--seq_len", type=int, default=90, help="max length of the sequences to load")
    parser.add_argument("--device", type=str, default="cuda", help="device to run the smplx model on")
    args = parser.parse_args()
    # example script to load and render a batch of data
    load_and_visualize_data(args)

    # python src/data/run.py --smplx_model_path /nfs/deepspeech/home/annkle/lightning-hydra-template/body_models/models_lockedhead/ --smplx_topology_path /nfs/deepspeech/home/annkle/lightning-hydra-template/body_models/models_lockedhead/smplx_mesh_neutral.obj --data_path /nfs/signbot1/data/Embody3D-Meta/embody-3d/embody_3d__20251016__HANDS_smplx/hand_contact/ --output_dir /nfs/deepspeech/home/annkle/lightning-hydra-template/data/data/embody3d-sample/tmp/ --load_multiperson