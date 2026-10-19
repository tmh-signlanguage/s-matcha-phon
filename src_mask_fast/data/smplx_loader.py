"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json, sys
import joblib as jl
import logging
import os
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple
import h5py
import numpy as np
import random
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torchaudio

from src_mask_fast.utils.model import sequence_mask
from src_mask_fast.data.enums import FeatName
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation
from bvhTools.angleConversion import scipyToSixD, sixDToScipy
from einops import rearrange


# Fallback for standalone callers (smoke_test / dataloader_test / integrity_test below).
# Training runs should set this via Hydra: `data.h5_labels_path` in configs/data/smplx.yaml.
DEFAULT_H5_LABELS_PATH = (
    "XX"
    "combined_data_labels_no_gaps_120fps_fused.h5"
)
# Same pattern for the per-feature mean/std the pose h5 was z-scored with. Training
# runs set it via `data.norm_stats_path`; it has to be the stats file THAT pose h5
# was built with (mocap and PHOENIX differ), or unnormalize() silently de-scales
# with the wrong numbers.
DEFAULT_NORM_STATS_PATH = (
    "XX"
    "norm_stats_6d_mocap.json"
)



#def seed_worker(worker_id):
#    worker_seed = torch.initial_seed() % 2**32
#    np.random.seed(worker_seed)
#    random.seed(worker_seed)


#g = torch.Generator()
#g.manual_seed(0)


class BaseDataset(Dataset):
    """HDF5-backed dataset for Embody3D pose data."""
    def __init__(
        self,
        split: str,
        data_dir: str,  # path to embody3d.h5
        logger=None,
        seq_len: int = 600,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        features_to_load: Optional[List[str]] = None,
        load_multiperson: bool = False,
        norm_stats_path: str = "/pfs/lustrep1/scratch/project_465002480/lightning-hydra-template/data/norm_stats_6d_embody3d.json",
    ):
        assert features_to_load is not None, "features_to_load must be specified"
        self.split = split
        self.h5_path = data_dir
        self.seq_len = seq_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.features_to_load = features_to_load
        self.load_multiperson = load_multiperson
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self._build_index()
        self._load_normalization_stats(norm_stats_path)
        self._hf = h5py.File(self.h5_path, "r")
        self._cache_h5_keys() 

    # Index building
    def _build_index(self):
        """Reconstruct dataset_info and frame index from HDF5 metadata only.
        No array data is read here — just shapes and attributes.
        """
        with h5py.File(self.h5_path, "r") as hf:
            dataset_info = json.loads(hf.attrs["dataset_info"])

            # Sync lengths from HDF5 attrs in case dataset.json has stale values
            # FOR TESTING INDEX DATASET INFO
             # ======== sanity check first 10 subjects to speed up debugging ========
            for subject_id in list(dataset_info.keys()):  # [:10] 
                if subject_id not in hf:
                    continue
                for seq_name in dataset_info[subject_id]:
                    if seq_name not in hf[subject_id]:
                        continue
                    seq_grp = hf[subject_id][seq_name]
                    dataset_info[subject_id][seq_name]["length"] = int(
                        seq_grp.attrs["length"]
                    )
                    dataset_info[subject_id][seq_name]["multiperson"] = json.loads(
                        seq_grp.attrs.get("multiperson", "null")
                    )

        self.dataset_info = self._split_dataset(dataset_info)

        # Build flat index: global segment idx → (subject_id, seq_name)
        seqs = [
            (subject_id, seq_name)
            for subject_id in self.dataset_info
            for seq_name in self.dataset_info[subject_id]
        ]

        self.idx_per_seq = []
        self.idx2seq = {}
        idx = 0
        for subject_id, seq_name in seqs:
            self.idx2seq[idx] = (subject_id, seq_name)
            self.idx_per_seq.append(idx)
            length = self.dataset_info[subject_id][seq_name]["length"]
            idx += length - self.seq_len + 1

        self.num_valid_segments = idx

        n_seqs = len(seqs)
        self.logger.info(
            f"[BaseDataset] {self.split}: {n_seqs} sequences, "
            f"{self._calc_hours():.2f}h, {self.num_valid_segments} segments"
        )

    def _split_dataset(self, dataset_info: Dict) -> Dict:
        """Define train/test/val split"""
        subject_ids = list(dataset_info.keys())
        train_end = int(len(subject_ids) * self.train_ratio)
        val_end   = train_end + int(len(subject_ids) * self.val_ratio)

        if self.split == "train":
            keys = subject_ids[:train_end]
        elif self.split == "val":
            if train_end == val_end:
                raise Exception(
                    f"[BaseDataset] val partition is empty: "
                    f"{self.val_ratio} of {len(subject_ids)} subjects is 0"
                )
            keys = subject_ids[train_end:val_end]
        else:  # test
            if val_end == len(subject_ids):
                test_ratio = 1.0 - self.train_ratio - self.val_ratio
                raise Exception(
                    f"[BaseDataset] test partition is empty: "
                    f"{test_ratio} of {len(subject_ids)} subjects is 0"
                )
            keys = subject_ids[val_end:]

        split_dict = {k: dataset_info[k] for k in keys}
        n_seqs = sum(len(v) for v in split_dict.values())
        self.logger.info(
            f"[BaseDataset] {self.split}: {len(split_dict)} subjects, {n_seqs} sequences"
        )
        return split_dict

    # Normalization
    def _load_normalization_stats(self, stats_path: str) -> None:
        with open(stats_path, "r") as f:
            raw = json.load(f)
        self.norm_stats = {
            feat: {
                "mean": np.array(v["mean"], dtype=np.float32),
                "std":  np.array(v["std"],  dtype=np.float32),
            }
            for feat, v in raw.items()
        }
        self.logger.info(f"[BaseDataset] loaded normalization stats from {stats_path}")

    def unnormalize(self, data: torch.Tensor, feat_name: str) -> torch.Tensor:
        """Undo normalization — useful for visualization and evaluation."""
        assert hasattr(self, "norm_stats"), "norm_stats not loaded"
        mean = torch.tensor(self.norm_stats[feat_name]["mean"], dtype=torch.float32)
        std  = torch.tensor(self.norm_stats[feat_name]["std"],  dtype=torch.float32)
        return data * std.clamp(min=1e-8) + mean


    # Indexing
    def _idx2segment(self, idx: int) -> Tuple[str, str, int]:
        closest = np.searchsorted(self.idx_per_seq, idx, side="right") - 1
        start_idx = self.idx_per_seq[closest]
        subject_id, seq_name = self.idx2seq[start_idx]
        frame_offset = idx - start_idx
        return subject_id, seq_name, frame_offset

    def __len__(self) -> int:
        return self.num_valid_segments

    # Data loading
    def _load_chunk(
        self,
        subject_id: str,
        seq_name: str,
        start_frame: int,
    ) -> Dict[str, torch.Tensor]:
        """Load one sliding-window chunk from HDF5."""
        end_frame = start_frame + self.seq_len
        hf = self._hf

        if subject_id not in hf:
            raise KeyError(f"Subject {subject_id} not found in HDF5")
        if seq_name not in hf[subject_id]:
            raise KeyError(f"Sequence {seq_name} not found under subject {subject_id}")

        seq_grp = hf[subject_id][seq_name]

        data = {
            FeatName.SUBJECT_ID.value:  subject_id,
            FeatName.SEQUENCE_ID.value: seq_name,
            FeatName.START_FRAME.value: torch.tensor(start_frame, dtype=torch.long),
            FeatName.LENGTH.value:      torch.tensor(self.seq_len, dtype=torch.long),
        }

        for feat_name in self.features_to_load:
            # Map FeatName string to the key stored in HDF5
            h5_key = self._feat_name_to_h5_key(feat_name)
            if h5_key is None:
                self.logger.warning(f"[BaseDataset] no HDF5 key for feat '{feat_name}', skipping")
                continue
            if h5_key not in seq_grp:
                self.logger.warning(f"[BaseDataset] '{h5_key}' not in HDF5 for {subject_id}/{seq_name}")
                continue

            arr = seq_grp[h5_key][start_frame:end_frame]  # (seq_len, D)

            assert arr.shape[0] == self.seq_len, (
                f"[BaseDataset] expected {self.seq_len} frames for {feat_name}, "
                f"got {arr.shape[0]} (subject={subject_id}, seq={seq_name}, "
                f"start={start_frame})"
            )

            data[feat_name] = torch.from_numpy(arr.copy())

        return data

    def _feat_name_to_h5_key(self, feat_name: str) -> Optional[str]:
        """Map FeatName enum values to HDF5 dataset keys.
        HDF5 keys are the raw feature directory names from disk
        (e.g. 'body_pose', 'left_hand_pose'), which is what the
        conversion script used as group names.
        Adjust this mapping to match your actual FeatName enum values.
        """
        # Direct match first — if FeatName.value == the folder name on disk
        return feat_name if feat_name in self._known_h5_keys else None

    def _cache_h5_keys(self):
        """Cache the set of available HDF5 keys from the first sequence.
        Called lazily on first __getitem__ to avoid opening HDF5 at init.
        """
        if not hasattr(self, "_known_h5_keys"):
            hf = self._hf
            # Grab keys from the first available sequence
            first_subject = next(iter(self.dataset_info))
            first_seq = next(iter(self.dataset_info[first_subject]))
            self._known_h5_keys = set(hf[first_subject][first_seq].keys())
            self.logger.info(
                f"[BaseDataset] available HDF5 keys: {sorted(self._known_h5_keys)}"
            )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Cache available keys on first access
        # self._cache_h5_keys()

        subject_id, seq_name, frame_offset = self._idx2segment(idx)
        data = self._load_chunk(subject_id, seq_name, frame_offset)

        if self.load_multiperson:
            other_ids = self.dataset_info[subject_id][seq_name].get("multiperson")
            other_id_dict = {}
            if other_ids:
                hf = self._hf
                for other_id in other_ids:
                    if other_id in hf and seq_name in hf[other_id]:
                        other_id_dict[other_id] = self._load_chunk(
                            other_id, seq_name, frame_offset
                        )
                    else:
                        self.logger.warning(
                            f"[BaseDataset] multiperson subject {other_id} "
                            f"not found for seq {seq_name}"
                        )
            data[FeatName.MULTIPERSON.value] = other_id_dict
        return data

    # Utilities
    def _calc_hours(self, fps: int = 30) -> float:
        num_frames = self.num_valid_segments + self.seq_len - 1
        return num_frames / fps / 3600.0



FEAT_TO_H5_KEY: Dict[str, str] = {
    FeatName.BODY.value:       "body_pose",
    FeatName.ROT.value:        "global_orient",
    FeatName.TRANS.value:      "transl",
    FeatName.SHAPE.value:      "betas",
    FeatName.LEFT_HAND.value:  "lhand_pose",
    FeatName.RIGHT_HAND.value: "rhand_pose",
    # Face stream. The h5 group name is "expression", matching the npz key that
    # convert_mocap_to_hdf5.py copies across (body_pose lhand_pose rhand_pose
    # jaw_pose leye_pose reye_pose betas expression global_orient transl).
    FeatName.EXPRESSION.value: "expression",
    FeatName.ANNO_DOM_FEATS.value: "dom_labels",
    FeatName.ANNO_NONDOM_FEATS.value: "nondom_labels",
    FeatName.ANNO_BODY_FEATS.value: "body_labels",
    FeatName.ANNO_STSNET_LABELS.value: "stsnet_labels",
    FeatName.ANNO_START_FRAME_SEGMENT.value: "start_frame",
    FeatName.ANNO_END_FRAME_SEGMENT.value: "end_frame",
}
 
# betas has no time axis — read whole array, no frame slicing
STATIC_FEATS = {
    FeatName.SHAPE.value}

# Optional jaw rotation appended to the face stream. Stored in the pose h5 as
# z-scored 6D, exactly like body/hands, so it is concatenated as-is: face_dim is
# then expression width + 6 (10 + 6 = 16 on PHOENIX). It has no FeatName of its
# own -- it only ever travels inside FeatName.EXPRESSION.
JAW_H5_KEY = "jaw_pose"

ANNOTATION_FEATS = {
    FeatName.ANNO_DOM_FEATS.value,
    FeatName.ANNO_NONDOM_FEATS.value,
    FeatName.ANNO_BODY_FEATS.value,
    FeatName.ANNO_STSNET_LABELS.value,
    FeatName.ANNO_START_FRAME_SEGMENT.value,
    FeatName.ANNO_END_FRAME_SEGMENT.value,
}
 
 
class MocapHDF5Dataset(Dataset):
    """HDF5-backed dataset for mocap SMPL-X data.
 
    Drop-in replacement for MocapSmplxDataset — same constructor signature and
    same __getitem__ output keys — but reads lazily from the HDF5 file produced
    by convert_mocap_to_hdf5.py, following the same pattern as BaseDataset.
 
    A single file handle (self._hf) is opened at init and reused across all
    __getitem__ calls. The sliding-window index (sequence names + lengths) is
    built from HDF5 attrs only — no array data is read at construction time.
 
    HDF5 layout expected:
        /{subject_id}/{filename.npz}/body_pose      (T, 126) — 21j × 6D, z-scored
        /{subject_id}/{filename.npz}/global_orient  (T, 6)
        /{subject_id}/{filename.npz}/lhand_pose     (T, 90)
        /{subject_id}/{filename.npz}/rhand_pose     (T, 90)
        /{subject_id}/{filename.npz}/jaw_pose       (T, 6)
        /{subject_id}/{filename.npz}/transl         (T, 3)   — raw
        /{subject_id}/{filename.npz}/betas          (10,)    — raw, static
        root attr "dataset_info" — JSON matching MocapSmplxDataset schema
 
    Args:
        split:            "train" / "val" / "test"
        data_dir:         path to the .h5 file (replaces the npz folder path)
        seq_len:          frames per sliding window
        train_ratio:      fraction of sequences for training
        val_ratio:        fraction of sequences for validation
        features_to_load: list of FeatName values to return per sample
        logger:           optional logger
        load_multiperson: unused, kept for API compatibility
    """
 
    def __init__(
        self,
        split: str,
        data_dir: str,
        #seq_len: int = 90,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        features_to_load: Optional[List[str]] = None,
        logger: Optional[Logger] = None,
        load_multiperson: bool = False,  # API compatibility, unused
        norm_stats_path: str = DEFAULT_NORM_STATS_PATH,
        h5_labels_path: Optional[str] = None,
        label_dim: int = 256,
        return_tokens: bool = False,
        return_contacts: bool = False,
        h5_contacts_path: Optional[str] = None,
        return_face: bool = False,
        face_dim: int = 10,
        name_convention: Optional[str] = None,
        face_label_dim: int = 0,
    ):
        assert features_to_load is not None, "features_to_load must be specified"
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        # Width of each segment's `face` label, appended to its token after
        # body|dom|nondom. 0 -> the face slot is never read and tokens are unchanged;
        # the STS-Net files' face slot is just another copy of the fused vector.
        self.face_label_dim = int(face_label_dim)
        if self.face_label_dim and not return_tokens:
            raise ValueError("face_label_dim > 0 only feeds the segment tokens; "
                             "it needs return_tokens=True (model.use_duration).")
 
        self.split            = split
        self.h5_path          = data_dir  # data_dir points at the .h5 file
        self.h5_labels_path   = h5_labels_path or DEFAULT_H5_LABELS_PATH
        print(f"Using h5 combinedlabels path =================== : {self.h5_labels_path}")
        # Warn once per sequence instead of once per __getitem__ call.
        self._warned_missing_labels = set()
        # Conditioning width; cross-checked against the h5 in _build_index().
        self.label_dim        = label_dim
        # Emit one token per annotated segment (+ its true duration) alongside the
        # frame-level features, for the duration predictor. Off by default so the
        # fixed-conditioning path is untouched.
        self.return_tokens    = return_tokens
        # self.seq_len          = seq_len
        self.train_ratio      = train_ratio
        self.val_ratio        = val_ratio
        self.features_to_load = features_to_load
        self.logger           = logger if logger is not None else logging.getLogger(__name__)
 
        # Per-frame soft-contact targets, sliced with the same [start:end] as the
        # poses so the two stay aligned by construction rather than by a lookup in
        # the training step.
        self.return_contacts = bool(return_contacts)
        self.h5_contacts_path = h5_contacts_path
        self._hf_contacts = None

        # FLAME expression coefficients for the face stream. Read through the same
        # features_to_load path as every pose stream -- the datamodule appends
        # FeatName.EXPRESSION when this is on -- so the only thing this flag adds
        # here is the startup check in _check_face_available().
        self.return_face = bool(return_face)
        self.face_dim = int(face_dim)
        # Whether jaw_pose is appended after the expression coefficients. Decided
        # from face_dim against the real h5 widths in _check_face_available():
        # face_dim == expression width -> expression only (the original setup),
        # face_dim == expression + jaw width -> [expression | jaw_pose 6D].
        self.face_with_jaw = False

        # How clip names in the pose h5 map onto keys in the labels h5; see
        # _labels_key(). None -> infer from the pose filename.
        if name_convention is None:
            name_convention = self._infer_name_convention(data_dir)
        if name_convention not in ("mocap", "phoenix"):
            raise ValueError(
                f"name_convention must be 'mocap' or 'phoenix', got {name_convention!r}")
        self.name_convention = name_convention

        self._build_index()
        self._load_normalization_stats(norm_stats_path)
        self._hf = h5py.File(self.h5_path, "r")
        if self.return_contacts:
            if not self.h5_contacts_path:
                raise ValueError("return_contacts=True needs h5_contacts_path")
            self._hf_contacts = h5py.File(self.h5_contacts_path, "r")
        self._cache_h5_keys()
        if self.return_face:
            self._check_face_available()

    def _check_face_available(self) -> None:
        """Fail at startup, not 40 minutes into an epoch.

        _load_chunk skips any feature whose h5 key is missing, so without this the
        expression would simply never appear in the batch and the first the model
        would know is the KeyError in get_losses. Checking the width here also
        catches a face_dim that disagrees with the file, and decides whether
        jaw_pose rides along (face_dim == expression + jaw width).
        """
        h5_key = FEAT_TO_H5_KEY[FeatName.EXPRESSION.value]
        if h5_key not in self._known_h5_keys:
            raise KeyError(
                f"model.use_face=True but '{h5_key}' is not in {self.h5_path}. "
                f"Available per-sequence keys: {sorted(self._known_h5_keys)}. "
                "Rebuild the pose h5 with the expression coefficients included."
            )
        seq_grp = self._hf[self.subject_id][next(iter(self.dataset_info))]
        n_expr = seq_grp[h5_key].shape[-1]
        n_jaw = seq_grp[JAW_H5_KEY].shape[-1] if JAW_H5_KEY in self._known_h5_keys else None

        if self.face_dim == n_expr:
            self.face_with_jaw = False
        elif n_jaw is not None and self.face_dim == n_expr + n_jaw:
            self.face_with_jaw = True
        else:
            options = f"{n_expr} (expression only)"
            if n_jaw is not None:
                options += f" or {n_expr + n_jaw} (expression + {n_jaw}-D jaw_pose)"
            raise ValueError(
                f"model.face_dim={self.face_dim} does not match {self.h5_path}: "
                f"'{h5_key}' is {n_expr}-dimensional"
                + (f" and '{JAW_H5_KEY}' is {n_jaw}-dimensional" if n_jaw is not None
                   else f" and there is no '{JAW_H5_KEY}'")
                + f". Set model.face_dim to {options} (the model sizes decoder_face from it)."
            )
        self.logger.info(
            f"[MocapHDF5Dataset] face stream ON: '{h5_key}' ({n_expr} coefficients)"
            + (f" + '{JAW_H5_KEY}' ({n_jaw}) = {self.face_dim}" if self.face_with_jaw else ""))

    # Names a top-level group may use for each split. PHOENIX ships train/dev/test,
    # this codebase says train/val/test, so both are accepted.
    _SPLIT_ALIASES = {
        "train": ("train",),
        "val":   ("val", "dev", "valid", "validation"),
        "test":  ("test",),
    }
    _ALL_SPLIT_NAMES = {a for v in _SPLIT_ALIASES.values() for a in v}

    def _labels_key(self, og_seq_name: str) -> str:
        """Map a clip name in the pose h5 onto its key in the labels h5.

        The two corpora name their clips differently, and the labels file is keyed
        by neither convention exactly:

        * mocap    "april17_a_pose_001_IngaLena.npz" -> "a_pose_001_IngaLena"
          The leading token is the recording-session prefix and is NOT part of the
          label key, so it is dropped along with the .npz suffix.

        * phoenix  "01April_2010_Thursday_heute-6694" -> unchanged
          The clip name IS the key. Splitting on "_" here would eat "01April" and
          nothing would ever match.
        """
        if self.name_convention == "phoenix":
            return og_seq_name[:-4] if og_seq_name.endswith(".npz") else og_seq_name
        # mocap: drop the leading session token, then the extension
        stripped = "_".join(og_seq_name.split("_")[1:])
        return stripped[:-4] if stripped.endswith(".npz") else stripped

    @staticmethod
    def _infer_name_convention(h5_path: str) -> str:
        """Pick the clip-naming convention from the pose file's name."""
        stem = str(h5_path).lower()
        if "phoenix" in stem:
            return "phoenix"
        return "mocap"

    def _resolve_group(self, dataset_info: dict) -> Tuple[str, bool]:
        """Pick the top-level HDF5 group this split should read.

        Two layouts are supported:

        * SUBJECT layout (the mocap files): one top-level group per subject, e.g.
          /ABCD01/<clip>.npz. The splits do not exist in the file, so they are cut
          out of the clip list here by train_ratio/val_ratio.

        * SPLIT layout (PHOENIX and anything else with real splits): the top level
          names the splits, e.g. /train/<clip>, /dev/<clip>, /test/<clip>. The
          file's own splits are authoritative and the ratios are IGNORED --
          re-slicing a curated split by position would silently destroy it, which
          is exactly what the old code did with `list(keys)[0]`: it kept the first
          group and dropped the rest.

        Returns (group_key, uses_native_splits).
        """
        keys = list(dataset_info.keys())
        if not keys:
            raise ValueError(f"[MocapHDF5Dataset] empty dataset_info in {self.h5_path}")
        lowered = {k.lower(): k for k in keys}

        # Split layout only if EVERY top-level group names a split -- otherwise a
        # subject who happens to be called "test" would flip the interpretation.
        if all(k.lower() in self._ALL_SPLIT_NAMES for k in keys):
            for alias in self._SPLIT_ALIASES[self.split]:
                if alias in lowered:
                    return lowered[alias], True
            raise KeyError(
                f"[MocapHDF5Dataset] {self.h5_path} is split-partitioned "
                f"({sorted(keys)}) but has no group for split '{self.split}'. "
                f"Accepted names: {sorted(self._SPLIT_ALIASES[self.split])}."
            )

        if len(keys) > 1:
            self.logger.warning(
                f"[MocapHDF5Dataset] {self.h5_path} has {len(keys)} top-level groups "
                f"{sorted(keys)[:5]}...; only '{keys[0]}' is read. Name the groups "
                "train/val/test to use real splits."
            )
        return keys[0], False

    def _build_index(self) -> None:
        """Build the sliding-window index from HDF5 attrs only.
        No array data is read here — just lengths from group attributes.
        """
        with h5py.File(self.h5_path, "r") as hf:
            dataset_info: dict = json.loads(hf.attrs["dataset_info"])

            # Top-level group for this split: a subject id, or the split itself.
            subject_id, self.native_splits = self._resolve_group(dataset_info)

            # Sync lengths from HDF5 attrs in case dataset_info has stale values
            for seq_name in dataset_info[subject_id]:
                if seq_name not in hf[subject_id]:
                    continue
                seq_grp = hf[subject_id][seq_name]
                dataset_info[subject_id][seq_name]["length"] = int(
                    seq_grp.attrs["length"]
                )
                self.lengths = {seq_name: dataset_info[subject_id][seq_name]["length"] for seq_name in dataset_info[subject_id]}

        # -- Load labels ---
        with h5py.File(self.h5_labels_path, "r") as hf_labels:
            hf_labels_keys = set(hf_labels.keys())
            n_seen = n_matched = 0
            for og_seq_name in dataset_info[subject_id]:
                # Naming differs per corpus; see _labels_key().
                #   mocap:   -sidor_..._0_001_IngaLena62.npz -> -sidor_..._0_001_IngaLena62
                #   phoenix: 01April_2010_Thursday_heute-6694 -> unchanged
                seq_name = self._labels_key(og_seq_name)
                n_seen += 1
                if seq_name not in hf_labels_keys:
                    self.logger.warning(f"Sequence name '{seq_name}' does not match any of the label filenames")
                    continue
                n_matched += 1
                seq_grp = hf_labels[seq_name]
                n_segments = seq_grp["n_segments"][()]
                dataset_info[subject_id][og_seq_name]["n_segments"] = int(n_segments)
                segment_or_not = np.zeros(dataset_info[subject_id][og_seq_name]["length"], dtype=bool)

                # if embeddings exist for all frames, don't zero pad, just fill in the labels
                dom = seq_grp["dom"][:]
                nondom = seq_grp["nondom"][:]
                body = seq_grp["body"][:]

                # Fail in the first second on a conditioning-width mismatch, rather than
                # with an opaque conv channel error minutes into training.
                for _name, _arr in (("dom", dom), ("nondom", nondom), ("body", body)):
                    _got = _arr.shape[1] if _arr.ndim > 1 else 1
                    if _got != self.label_dim:
                        raise ValueError(
                            f"[MocapHDF5Dataset] conditioning width mismatch for '{_name}': "
                            f"expected label_dim={self.label_dim}, h5 has {_got}.\n"
                            f"  labels file: {self.h5_labels_path}\n"
                            f"  Fix data.label_dim in configs/data/smplx.yaml (and it flows to "
                            f"the model via ${{data.label_dim}}), or point h5_labels_path at the "
                            f"matching file."
                        )
                stsnet_labels = np.zeros((dataset_info[subject_id][og_seq_name]["length"],), dtype=int)
                
                # if embeddings exist only for segments --> zero pad
                #dom = np.zeros((dataset_info[subject_id][og_seq_name]["length"], 512))
                #nondom = np.zeros((dataset_info[subject_id][og_seq_name]["length"], 512), dtype=int)
                #body = np.zeros((dataset_info[subject_id][og_seq_name]["length"], 512), dtype=int)
                #stsnet_labels = np.empty((dataset_info[subject_id][og_seq_name]["length"],), dtype=object)

                for segment in range(n_segments):
                    segment_i = f"seg_{str(segment)}"
                    if segment_i in seq_grp:
                        seg_labels = {
                            "dom": seq_grp[segment_i]["dom"][:],
                            "nondom": seq_grp[segment_i]["nondom"][:],
                            "body": seq_grp[segment_i]["body"][:],
                            "stsnet_labels": seq_grp[segment_i]["stsnet_labels"][:],
                            "start_frame": seq_grp[segment_i]["start_frame"][()],
                            "end_frame": seq_grp[segment_i]["end_frame"][()],
                            # Present in the *_fused labels file; marks real signs
                            # apart from the in-between transition segments.
                            "is_sign": bool(seq_grp[segment_i]["is_sign"][()])
                            if "is_sign" in seq_grp[segment_i] else True,
                        }
                        if self.face_label_dim:
                            _face = seq_grp[segment_i]["face"][:]
                            if _face.ndim != 2 or _face.shape[1] != self.face_label_dim:
                                raise ValueError(
                                    f"[MocapHDF5Dataset] data.face_label_dim={self.face_label_dim} but "
                                    f"'{seq_name}/{segment_i}/face' is {_face.shape} in "
                                    f"{self.h5_labels_path}. Only a labels file with a face-specific "
                                    "face slot (e.g. phoenix_mean_pose_6d_norm_cond.h5) supports it; "
                                    "set data.face_label_dim=0 otherwise.")
                            seg_labels["face"] = _face
                        dataset_info[subject_id][og_seq_name][segment_i] = seg_labels
                        # zero pad frames that are not in start-end range of any segment, 1 for frames that are in start-end range
                        segment_or_not[seq_grp[segment_i]["start_frame"][()]:seq_grp[segment_i]["end_frame"][()]] = True
                        
                        #dom[seq_grp[segment_i]["start_frame"][()]:seq_grp[segment_i]["end_frame"][()]] = seg_labels["dom"]
                        #nondom[seq_grp[segment_i]["start_frame"][()]:seq_grp[segment_i]["end_frame"][()]] = seg_labels["nondom"]
                        #body[seq_grp[segment_i]["start_frame"][()]:seq_grp[segment_i]["end_frame"][()]] = seg_labels["body"]
                        #stsnet_labels[seq_grp[segment_i]["start_frame"][()]:seq_grp[segment_i]["end_frame"][()]] = seg_labels["stsnet_labels"]
                    else:
                        self.logger.warning(f"Segment '{segment_i}' not found in labels for sequence '{og_seq_name}'")
                dataset_info[subject_id][og_seq_name]["segment_or_not"] = segment_or_not
                dataset_info[subject_id][og_seq_name]["dom"] = dom
                dataset_info[subject_id][og_seq_name]["nondom"] = nondom
                dataset_info[subject_id][og_seq_name]["body"] = body
                #dataset_info[subject_id][og_seq_name]["stsnet_labels"] = stsnet_labels

        # A total miss used to surface as thousands of warnings followed by an
        # empty dataset (n_segments stays 0, so every clip is skipped below).
        if n_seen and not n_matched:
            sample = list(dataset_info[subject_id])[:3]
            raise KeyError(
                f"[MocapHDF5Dataset] none of the {n_seen} clips in "
                f"{self.h5_path} (group '{subject_id}') matched a key in "
                f"{self.h5_labels_path}.\n"
                f"  name_convention = '{self.name_convention}'\n"
                f"  clip names   : {sample}\n"
                f"  derived keys : {[self._labels_key(x) for x in sample]}\n"
                f"  label keys   : {sorted(hf_labels_keys)[:3]}\n"
                "  Either the labels file is for a different corpus, or "
                "name_convention is wrong (set data.name_convention explicitly)."
            )

        self.subject_id = subject_id
        all_seqs = dataset_info[subject_id]
 
        # ── Split ─────────────────────────────────────────────────────────
        seq_names = list(all_seqs.keys())
        n_total = len(seq_names)

        if self.native_splits:
            # The file already IS this split; take it whole. train_val_test_split
            # is not consulted, because re-cutting a curated split by position
            # would mix its clips into the wrong partitions.
            if n_total == 0:
                raise Exception(
                    f"[MocapHDF5Dataset] split group '{subject_id}' in "
                    f"{self.h5_path} contains no sequences")
            chosen = seq_names
            self.logger.info(
                f"[MocapHDF5Dataset] {self.split}: using the file's own "
                f"'{subject_id}' group ({n_total} sequences); "
                f"train_val_test_split ignored")
        else:
            chosen = self._ratio_split(seq_names)

        self.dataset_info = {name: all_seqs[name] for name in chosen}
        self.logger.info(
            f"[MocapHDF5Dataset] {self.split}: {len(self.dataset_info)} sequences"
        )
        self._build_windows()

    def _ratio_split(self, seq_names: List[str]) -> List[str]:
        """Positional train/val/test cut for files that carry no splits of their own."""
        train_end = int(len(seq_names) * self.train_ratio)
        val_end   = train_end + int(len(seq_names) * self.val_ratio)

        n_total = len(seq_names)
        if self.split == "train":
            if train_end == 0:
                raise Exception(
                    f"[MocapHDF5Dataset] train partition is empty: "
                    f"train_ratio={self.train_ratio} of {n_total} sequences = 0"
                )
            chosen = seq_names[:train_end]
        elif self.split == "val":
            if train_end == val_end:
                raise Exception(
                    f"[MocapHDF5Dataset] val partition is empty: "
                    f"val_ratio={self.val_ratio} of {n_total} sequences = 0 "
                    f"(train_end={train_end}, val_end={val_end})"
                )
            chosen = seq_names[train_end:val_end]
        else:  # test
            if val_end == n_total:
                raise Exception(
                    f"[MocapHDF5Dataset] test partition is empty: "
                    f"test_ratio={1 - self.train_ratio - self.val_ratio} of {n_total} sequences = 0 "
                    f"(val_end={val_end} == n_total={n_total})"
                )
            chosen = seq_names[val_end:]
        return chosen

    def _build_windows(self) -> None:
        # ── Sliding-window index ───────────────────────────────────────────
        self.idx_per_seq: List[int]      = []
        self.idx2seq:     Dict[int, str] = {}
        self.sentences: List[Tuple[str, int, int]] = []  # (seq_name, first_sign_start, last_sign_end)
        n_clamped = 0

        for seq_name, seq_meta in self.dataset_info.items():
            n_segments = seq_meta.get("n_segments", 0)
            if n_segments == 0:
                continue

            # Collect all valid segment boundaries
            seg_starts, seg_ends = [], []
            for i in range(n_segments):
                seg_key = f"seg_{i}"
                if seg_key not in seq_meta:
                    continue
                seg_starts.append(int(seq_meta[seg_key]["start_frame"]))
                seg_ends.append(int(seq_meta[seg_key]["end_frame"]))

            if not seg_starts:
                self.logger.warning(f"[MocapHDF5Dataset] '{seq_name}' has no valid segments, skipping.")
                continue

            sentence_start = min(seg_starts)
            sentence_end   = max(seg_ends)
            # Labels built on a longer timeline than the pose clip (mocap_smplx_face_30fps.h5
            # cuts 1-9 frames off the end where the face capture stops) would put the
            # sentence past the last pose frame: the pose slices come back short while
            # LENGTH says otherwise, and collate pads the gap with zeros the mask treats
            # as real frames. Clamp to the clip; a no-op when the segments fit.
            clip_len = int(seq_meta.get("length", sentence_end))
            if sentence_end > clip_len:
                sentence_end = clip_len
                n_clamped += 1
            sentence_len   = sentence_end - sentence_start

            if sentence_len <= 0:
                self.logger.warning(f"[MocapHDF5Dataset] '{seq_name}' sentence has zero length, skipping.")
                continue
            self.sentences.append((seq_name, sentence_start, sentence_end))

        # New pipeline with masking: 1247 sentences, 3.782 h in train set
        # Old pipeline with fixed sequence length: train: 860942 windows, 3.987 h
        self.sentence_lengths = np.array(
            [e - s for _, s, e in self.sentences], dtype=np.int64
        )
        if n_clamped:
            self.logger.info(
                f"[MocapHDF5Dataset] {self.split}: {n_clamped} sentences ran past their clip's "
                "last frame (labels timeline longer than the pose clip); clamped to the clip")

        self.logger.info(
            f"[MocapHDF5Dataset] {self.split}: {len(self.sentences)} sentences, "
            f"{self._calc_hours():.3f} h"
            f"total frames: {self.sentence_lengths.sum()}"
        )
 

    def _cache_h5_keys(self) -> None:
        if not self.dataset_info:
            raise RuntimeError(...)
        first_seq = next(iter(self.dataset_info))
        self._known_h5_keys = set(self._hf[self.subject_id][first_seq].keys())
        self.logger.info(
            f"[MocapHDF5Dataset] available HDF5 keys: {sorted(self._known_h5_keys)}"
        )
 
    def _feat_name_to_h5_key(self, feat_name: str) -> Optional[str]:
        """Map a FeatName value to the corresponding HDF5 dataset key."""
        h5_key = FEAT_TO_H5_KEY.get(feat_name)
        if h5_key is None or h5_key not in self._known_h5_keys:
            return None
        return h5_key
 
    def _load_normalization_stats(self, stats_path: str) -> None:
        """Load per-feature mean/std from JSON — called at init."""
        if not stats_path:
            raise ValueError(
                "[MocapHDF5Dataset] norm_stats_path must be provided. "
                "Pass the path to norm_stats_6d_mocap.json produced by convert_mocap_to_hdf5.py."
            )
        with open(stats_path, "r") as f:
            raw = json.load(f)
        # The two corpora key their stats differently: norm_stats_6d_mocap.json by
        # FeatName ("smplx_mesh_body_pose"), norm_stats_6d_phoenix.json by h5 group
        # ("body_pose"). unnormalize() is called with FeatName values, so h5-group
        # keys are translated back through FEAT_TO_H5_KEY. Keys with no FeatName
        # (jaw_pose) are kept as they are.
        h5_to_feat = {h5: feat for feat, h5 in FEAT_TO_H5_KEY.items()}
        self.norm_stats = {
            h5_to_feat.get(key, key): {
                "mean": np.array(v["mean"], dtype=np.float32),
                "std":  np.array(v["std"],  dtype=np.float32),
            }
            for key, v in raw.items()
        }
        self.logger.info(f"[MocapHDF5Dataset] loaded norm stats from {stats_path}")
 
    def unnormalize(self, data: torch.Tensor, feat_name: str) -> torch.Tensor:
        """Undo z-score normalization — useful for visualization / evaluation."""
        assert hasattr(self, "norm_stats"), "call load_normalization_stats() first"
        mean = torch.tensor(self.norm_stats[feat_name]["mean"], dtype=torch.float32)
        std  = torch.tensor(self.norm_stats[feat_name]["std"],  dtype=torch.float32)
        return data * std.clamp(min=1e-8) + mean
 
    def __len__(self) -> int:
        return len(self.sentences)

    def _idx2segment(self, idx: int) -> Tuple[str, int, int]:
        return self.sentences[idx]

    def _build_tokens(self, seq_name: str, start_frame: int, end_frame: int):
        """One token per annotated segment overlapping [start_frame, end_frame).

        The duration-predictor analogue of Matcha's phoneme sequence: each token
        is the segment's pooled STS-Net triple (body|dom|nondom) and its duration
        is how many frames of this window the segment covers.

        Durations are derived from *boundaries* clipped to the window rather than
        from each segment's own length, so they tile the window exactly --
        ``generate_path`` builds a malformed alignment otherwise. A leading gap is
        absorbed by the first token; the last token is stretched to the window end.

        Returns:
            tokens:    (n_tokens, 3 * label_dim) float32
            durations: (n_tokens,) int64, all >= 1, summing to end_frame - start_frame
            is_sign:   (n_tokens,) bool
        """
        seq_meta = self.dataset_info[seq_name]
        seg_len = end_frame - start_frame
        n_segments = int(seq_meta.get("n_segments", 0))

        vecs, signs, edges = [], [], [0]
        for i in range(n_segments):
            seg = seq_meta.get(f"seg_{i}")
            if seg is None:
                continue
            s_f, e_f = int(seg["start_frame"]), int(seg["end_frame"])
            if e_f <= start_frame or s_f >= end_frame:
                continue  # segment lies outside this window

            # Mean-pool the segment into a single token. In the *_fused labels every
            # row of a segment is already the same pooled vector, so this is exactly
            # row 0 there; in *_interp the rows vary per frame and the mean is a
            # genuine summary rather than an arbitrary first frame.
            parts = (seg["body"].mean(axis=0), seg["dom"].mean(axis=0), seg["nondom"].mean(axis=0))
            if self.face_label_dim:
                parts += (seg["face"].mean(axis=0),)       # face tail, see face_label_dim
            vecs.append(np.concatenate(parts).astype(np.float32))
            signs.append(bool(seg.get("is_sign", True)))
            # Monotonic, clipped boundary: this token ends here.
            edges.append(min(max(e_f - start_frame, edges[-1]), seg_len))

        if not vecs:
            if seq_name not in self._warned_missing_labels:
                self._warned_missing_labels.add(seq_name)
                self.logger.warning(
                    f"[MocapHDF5Dataset] '{seq_name}' has no segments in "
                    f"[{start_frame}, {end_frame}); emitting one zero token."
                )
            return (
                torch.zeros(1, 3 * self.label_dim + self.face_label_dim, dtype=torch.float32),
                torch.tensor([seg_len], dtype=torch.long),
                torch.zeros(1, dtype=torch.bool),
            )

        edges[-1] = seg_len  # the final token always runs to the end of the window
        durations = np.diff(np.asarray(edges, dtype=np.int64))

        # Two segments clipped to the same boundary produce a 0-frame token, which
        # would hand the duration predictor a log(1e-8) target. Drop those; the sum
        # is unchanged because they contribute nothing.
        keep = durations > 0
        if not keep.any():  # window shorter than one frame per segment
            return (
                torch.from_numpy(vecs[0]).unsqueeze(0),
                torch.tensor([seg_len], dtype=torch.long),
                torch.tensor([signs[0]], dtype=torch.bool),
            )

        tokens = torch.from_numpy(np.stack(vecs)[keep])
        durations = torch.from_numpy(durations[keep])
        is_sign = torch.tensor(np.asarray(signs)[keep], dtype=torch.bool)

        assert int(durations.sum()) == seg_len, (
            f"[MocapHDF5Dataset] durations sum to {int(durations.sum())} but the "
            f"window is {seg_len} frames (seq={seq_name}, start={start_frame})"
        )
        return tokens, durations, is_sign

    def _load_chunk(self, seq_name: str, start_frame: int, end_frame: int) -> Dict[str, Any]:
        seg_len = end_frame - start_frame
        seq_grp = self._hf[self.subject_id][seq_name]

        data: Dict[str, Any] = {
            FeatName.SUBJECT_ID.value:  self.subject_id,
            FeatName.SEQUENCE_ID.value: seq_name,
            FeatName.START_FRAME.value: torch.tensor(start_frame, dtype=torch.long),
            FeatName.LENGTH.value:      torch.tensor(seg_len, dtype=torch.long),
        }

        for feat_name in self.features_to_load:
            if feat_name in ANNOTATION_FEATS:
                continue
            h5_key = self._feat_name_to_h5_key(feat_name)
            if h5_key is None:
                continue

            if feat_name in STATIC_FEATS:
                arr = seq_grp[h5_key][:]
                arr = np.tile(arr, (seg_len, 1))
            else:
                arr = seq_grp[h5_key][start_frame:end_frame]

            if feat_name == FeatName.EXPRESSION.value and self.face_with_jaw:
                # [expression | jaw_pose 6D], both z-scored, same frames.
                arr = np.concatenate(
                    (arr, seq_grp[JAW_H5_KEY][start_frame:end_frame]), axis=-1)

            data[feat_name] = torch.from_numpy(arr.astype(np.float32).copy())

        seq_meta = self.dataset_info[seq_name]
        if "segment_or_not" in seq_meta:
            # These are already frame-indexed, so slicing just works —
            # frames between signs will have segment_or_not=False and zero embeddings,
            # which is exactly the inter-sign gap behaviour you want.
            data["segment_or_not"] = torch.from_numpy(
                seq_meta["segment_or_not"][start_frame:end_frame].astype(bool).copy())
            data["dom_labels"]     = torch.from_numpy(
                seq_meta["dom"][start_frame:end_frame].astype(np.float32).copy())
            data["nondom_labels"]  = torch.from_numpy(
                seq_meta["nondom"][start_frame:end_frame].astype(np.float32).copy())
            data["body_labels"]    = torch.from_numpy(
                seq_meta["body"][start_frame:end_frame].astype(np.float32).copy())
        else:
            if seq_name not in self._warned_missing_labels:
                self._warned_missing_labels.add(seq_name)
                self.logger.warning(
                    f"[MocapHDF5Dataset] '{seq_name}' has no annotation labels; returning zero tensors."
                )
            data["segment_or_not"] = torch.zeros(seg_len, dtype=torch.bool)
            data["dom_labels"]     = torch.zeros(seg_len, self.label_dim, dtype=torch.float32)
            data["nondom_labels"]  = torch.zeros(seg_len, self.label_dim, dtype=torch.float32)
            data["body_labels"]    = torch.zeros(seg_len, self.label_dim, dtype=torch.float32)

        if self.return_contacts:
            # Same [start:end] as every pose stream above. Column order is the full
            # 61 pairs of sign-contacts' default_pairs(); the model selects its
            # keep_pairs subset, so the target file stays independent of that choice.
            cg = self._hf_contacts[self.subject_id].get(seq_name)
            if cg is None:
                raise KeyError(
                    f"'{seq_name}' missing from {self.h5_contacts_path}; the contacts "
                    f"file must be built from the same pose h5 as data_dir.")
            data["soft_dist"] = torch.from_numpy(
                cg["soft_dist"][start_frame:end_frame].astype(np.float32).copy())

        if self.return_tokens:
            tokens, durations, is_sign = self._build_tokens(seq_name, start_frame, end_frame)
            data["tokens"] = tokens
            data["durations"] = durations
            # Always emitted; SegmentEncoder.use_is_sign decides whether the model
            # actually reads it. Emitting it unconditionally is what makes that flag a
            # switch -- with it on, the encoder raises if the batch has no is_sign.
            data["is_sign"] = is_sign

        return data
 
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_name, start_frame, end_frame = self._idx2segment(idx)
        return self._load_chunk(seq_name, start_frame, end_frame)
 
    def _calc_hours(self, fps: int = 120) -> float:
        total_frames = sum(end - start for _, start, end in self.sentences)
        return total_frames / fps / 3600.0


def mocap_padded_collate(
    batch: List[Dict[str, Any]],
    pad_multiple: int = 1,
    label_pad: int = -100,
) -> Dict[str, Any]:
    """Pad variable-length samples to the longest sequence in the batch."""
    _TIME_KEYS = frozenset({
        FeatName.BODY.value, FeatName.ROT.value, FeatName.TRANS.value,
        FeatName.SHAPE.value, FeatName.LEFT_HAND.value, FeatName.RIGHT_HAND.value,
        # (T, face_dim) expression coefficients; padded on the frame axis exactly
        # like the pose streams, so mask/length handling is shared.
        FeatName.EXPRESSION.value,
        "segment_or_not", "dom_labels", "nondom_labels", "body_labels",
        # (T, n_pairs) soft-contact targets; padded on the frame axis exactly like
        # the pose streams, so mask/length handling is shared rather than duplicated.
        "soft_dist",
    })
    # Padded along the *segment* axis, not the frame axis (duration predictor).
    _TOKEN_KEYS = frozenset({"tokens", "durations", "is_sign"})

    B = len(batch)
    lengths = torch.as_tensor(
        [int(s[FeatName.LENGTH.value]) for s in batch], dtype=torch.long
    )
    max_len = int(lengths.max())
    if pad_multiple > 1:  # e.g. 2 ** num_downsamplings, for the U-Net
        max_len = -(-max_len // pad_multiple) * pad_multiple

    has_tokens = "tokens" in batch[0]
    if has_tokens:
        x_lengths = torch.as_tensor(
            [int(s["tokens"].shape[0]) for s in batch], dtype=torch.long
        )
        max_tokens = int(x_lengths.max())

    out: Dict[str, Any] = {}
    for key, proto in batch[0].items():
        if key == FeatName.LENGTH.value:
            continue

        if key in _TOKEN_KEYS:
            buf = torch.zeros((B, max_tokens, *proto.shape[1:]), dtype=proto.dtype)
            for i, sample in enumerate(batch):
                t = sample[key]
                buf[i, : t.shape[0]] = t
            out[key] = buf
            continue

        if key not in _TIME_KEYS:
            vals = [s[key] for s in batch]
            out[key] = torch.stack(vals) if torch.is_tensor(proto) else vals
            continue

        shape = (B, max_len, *proto.shape[1:])
        if proto.is_floating_point():
            buf = torch.zeros(shape, dtype=proto.dtype)
        else:
            buf = torch.full(shape, label_pad, dtype=proto.dtype)

        for i, sample in enumerate(batch):
            t = sample[key]
            buf[i, : t.shape[0]] = t
        out[key] = buf

    out[FeatName.LENGTH.value] = lengths
    out["mask"] = (
        torch.arange(max_len)[None, :] < lengths[:, None]
    ).unsqueeze(1)  # (B, 1, T), bool
    if has_tokens:
        out["x_lengths"] = x_lengths
    return out


class MocapSmplxDataset(torch.utils.data.Dataset):
    """Parent dataset class for loading in and processing data for MoFo framework"""

    def __init__(
        self,
        split: str,
        data_dir: List[str],
        logger: Logger | None = None,
        seq_len: int = 90,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        features_to_load: Optional[List[str]] = None,
        norm_stats_path: Optional[str] = None,
    ):
        """
        Args:
            split: train/val/test
            data_dir: directory to load the data from
            logger: logger to log the data
            seq_len: sequence length to load
            train_ratio: ratio of full dataset to use for train partition
            val_ratio: ratio of full dataset to use for val partition
            features_to_load: list of features to load (
                e.g. [FeatName.BODY.v, FeatName.AUDIO_SEPARATED.v, etc.])
            load_multiperson: whether to load multiperson data or not

        DATASET FORMAT

        Datasets are expected to come in to following folder structure, assuming
        data_dir as the root folder.
        list of files of the following structure:
            {month-date-of-recording}_{article-label}_{year-dddd}_{article-num-dd}_{article-title}_{sentence-num-dd}_{take-num-ddd}_{signer}.npz

        Further, a dataset.json file exist in the root folder.
        The file has the following format:
        {
            subject_id_1: {
                sentence_name_1: {
                    "length": num_frames_in_sequence.
                    "take_id": take_id, 
                    "id": full_file_id,
                    "text": "actual sentence text in Swedish".
                },
                sentence_name_2: {
                    ...
                }
            },
            subject_id_2: {
                ...
            },
            ...
        }
        """
        assert features_to_load is not None, "features_to_load must be specified"
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seq_len = seq_len
        self.norm_stats_path = norm_stats_path
        self.features_to_load = features_to_load
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        #### load dataset info; load all of the info jsons containing the lengths of each sequences
        # data_dir is a path, not list of paths
        self.data_dir = data_dir
        dataset_info = {}
        # for data_dir_path in data_dir:
        # in my case I have only one path
        #    self.data_dir.append(data_dir_path)
        subject_id = "ABCD01"
        for root, directory, files in os.walk(data_dir):
            if subject_id not in dataset_info:
                dataset_info[subject_id] = {}
            sentences = {}
            for name in files:
                file_path = os.path.join(root, name)
                # Load the file to count frames
                data = np.load(file_path, allow_pickle=True)
                # print("data keys: ", data.keys())
                #for key in data.files:  # .files is a list of all keys
                #    print(key) # body_pose lhand_pose rhand_pose jaw_pose leye_pose reye_pose betas expression global_orient transl
                sentences[name] = {
                    "length": data["body_pose"].shape[0],
                    "data_dir": root
                    }
            dataset_info[subject_id].update(sentences)

        #### create train, val, or test partition
        self.dataset_info = {}
        self.dataset_info[subject_id] = self._split_dataset(dataset_info)

        # print("self.dataset_info: ", self.dataset_info)

        #### create a mapping from global (dataset-level) frame index into sequences
        # idx2seq is a dictionary of {global_frame_index: (subject_id, sequence_name)}
        # idx_per_seq is a list of global_frame_index for each sequence
        # num_valid_segments is the total number of valid segments in the dataset
        seqs = [
            (subject_id, sequence_name)
            for subject_id in self.dataset_info.keys()
            for sequence_name in self.dataset_info[subject_id].keys()
        ]
        idx = 0
        self.idx2seq = {}
        self.idx_per_seq = []
        for seq in seqs:
            # print(f"seq {seq}")
            self.idx2seq[idx] = seq
            self.idx_per_seq.append(idx)
            idx += self.dataset_info[seq[0]][seq[1]]["length"] - self.seq_len + 1
        self.num_valid_segments = idx

        """seqs = list(self.dataset_info[subject_id].keys())  # all sequences for this subject
        idx = 0
        self.idx2seq = {}
        self.idx_per_seq = []

        for sequence_name in seqs:
            self.idx2seq[idx] = (subject_id, sequence_name)
            self.idx_per_seq.append(idx)
            seq_length = self.dataset_info[subject_id][sequence_name]["length"]
            idx += seq_length - self.seq_len + 1  # number of sliding windows in this sequence

        self.num_valid_segments = idx"""

        #### preload the dataset into memory
        self._preload_dataset()

    def _configure_dataset(self, data_dir: str):
        """load the dataset info json file
        Args:
            data_dir: directory to load the dataset info from
        """
        dataset_info_path = f"{data_dir}/dataset.json"
        with open(dataset_info_path, "r") as f:
            dataset_info = json.load(f)
        return dataset_info

    def _split_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """define the train/val/test split along sentences/recordings
        Args:
            dataset_info: dataset info dictionary (see DATASET FORMAT above)
        """
        dataset_info = dataset_info["ABCD01"]
        split_list = list(dataset_info.keys())
        train_partition = int(len(split_list) * self.train_ratio)
        val_partition = int(len(split_list) * self.val_ratio)
        test_partition = len(split_list) - train_partition - val_partition

        if self.split == "val":
            if val_partition == 0:
                raise Exception(
                    f"[base_dataloader.py] val partition can't be built: {self.val_ratio} of {len(split_list)} is 0."
                )
            split_dataset = {k: dataset_info[k] for k in split_list[train_partition : train_partition + val_partition]}
        elif self.split == "test":
            if test_partition == 0:
                test_ratio = 1.0 - self.train_ratio - self.val_ratio
                raise Exception(
                    f"[base_dataloader.py] test partition can't be built: {test_ratio} of {len(split_list)} is 0."
                )
            split_dataset = {k: dataset_info[k] for k in split_list[train_partition + val_partition :]}
        else:  # train
            if train_partition == 0:
                raise Exception(
                    f"[base_dataloader.py] train partition can't be built: {self.train_ratio} of {len(split_list)} is 0."
                )
            split_dataset = {k: dataset_info[k] for k in split_list[:train_partition]}

        # logging
        n_seqs = sum([len(v) for v in split_dataset.values()])
        self.logger.info(f"[base_dataloader.py] {self.split} dataset {len(split_dataset)} subjects, {n_seqs} sequences")

        return split_dataset

    def _get_chunk_data(self, subject_id: str, sequence_name: str, start_frame: int | None = None) -> Dict[str, torch.Tensor]:
        """load a single sequence from the dataset
        Args:
            subject_id: subject id of the person to load
            sequence_name: name of the sequence to load
            start_frame: start frame index of the sequence to load
            end_frame: end frame index of the sequence to load
        """
        data = {}

        # logic to set the start and end frame according to settings
        start_frame = start_frame if start_frame is not None else 0
        end_frame = start_frame + self.seq_len

        # metadata
        data[FeatName.SUBJECT_ID.value] = subject_id
        data[FeatName.SEQUENCE_ID.value] = sequence_name
        data[FeatName.START_FRAME.value] = torch.tensor(start_frame, dtype=torch.long)
        data[FeatName.LENGTH.value] = torch.tensor(end_frame - start_frame, dtype=torch.long)

        # features
        for feat_name in self.features_to_load:
            if subject_id not in self.dataset:
                self.logger.warning(f"[dataloader.py] subject {subject_id} not found in dataset")
                return data
            _data = np.copy(self.dataset[subject_id][sequence_name][feat_name])
            if feat_name == FeatName.TEXT.value:
                # sometimes the text annotations are not available just store an empty dict
                # for instance, we don't annotate the first 10 seconds of the first sequence
                # and there are cases where we don't annotate the end of the sequence
                # NOTE: for the free form text, you will need to pad them such that they are the same length text
                _data = _data.item()
                if _data is None:
                    self.logger.warning("[dataloader.py] text not found for sequence")
                    _data = ""
                elif "overview" in _data:
                    # only broad annotations are given (not per subsection)
                    data[feat_name] = _data["overview"]
                else:
                    # text annotations are stored in seconds at 10 second increments
                    seq_frames = sequence_name.split("--")[-1]
                    seq_start_frame = int(seq_frames.split("-")[0])
                    text_start_frame = int((start_frame / 30) // 10 * 10) * 30
                    text_frame_index = str(text_start_frame + seq_start_frame)
                    offset_in_seconds = (text_start_frame - start_frame) / 30
                    anno_to_select = FeatName.ANNO_MOVEMENT.value
                    if not text_frame_index in _data or anno_to_select not in _data[text_frame_index]:
                        self.logger.warning(f"[dataloader.py] text '{anno_to_select}' not found; change hardcoding")
                        _data = ""
                    else:
                        random_idx = np.random.randint(0, len(_data[text_frame_index][anno_to_select]))
                        _data = _data[text_frame_index][anno_to_select][random_idx]
                        data["text_offset_in_seconds"] = offset_in_seconds
                    data[feat_name] = _data
            elif feat_name == FeatName.TEXT_HOLISTIC.value:
                _data = _data.item()
                if _data is None:
                    self.logger.warning("[dataloader.py] text not found for sequence")
                    _data = ""
                anno_to_select = FeatName.ANNO_SCENE_MOOD.value
                if anno_to_select not in _data:
                    self.logger.warning(f"[dataloader.py] text '{anno_to_select}' not found; change hardcoding")
                    _data = ""
                else:
                    random_idx = np.random.randint(0, len(_data[anno_to_select]))
                    _data = _data[anno_to_select][random_idx]
                data[feat_name] = _data
            else:
                if not feat_name == "smplx_mesh_betas":
                    # load all the other smplx body features
                    #print("feat_name: ", feat_name)
                    #print("_data.shape: ", _data.shape)
                    #print("start_frame:end_frame ", start_frame, end_frame)
                    _data = _data[start_frame:end_frame]
                    
                    assert _data.shape[0] == self.seq_len, f"data shape {_data.shape[0]} is not seq len {self.seq_len}"
                data[feat_name] = torch.from_numpy(np.copy(_data))
        return data

    def _preload_dataset(self, quiet: bool = False) -> None:
        """Preload the dataset to avoid loading the same data multiple times"""
        dataset = {}
        seqs = [
            (subject_id, sequence_name)
            for subject_id in self.dataset_info.keys()
            for sequence_name in self.dataset_info[subject_id].keys()
        ]

        #### load the data by iterating through the sequences inside of the dataset info json
        for subject_id, sequence_name in tqdm(seqs, desc=f"Preloading {self.split} dataset", disable=quiet):
            if subject_id not in dataset:
                dataset[subject_id] = {}
            base_sequence_name = os.path.splitext(sequence_name)[0]
            data_dir = self.dataset_info[subject_id][sequence_name]["data_dir"]
            seq_dict = dataset[subject_id].get(sequence_name, {})
            for feat_name in self.features_to_load:
                if feat_name == FeatName.TEXT.value:
                    text_anno = str(self.dataset_info[subject_id][sequence_name]["text"])
                    if text_anno is None:
                        seq_dict[feat_name] = {}
                    elif text_anno.endswith(".json"):
                        with open(text_anno, "r") as f:
                            seq_dict[feat_name] = json.load(f)
                    else:
                        seq_dict[feat_name] = {"overview": text_anno}
                elif feat_name == FeatName.TEXT_HOLISTIC.value:
                    # NOTE: assumes a certain data format to calculate the text path for holistic
                    # the default file structure upon untaring the dataset
                    text_anno = str(self.dataset_info[subject_id][sequence_name]["text"])
                    if text_anno is None:
                        seq_dict[feat_name] = {}
                    else:
                        text_anno_separated = os.path.dirname(text_anno.split("text_annotations")[0][:-1])
                        text_anno_separated = os.path.join(text_anno_separated, "text_annotations_holistic")
                        text_basename = os.path.basename(text_anno)
                        text_basename = "--".join(os.path.splitext(text_basename)[0].split("--")[-2:])
                        holistic_text_path = f"{text_anno_separated}/{text_basename}.json"
                        if os.path.exists(holistic_text_path):
                            with open(holistic_text_path, "r") as f:
                                seq_dict[feat_name] = json.load(f)
                        else:
                            seq_dict[feat_name] = {}
                else:
                    # print(f"feat name {feat_name}") # smplx_mesh_body_pose
                    if "body" in feat_name:
                        data_key = "body_pose"
                    elif "right_hand" in feat_name:
                        data_key = "rhand_pose"
                    elif "left_hand" in feat_name:
                        data_key = "lhand_pose"
                    elif "transl" in feat_name:
                        data_key = "transl"
                    elif "orient" in feat_name:
                        data_key = "global_orient"
                    elif "jaw" in feat_name:
                        data_key = "jaw_pose"
                    elif "betas" in feat_name:
                        data_key = "betas"
                    # the head is locked so far
                    #elif "expression" in feat_name:
                    #    data_key = "expression"
                    #elif "right_eye" in feat_name:
                    #    data_key = "reye_pose"
                    #elif "left_eye" in feat_name:
                    #    data_key = "leye_pose"
                    # body_pose lhand_pose rhand_pose jaw_pose leye_pose reye_pose betas expression global_orient transl
                    data = np.load(
                        f"{data_dir}/{base_sequence_name}.npz", allow_pickle=True
                    )
                    seq_dict[feat_name] = data[data_key].astype(np.float32)
            dataset[subject_id][sequence_name] = seq_dict
        self.dataset = dataset
        self.logger.info(
            f"[base_dataloader.py] preloaded dataset with {len(seqs)} sequences, {self._calc_hours()} hours"
        )

    def __len__(self) -> int:
        return self.num_valid_segments

    def _calc_hours(self, fps: int = 60) -> float:
        """Calculates the number of hours in the dataset"""
        num_frames = self.num_valid_segments + self.seq_len - 1
        return num_frames / fps / 3600.0

    def _idx2segment(self, idx: int) -> Tuple[str, str, int]:
        """Get the subject_id, sequence_name, and frame_offset for a given index"""
        closest_idx = np.searchsorted(list(self.idx2seq.keys()), idx, side="right") - 1
        closest_idx = list(self.idx2seq.keys())[closest_idx]
        subject_id, sequence_name = self.idx2seq[closest_idx]
        frame_offset = idx - closest_idx
        return subject_id, sequence_name, frame_offset

    def _idx2segment(self, idx: int):
        # find the closest starting index for the sequence
        closest_idx = np.searchsorted(self.idx_per_seq, idx, side="right") - 1
        sequence_name = self.idx2seq[self.idx_per_seq[closest_idx]][1]
        frame_offset = idx - self.idx_per_seq[closest_idx]
        return "ABCD01", sequence_name, frame_offset

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        subject_id, sequence_name, frame_offset = self._idx2segment(idx)
        data = self._get_chunk_data(subject_id, sequence_name, start_frame=frame_offset)
        # print("data keys in getitem: ", data.keys())
        #  dict_keys(['subject_id', 'sequence_id', 'start_frame', 'length',
        #               'smplx_mesh_body_pose', 'smplx_mesh_global_orient', 'smplx_mesh_transl',
        #               'smplx_mesh_betas', 'smplx_mesh_left_hand_pose', 'smplx_mesh_right_hand_pose'])
        return data

from torch.utils.data import Sampler

class TokenBudgetSampler(Sampler):
    """Yields lists of indices whose padded cost stays under max_tokens."""

    def __init__(self, lengths, max_tokens, shuffle=True, seed=0,
                 num_replicas=1, rank=0, bucket_size_mult=100):
        self.lengths = np.asarray(lengths)
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_replicas, self.rank = num_replicas, rank
        self.pool = max_tokens * bucket_size_mult // int(self.lengths.mean())

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = rng.permutation(len(self.lengths)) if self.shuffle \
              else np.arange(len(self.lengths))
        # sort within a large window: near-uniform batches, still stochastic
        for s in range(0, len(idx), self.pool):
            window = idx[s:s + self.pool]
            idx[s:s + self.pool] = window[np.argsort(self.lengths[window], kind="stable")]

        batches, cur, cur_max = [], [], 0
        for i in idx:
            m = max(cur_max, self.lengths[i])
            if cur and m * (len(cur) + 1) > self.max_tokens:
                batches.append(cur)
                cur, cur_max = [i], self.lengths[i]
            else:
                cur, cur_max = cur + [i], m
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        b = self._batches()
        b = b[: len(b) - len(b) % self.num_replicas]  # keep ranks in step
        return iter(b[self.rank::self.num_replicas])

    def __len__(self):
        return len(self._batches()) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


class SmplxDataloader(LightningDataModule):
    def __init__(
        self,
        data_dir: str,  # = "/home/annkle/code/mocap-modeling/lightning-hydra-template/data/data/processed_tsg2"
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        # sequence_length: int = 90,
        load_multiperson: bool = True,
        kmeans: bool = False,
        conditional: bool = True,
        select_keyframes: bool = False,
        three_d: bool = False,
        codes: bool = False,
        latents: bool = False,
        h5_labels_path: Optional[str] = None,
        label_dim: int = 256,
        return_tokens: bool = False,
        frames_per_sample: int = 250,
        return_contacts: bool = False,
        h5_contacts_path: Optional[str] = None,
        return_face: bool = False,
        face_dim: int = 10,
        name_convention: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
        face_label_dim: int = 0,
    ):

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # Which dataset class to build. This used to be `"mocap" in data_dir`, which
        # made the choice depend on what the file happened to be CALLED: an h5 named
        # phoenix_smplx_face.h5 fell through to BaseDataset and died on
        # hparams.sequence_length. Decide on the actual format instead -- a single
        # .h5 file is the MocapHDF5Dataset layout, a directory of npz is BaseDataset.
        # Every shipped config points at an .h5, so this keeps the current behaviour
        # and stops depending on the name.
        self.mocap = str(self.hparams.data_dir).endswith((".h5", ".hdf5"))


    def setup(self, stage: Optional[str] = None):
        smplx_keys = [
            FeatName.BODY.value,
            FeatName.ROT.value,
            FeatName.TRANS.value,
            FeatName.SHAPE.value,
            FeatName.LEFT_HAND.value,
            FeatName.RIGHT_HAND.value,
        ]
        anno_keys = [
            FeatName.ANNO_DOM_FEATS.value,
            FeatName.ANNO_NONDOM_FEATS.value,
            FeatName.ANNO_BODY_FEATS.value,
            FeatName.ANNO_STSNET_LABELS.value
        ]
        # The face stream reads its coefficients through the ordinary
        # features_to_load path, so nothing about the batch changes with it off.
        if self.hparams.return_face:
            smplx_keys.append(FeatName.EXPRESSION.value)
        # Added annotations to the features to load
        self.features_to_load = smplx_keys + anno_keys
        if not self.data_train and not self.data_val and not self.data_test:
            if self.mocap:
                """self.data_train = MocapSmplxDataset(
                    split="train",
                    data_dir=self.hparams.data_dir,
                    seq_len=self.hparams.sequence_length,
                    features_to_load=self.features_to_load,
                    load_multiperson=self.hparams.load_multiperson,
                )"""
                self.data_train = MocapHDF5Dataset(
                    split="train",
                    data_dir=self.hparams.data_dir,
                    #seq_len=self.hparams.sequence_length,
                    features_to_load=self.features_to_load,
                    load_multiperson=self.hparams.load_multiperson,
                    h5_labels_path=self.hparams.h5_labels_path,
                    label_dim=self.hparams.label_dim,
                    return_tokens=self.hparams.return_tokens,
                    return_contacts=self.hparams.return_contacts,
                    h5_contacts_path=self.hparams.h5_contacts_path,
                    return_face=self.hparams.return_face,
                    face_dim=self.hparams.face_dim,
                    name_convention=self.hparams.name_convention,
                    norm_stats_path=self.hparams.norm_stats_path or DEFAULT_NORM_STATS_PATH,
                    face_label_dim=self.hparams.face_label_dim,
                )
                self.data_val = MocapHDF5Dataset(  # MocapSmplxDataset(
                    split="val",
                    data_dir=self.hparams.data_dir,
                    #seq_len=self.hparams.sequence_length,
                    features_to_load=self.features_to_load,
                    load_multiperson=self.hparams.load_multiperson,
                    h5_labels_path=self.hparams.h5_labels_path,
                    label_dim=self.hparams.label_dim,
                    return_tokens=self.hparams.return_tokens,
                    return_contacts=self.hparams.return_contacts,
                    h5_contacts_path=self.hparams.h5_contacts_path,
                    return_face=self.hparams.return_face,
                    face_dim=self.hparams.face_dim,
                    name_convention=self.hparams.name_convention,
                    norm_stats_path=self.hparams.norm_stats_path or DEFAULT_NORM_STATS_PATH,
                    face_label_dim=self.hparams.face_label_dim,
                )
                self.data_test = MocapHDF5Dataset(  # MocapSmplxDataset(
                    split="test",
                    data_dir=self.hparams.data_dir,
                    #seq_len=self.hparams.sequence_length,
                    features_to_load=self.features_to_load,
                    load_multiperson=self.hparams.load_multiperson,
                    h5_labels_path=self.hparams.h5_labels_path,
                    label_dim=self.hparams.label_dim,
                    return_tokens=self.hparams.return_tokens,
                    return_contacts=self.hparams.return_contacts,
                    h5_contacts_path=self.hparams.h5_contacts_path,
                    return_face=self.hparams.return_face,
                    face_dim=self.hparams.face_dim,
                    name_convention=self.hparams.name_convention,
                    norm_stats_path=self.hparams.norm_stats_path or DEFAULT_NORM_STATS_PATH,
                    face_label_dim=self.hparams.face_label_dim,
                )
            else:
                self.data_train = BaseDataset(
                    split="train",
                    data_dir=self.hparams.data_dir,
                    seq_len=self.hparams.sequence_length,
                    features_to_load=self.features_to_load,
                    load_multiperson=self.hparams.load_multiperson,
                )
                self.data_val = BaseDataset(
                    split="val",
                    data_dir=self.hparams.data_dir,
                    seq_len=self.hparams.sequence_length,
                    features_to_load=self.features_to_load,
                    load_multiperson=self.hparams.load_multiperson,
                )
                self.data_test = BaseDataset(
                    split="test",
                    data_dir=self.hparams.data_dir,
                    seq_len=self.hparams.sequence_length,
                    features_to_load=self.features_to_load,
                    load_multiperson=self.hparams.load_multiperson,
                )


    def train_dataloader(self):
        world_size = self.trainer.world_size # getattr(self.trainer, "world_size", 1)
        rank = self.trainer.global_rank # getattr(self.trainer, "global_rank", 0)

        train_dl = DataLoader(
            dataset=self.data_train,  # .__getitem__(0).unsqueeze(0),  # Use a single datapoint for debugging
            # batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # shuffle=True,   # False for overfit
            batch_sampler=TokenBudgetSampler(
                lengths=self.data_train.sentence_lengths,
                # Changed for the 120fps -> 30fps downsampling (mocap_smplx_30fps.h5
                # and the *_30fps label files, built by downsample_30fps.py).
                #
                # TokenBudgetSampler's budget is counted in FRAMES, so it has to
                # track the frame rate or the batches silently change shape. The old
                # constant was batch_size * 1000, chosen when the mean sentence ran
                # ~938 frames at 120fps -- i.e. "about batch_size sentences per
                # batch". Decimating to 30fps makes the mean ~234 frames, so leaving
                # 1000 in place would pack ~4x more sentences into every batch and
                # cut the epoch from ~77 optimizer steps to ~19. That trades the
                # speedup away: the epoch gets 4x faster but takes 4x fewer steps,
                # which is a wash for convergence. frames_per_sample keeps the step
                # count and batch composition where they were and makes each step
                # 4x cheaper instead, which is the actual win.
                #
                # Set frames_per_sample=1000 in the data config to run 120fps data,
                # 120 for PHOENIX (25 fps); see configs/data/smplx.yaml.
                max_tokens=self.hparams.batch_size * self.hparams.frames_per_sample,
                shuffle=True,
                seed=self.hparams.get("seed", 0),
                num_replicas=world_size,
                rank=rank,
            ),
            collate_fn=mocap_padded_collate,
            persistent_workers=self.hparams.num_workers > 0,
            #worker_init_fn=seed_worker,
            #generator=g,
        )
        # train_dl.dataset.standardize(self.scalers)
        print(f"Train DataLoader size: {len(train_dl.dataset)}")
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,  # True for overfit
            collate_fn=mocap_padded_collate,
            persistent_workers=self.hparams.num_workers > 0,
            #worker_init_fn=seed_worker,
            #generator=g,
        )
        # val_dl.dataset.standardize(self.scalers)

        # print the size of the val loade  
        print(f"Validation DataLoader size: {len(val_dl.dataset)}")   # 174150
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,  # # True for overfit
            collate_fn=mocap_padded_collate,
            persistent_workers=self.hparams.num_workers > 0,
            #worker_init_fn=seed_worker,
            #generator=g,
        )
        # test_dl.dataset.standardize(self.scalers)
        print(f"Test DataLoader size: {len(test_dl.dataset)}")
        return test_dl
    


# ============
# Embody3D dataset tests:
# ============

def smoke_test(h5_path: str, norm_stats_path: str, features_to_load: list):
    print("=" * 50)
    print("SMOKE TEST")
    print("=" * 50)

    # ── Init all three splits ──────────────────────
    for split in ["train", "val", "test"]:
        ds = MocapHDF5Dataset(
            split=split,
            data_dir=h5_path,
            seq_len=300,
            features_to_load=features_to_load,
            norm_stats_path=norm_stats_path,
        )
        print(f"\n{split}: {len(ds)} segments")

    # ── Single item ───────────────────────────────
    ds_train = MocapHDF5Dataset(
        split="train",
        data_dir=h5_path,
        seq_len=300,
        features_to_load=features_to_load,
        norm_stats_path=norm_stats_path,
    )

    item = ds_train[0]
    print("\nKeys in item:", list(item.keys()))
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                  f"min={v.min():.3f}, max={v.max():.3f}")
        else:
            print(f"  {k}: {v}")

    # ── Check a few random indices ─────────────────
    print("\nRandom index checks:")
    for idx in [0, 1, len(ds_train) // 2, len(ds_train) - 1]:
        item = ds_train[idx]
        for feat in features_to_load:
            if feat in item:
                if feat != "betas":  # betas are static, no time axis
                    assert item[feat].shape[0] == ds_train.seq_len, \
                        f"Wrong seq_len at idx {idx}, feat {feat}: {item[feat].shape}"
        print(f"  idx {idx}: ok")

    print("\nSmoke test passed ")


def dataloader_test(h5_path: str, norm_stats_path: str, features_to_load: list):
    print("=" * 50)
    print("DATALOADER TEST")
    print("=" * 50)

    ds = MocapHDF5Dataset(
        split="train",
        data_dir=h5_path,
        seq_len=300,
        features_to_load=features_to_load,
        norm_stats_path=norm_stats_path,
    )

    # Test with num_workers=0 first (no multiprocessing)
    print("\nTesting num_workers=0...")
    dl = DataLoader(ds, batch_size=4, num_workers=0, shuffle=True)
    batch = next(iter(dl))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    print("  num_workers=0: ok ✓")

    # Test with num_workers>0 (multiprocessing — this is where HDF5 bugs appear)
    print("\nTesting num_workers=4...")
    dl = DataLoader(ds, batch_size=4, num_workers=4, shuffle=True)
    for i, batch in enumerate(dl):
        if i == 3:  # just a few batches is enough
            break
    print("  num_workers=4: ok ✓")

    # Check batch shapes are correct
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            assert v.shape[0] == 4, f"Expected batch size 4, got {v.shape[0]}"
            if k not in (FeatName.SUBJECT_ID.value, FeatName.SEQUENCE_ID.value,
                         FeatName.START_FRAME.value, FeatName.LENGTH.value):
                assert v.shape[1] == 300, f"Expected seq_len 300, got {v.shape[1]}"
    print("  batch shapes: ok ✓")

    print("\nDataLoader test passed ✓")


def integrity_test(h5_path: str, norm_stats_path: str, features_to_load: list):
    print("=" * 50)
    print("INTEGRITY TEST")
    print("=" * 50)

    ds = MocapHDF5Dataset(
        split="train",
        data_dir=h5_path,
        seq_len=300,
        features_to_load=features_to_load,
        norm_stats_path=norm_stats_path,
    )

    pose_feats = [f for f in features_to_load if FeatName.SHAPE.value != f]

    # Sample 10 random items and check stats
    indices = np.random.choice(len(ds), size=10, replace=False)
    all_means = {f: [] for f in pose_feats}
    all_stds  = {f: [] for f in pose_feats}

    print("Sampling 10 random items...")
    for idx in indices:
        item = ds[idx]
        for feat in pose_feats:
            if feat in item and isinstance(item[feat], torch.Tensor):
                all_means[feat].append(item[feat].float().mean().item())
                all_stds[feat].append(item[feat].float().std().item())

    print("\nNormalized data statistics (should be approx mean≈0, std≈1):")
    any_failed = False
    for feat in pose_feats:
        if not all_means[feat]:
            continue
        mean_of_means = np.mean(all_means[feat])
        mean_of_stds  = np.mean(all_stds[feat])
        status = "✓" if abs(mean_of_means) < 0.5 and 0.3 < mean_of_stds < 3.0 else "✗ SUSPICIOUS"
        if "✗" in status:
            any_failed = True
        print(f"  {feat}: mean={mean_of_means:.3f}, std={mean_of_stds:.3f}  {status}")

    # Check for NaNs and Infs
    print("\nChecking for NaN/Inf...")
    for idx in indices[:10]:
        item = ds[idx]
        for feat in pose_feats:
            if feat in item and isinstance(item[feat], torch.Tensor):
                assert not torch.isnan(item[feat]).any(), \
                    f"NaN found in {feat} at idx {idx}"
                assert not torch.isinf(item[feat]).any(), \
                    f"Inf found in {feat} at idx {idx}"
    print("  No NaN/Inf found ✓")

    # Check sliding window continuity — adjacent indices should be consecutive frames
    print("\nChecking sliding window continuity...")
    item_a = ds[100]
    item_b = ds[101]
    assert item_a[FeatName.SEQUENCE_ID.value] == item_b[FeatName.SEQUENCE_ID.value] or \
           item_b[FeatName.START_FRAME.value].item() == 0, \
        "Unexpected jump between adjacent indices"
    print("  Sliding window continuity: ok ✓")

    if any_failed:
        print("\n⚠ Some features have suspicious statistics — check normalization")
    else:
        print("\nIntegrity test passed ✓")


def check(name, condition, detail=""):
    PASS = "\033[92m PASS\033[0m"
    FAIL = "\033[91m FAIL\033[0m"
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition

if __name__ == "__main__":

    H5_PATH = "/pfs/lustrep1/scratch/project_465002480/lightning-hydra-template/data/mocap_smplx.h5"
    # "/pfs/lustrep1/scratch/project_465002480/lightning-hydra-template/data/embody3d_hand_contacts.h5"
    STATS_PATH = "/pfs/lustrep1/scratch/project_465002480/lightning-hydra-template/data/norm_stats_6d_mocap.json" # norm_stats_6d_embody3d.json"
    H5_LABELS_PATH = "/pfs/lustrep1/scratch/project_465002480/lightning-hydra-template/data/combined_data_labels.h5"
    
    print("\n── Building dataset ─────────────────────────────────────────────")
    ds = MocapHDF5Dataset(
        split="train",
        data_dir=H5_PATH,
        h5_labels_path=H5_LABELS_PATH,
        norm_stats_path=STATS_PATH,
        features_to_load=[
            FeatName.BODY.value,
            FeatName.LEFT_HAND.value,
            FeatName.RIGHT_HAND.value,
            FeatName.ROT.value,
            FeatName.TRANS.value,
            FeatName.SHAPE.value,
        ],
    )
 
    print(f"\n── Dataset-level checks ─────────────────────────────────────────")
    check("dataset non-empty",          len(ds) > 0,       f"{len(ds)} sentences")
    check("sentences list populated",   len(ds.sentences) > 0)

    # Sample individual items before batching
    lengths = []
    for i in range(min(20, len(ds))):
        item = ds[i]
        lengths.append(item[FeatName.LENGTH.value].item())
    check("individual items load without error", True)
    check("lengths vary (not all identical)",
          len(set(lengths)) > 1,
          f"unique lengths seen: {sorted(set(lengths))}")
    check("no zero-length items",  all(l > 0 for l in lengths))
 
    print(f"\n── DataLoader / collate checks ──────────────────────────────────")
    dl = DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        collate_fn=mocap_padded_collate,
        num_workers=0,
    )
 
    all_ok = True
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= 6:
            break
        print(f"\n  Batch {batch_idx}  ({'×'.join(str(s) for s in batch[FeatName.BODY.value].shape)})")
 
        y_lengths = batch[FeatName.LENGTH.value]   # (B,)
        B         = y_lengths.shape[0]
        T         = batch[FeatName.BODY.value].shape[1]   # padded time dim
 
        # ── Shape checks ──────────────────────────────────────────────────
        all_ok &= check("y_lengths shape is (B,)",     y_lengths.shape == (B,),
                        str(y_lengths.shape))
        all_ok &= check("y_lengths all > 0",           (y_lengths > 0).all().item())
        all_ok &= check("y_lengths all <= T_padded",   (y_lengths <= T).all().item(),
                        f"max_len={y_lengths.max().item()}, T={T}")
        all_ok &= check("batch has length variation",  y_lengths.unique().numel() > 1,
                        f"lengths={y_lengths.tolist()}")
 
        # ── Mask checks ───────────────────────────────────────────────────
        mask = sequence_mask(y_lengths, T)          # (B, T) bool, True=valid
        mask_float = mask.float().unsqueeze(-1)     # (B, T, 1)
 
        body = batch[FeatName.BODY.value]           # (B, T, C)
 
        # Padding frames should be exactly zero (collate pads with zeros)
        padding_mask  = ~mask                       # True = padding
        padding_vals  = body[padding_mask]          # all padding frames
        if padding_vals.numel() > 0:
            all_ok &= check("padding frames are zero",
                            padding_vals.abs().max().item() == 0.0,
                            f"max_abs_in_padding={padding_vals.abs().max().item():.4f}")
 
        # Real frames should not all be zero
        real_vals = body[mask]
        all_ok &= check("real frames are not all zero",
                        real_vals.abs().max().item() > 0,
                        f"max_abs_in_real={real_vals.abs().max().item():.4f}")
 
        # ── Statistics on real frames only ────────────────────────────────
        FEAT_KEYS = ["body_pose", "lhand_pose", "rhand_pose"]
        for feat_key in FEAT_KEYS:
            if feat_key not in batch:
                continue
            feat = batch[feat_key]                  # (B, T, C)
            real = feat[mask]                       # (N_real_frames * C,)
            mean = real.mean().item()
            std  = real.std().item()
            min_ = real.min().item()
            max_ = real.max().item()
            all_ok &= check(f"{feat_key}: mean in [-3, 3]",   -3 < mean < 3,
                            f"mean={mean:.3f}")
            all_ok &= check(f"{feat_key}: std in (0.01, 5)",  0.01 < std < 5,
                            f"std={std:.3f}")
            all_ok &= check(f"{feat_key}: no NaNs",           not real.isnan().any().item())
            all_ok &= check(f"{feat_key}: no Infs",           not real.isinf().any().item())
            all_ok &= check(f"{feat_key}: range reasonable",  min_ > -20 and max_ < 20,
                            f"min={min_:.2f} max={max_:.2f}")
 
        # ── Label checks ──────────────────────────────────────────────────
        seg = batch["segment_or_not"]               # (B, T) bool
        all_ok &= check("segment_or_not has some True frames",
                        seg[mask].any().item(),
                        f"sign_frame_ratio={seg[mask].float().mean().item():.2f}")
        all_ok &= check("segment_or_not False in padding",
                        not seg[~mask].any().item() if (~mask).any() else True)
 
        dom = batch["dom_labels"]                   # (B, T, 512)
        all_ok &= check("dom_labels: no NaNs",      not dom[mask].isnan().any().item())
        all_ok &= check("dom_labels non-zero in sign frames",
                        dom[mask & seg].abs().sum().item() > 0
                        if (mask & seg).any() else True)
 
        # ── fix_len_compatibility compatibility check ──────────────────────
        # T should be padded to nearest 2^N — check 2 and 3 downsamplings
        for n in [2, 3]:
            factor = 2 ** n
            T_fixed = int(np.ceil(T / factor)) * factor
            if T == T_fixed:
                check(f"T={T} already divisible by 2^{n}={factor}", True)
                break
        else:
            # Not yet padded — remind user to apply fix_len_compatibility in training_step
            check("T divisible by 2^2 or 2^3 (apply fix_len_compatibility in training_step)",
                  False, f"T={T}")
 
    print(f"\n── Summary ──────────────────────────────────────────────────────")
    if all_ok:
        print("  \033[92mAll checks passed.\033[0m")
    else:
        print("  \033[91mSome checks FAILED — see above.\033[0m")

    """try:
        FEATURES = [
            FeatName.BODY.value, FeatName.ROT.value, FeatName.TRANS.value,
            FeatName.SHAPE.value, FeatName.LEFT_HAND.value, FeatName.RIGHT_HAND.value
            ]

        smoke_test(H5_PATH, STATS_PATH, FEATURES)
        dataloader_test(H5_PATH, STATS_PATH, FEATURES)
        integrity_test(H5_PATH, STATS_PATH, FEATURES)

        # Clean up multiprocessing settings for Slurm compatibility
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)  # at the top of the script
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # non-zero exit code marks job as failed in Slurm"""


"""# setup
### iterate through the dataloader
for i, batch in enumerate(dataloader):
        print(f".....[batch {i}]:.....")
        print(f"subject_id: {batch['subject_id'][0]}")
        print(f"sequence_id: {batch['sequence_id'][0]}")
        print(f"loaded batch keys: {batch.keys()}")

datamodule = TSG2DataModule(
    data_dir="/nfs/deepspeech/home/annkle/lightning-hydra-template/data/data/2processed_mocap_body_and_fingers",
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    sequence_length=90,
)
datamodule.setup()
train_loader = datamodule.train_dataloader()


# build a training-like batch
batch = next(iter(train_loader))
x0 = batch['y']
print(x0.shape)  # ([1, 129, 90])"""
