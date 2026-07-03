import importlib
import os
import sys

import torch
import torch.nn as nn


def monkey_patch_p3sam():
    """Patch P3-SAM model construction to use the shared Hugging Face cache."""
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    hunyuan_part_root = os.path.abspath(
        os.path.join(current_dir, "../../../thirdparty/Hunyuan3D-Part")
    )
    p3sam_root = os.path.join(hunyuan_part_root, "P3-SAM")
    partgen_root = os.path.join(hunyuan_part_root, "XPart/partgen")

    for path in [p3sam_root, partgen_root]:
        if path not in sys.path:
            sys.path.insert(0, path)

    from models import sonata

    p3sam_model = importlib.import_module("model")

    def build_P3SAM(self):
        self.sonata = sonata.load(
            "sonata",
            repo_id="facebook/sonata",
            download_root=os.path.expanduser(
                "~/.cache/huggingface/hub/sonata"
            ),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1232, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
        self.transform = sonata.transform.default()

        self.seg_mlp_1 = nn.Sequential(
            nn.Linear(512 + 3 + 3, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )
        self.seg_mlp_2 = nn.Sequential(
            nn.Linear(512 + 3 + 3, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )
        self.seg_mlp_3 = nn.Sequential(
            nn.Linear(512 + 3 + 3, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

        self.seg_s2_mlp_g = nn.Sequential(
            nn.Linear(512 + 3 + 3 + 3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.seg_s2_mlp_1 = nn.Sequential(
            nn.Linear(512 + 3 + 3 + 3 + 256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.seg_s2_mlp_2 = nn.Sequential(
            nn.Linear(512 + 3 + 3 + 3 + 256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.seg_s2_mlp_3 = nn.Sequential(
            nn.Linear(512 + 3 + 3 + 3 + 256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        self.iou_mlp = nn.Sequential(
            nn.Linear(512 + 3 + 3 + 3 + 256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.iou_mlp_out = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )
        self.iou_criterion = torch.nn.MSELoss()

    p3sam_model.build_P3SAM = build_P3SAM
