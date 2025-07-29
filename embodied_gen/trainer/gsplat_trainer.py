# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# Part of the code comes from https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
# Both under the Apache License, Version 2.0.


import json
import os
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from fused_ssim import fused_ssim
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from embodied_gen.data.datasets import PanoGSplatDataset
from embodied_gen.utils.config import GsplatTrainConfig
from embodied_gen.utils.gaussian import (
    create_splats_with_optimizers,
    export_splats,
    resize_pinhole_intrinsics,
    set_random_seed,
)


class Runner:
    """Engine for training and testing from gsplat example.

    Code from https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
    """

    def __init__(
        self,
        local_rank: int,
        world_rank,
        world_size: int,
        cfg: GsplatTrainConfig,
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        self.trainset = PanoGSplatDataset(cfg.data_dir, split="train")
        self.valset = PanoGSplatDataset(
            cfg.data_dir, split="train", max_sample_num=6
        )
        self.testset = PanoGSplatDataset(cfg.data_dir, split="eval")
        self.scene_scale = cfg.scene_scale

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.trainset.points,
            self.trainset.points_rgb,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=None,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            self.device
        )
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        image_ids = kwargs.pop("image_ids", None)

        colors = torch.cat(
            [self.splats["sh0"], self.splats["shN"]], 1
        )  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = (
                "antialiased" if self.cfg.antialiased else "classic"
            )
        if camera_model is None:
            camera_model = self.cfg.camera_model

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            image_ids = data["image_id"].to(device)
            masks = (
                data["mask"].to(device) if "mask" in data else None
            )  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            # sh schedule
            sh_degree_to_use = min(
                step // cfg.sh_degree_interval, cfg.sh_degree
            )

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2),
                padding="valid",
            )
            loss = (
                l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            )
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(
                    depths > 0.0, 1.0 / depths, torch.zeros_like(depths)
                )
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += (
                    cfg.opacity_reg
                    * torch.sigmoid(self.splats["opacities"]).mean()
                )
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()

            desc = (
                f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            )
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if (
                world_rank == 0
                and cfg.tb_every > 0
                and step % cfg.tb_every == 0
            ):
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means"]), step
                )
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar(
                        "train/depthloss", depthloss.item(), step
                    )
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors], dim=2)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if (
                step in [i - 1 for i in cfg.save_steps]
                or step == max_steps - 1
            ):
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                torch.save(
                    data,
                    f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt",
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                sh0 = self.splats["sh0"]
                shN = self.splats["shN"]
                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert (
                    cfg.packed
                ), "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_video(step)

    @torch.no_grad()
    def eval(
        self,
        step: int,
        stage: str = "val",
        canvas_h: int = 512,
        canvas_w: int = 1024,
    ):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            masks = data["mask"].to(device) if "mask" in data else None

            pixels = pixels.permute(0, 3, 1, 2)  # NHWC -> NCHW
            pixels = F.interpolate(pixels, size=(canvas_h, canvas_w // 2))

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = colors.permute(0, 3, 1, 2)  # NHWC -> NCHW
            colors = F.interpolate(colors, size=(canvas_h, canvas_w // 2))
            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                canvas = torch.cat(canvas_list, dim=2).squeeze(0)
                canvas = canvas.permute(1, 2, 0)  # CHW -> HWC
                canvas = (canvas * 255).to(torch.uint8).cpu().numpy()
                cv2.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas[..., ::-1],
                )
                metrics["psnr"].append(self.psnr(colors, pixels))
                metrics["ssim"].append(self.ssim(colors, pixels))
                metrics["lpips"].append(self.lpips(colors, pixels))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {
                k: torch.stack(v).mean().item() for k, v in metrics.items()
            }
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(
                f"{self.stats_dir}/{stage}_step{step:04d}.json", "w"
            ) as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_video(
        self, step: int, canvas_h: int = 512, canvas_w: int = 1024
    ):
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=1, shuffle=False, num_workers=1
        )

        images_cache = []
        depth_global_min, depth_global_max = float("inf"), -float("inf")
        for data in testloader:
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = resize_pinhole_intrinsics(
                data["K"].squeeze(),
                raw_hw=(data["image_h"].item(), data["image_w"].item()),
                new_hw=(canvas_h, canvas_w // 2),
            ).to(self.device)
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks[None, ...],
                width=canvas_w // 2,
                height=canvas_h,
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            colors = (colors * 255).to(torch.uint8).cpu().numpy()
            depths = renders[0, ..., 3:4]  # [H, W, 1], tensor in device.
            images_cache.append([colors, depths])
            depth_global_min = min(depth_global_min, depths.min().item())
            depth_global_max = max(depth_global_max, depths.max().item())

        video_path = f"{self.render_dir}/video_step{step}.mp4"
        writer = imageio.get_writer(video_path, fps=30)
        for rgb, depth in images_cache:
            depth_normalized = torch.clip(
                (depth - depth_global_min)
                / (depth_global_max - depth_global_min + 1e-8),
                0,
                1,
            )
            depth_normalized = (
                (depth_normalized * 255).to(torch.uint8).cpu().numpy()
            )
            depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            image = np.concatenate([rgb, depth_map], axis=1)
            writer.append_data(image)

        writer.close()


def entrypoint(
    local_rank: int, world_rank, world_size: int, cfg: GsplatTrainConfig
):
    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat(
                [ckpt["splats"][k] for ckpt in ckpts]
            )
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_video(step=step)
    else:
        runner.train()
        runner.render_video(step=cfg.max_steps - 1)


if __name__ == "__main__":
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            GsplatTrainConfig(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            GsplatTrainConfig(
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(entrypoint, cfg, verbose=True)
