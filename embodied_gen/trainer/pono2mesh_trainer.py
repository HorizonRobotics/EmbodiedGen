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


from embodied_gen.utils.monkey_patches import monkey_patch_pano2room

monkey_patch_pano2room()

import os

import cv2
import numpy as np
import torch
import trimesh
from equilib import cube2equi, equi2pers
from kornia.morphology import dilation
from PIL import Image
from embodied_gen.models.sr_model import ImageRealESRGAN
from embodied_gen.utils.config import Pano2MeshSRConfig
from embodied_gen.utils.geometry import compute_pinhole_intrinsics
from embodied_gen.utils.log import logger
from thirdparty.pano2room.modules.geo_predictors import PanoJointPredictor
from thirdparty.pano2room.modules.geo_predictors.PanoFusionDistancePredictor import (
    PanoFusionDistancePredictor,
)
from thirdparty.pano2room.modules.inpainters import PanoPersFusionInpainter
from thirdparty.pano2room.modules.mesh_fusion.render import (
    features_to_world_space_mesh,
    render_mesh,
)
from thirdparty.pano2room.modules.mesh_fusion.sup_info import SupInfoPool
from thirdparty.pano2room.utils.camera_utils import gen_pano_rays
from thirdparty.pano2room.utils.functions import (
    depth_to_distance,
    get_cubemap_views_world_to_cam,
    resize_image_with_aspect_ratio,
    rot_z_world_to_cam,
    tensor_to_pil,
)


class Pano2MeshSRPipeline:
    """Pipeline for converting panoramic RGB images into 3D mesh representations.

    This class integrates depth estimation, inpainting, mesh conversion, multi-view mesh repair,
    and 3D Gaussian Splatting (3DGS) dataset generation.

    Args:
        config (Pano2MeshSRConfig): Configuration object containing model and pipeline parameters.

    Example:
        ```py
        from embodied_gen.trainer.pono2mesh_trainer import Pano2MeshSRPipeline
        from embodied_gen.utils.config import Pano2MeshSRConfig

        config = Pano2MeshSRConfig()
        pipeline = Pano2MeshSRPipeline(config)
        pipeline(pano_image='example.png', output_dir='./output')
        ```
    """

    def __init__(self, config: Pano2MeshSRConfig) -> None:
        """Initializes the pipeline with models and camera poses.

        Args:
            config (Pano2MeshSRConfig): Configuration object.
        """
        self.cfg = config
        self.device = config.device

        # Init models.
        self.inpainter = PanoPersFusionInpainter(save_path=None)
        self.geo_predictor = PanoJointPredictor(save_path=None)
        self.pano_fusion_distance_predictor = PanoFusionDistancePredictor()
        self.super_model = ImageRealESRGAN(outscale=self.cfg.upscale_factor)

        # Init poses.
        cubemap_w2cs = get_cubemap_views_world_to_cam()
        self.cubemap_w2cs = [p.to(self.device) for p in cubemap_w2cs]
        self.camera_poses = self.load_camera_poses(self.cfg.trajectory_dir)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self.cfg.kernel_size
        )
        self.kernel = torch.from_numpy(kernel).float().to(self.device)

    def init_mesh_params(self) -> None:
        """Initializes mesh parameters and inpaint mask."""
        torch.set_default_device(self.device)
        self.inpaint_mask = torch.ones(
            (self.cfg.cubemap_h, self.cfg.cubemap_w), dtype=torch.bool
        )
        self.vertices = torch.empty((3, 0), requires_grad=False)
        self.colors = torch.empty((3, 0), requires_grad=False)
        self.faces = torch.empty((3, 0), dtype=torch.long, requires_grad=False)

    @staticmethod
    def read_camera_pose_file(filepath: str) -> np.ndarray:
        """Reads a camera pose file and returns the pose matrix.

        Args:
            filepath (str): Path to the camera pose file.

        Returns:
            np.ndarray: 4x4 camera pose matrix.
        """
        with open(filepath, "r") as f:
            values = [float(num) for line in f for num in line.split()]

        return np.array(values).reshape(4, 4)

    def load_camera_poses(
        self, trajectory_dir: str
    ) -> tuple[np.ndarray, list[torch.Tensor]]:
        """Loads camera poses from a directory.

        Args:
            trajectory_dir (str): Directory containing camera pose files.

        Returns:
            tuple[np.ndarray, list[torch.Tensor]]: List of relative camera poses.
        """
        pose_filenames = sorted(
            [
                fname
                for fname in os.listdir(trajectory_dir)
                if fname.startswith("camera_pose")
            ]
        )

        pano_pose_world = None
        relative_poses = []
        for idx, filename in enumerate(pose_filenames):
            pose_path = os.path.join(trajectory_dir, filename)
            pose_matrix = self.read_camera_pose_file(pose_path)

            if pano_pose_world is None:
                pano_pose_world = pose_matrix.copy()
                pano_pose_world[0, 3] += self.cfg.pano_center_offset[0]
                pano_pose_world[2, 3] += self.cfg.pano_center_offset[1]

            # Use different reference for the first 6 cubemap views
            reference_pose = pose_matrix if idx < 6 else pano_pose_world
            relative_matrix = pose_matrix @ np.linalg.inv(reference_pose)
            relative_matrix[0:2, :] *= -1  # flip_xy
            relative_matrix = (
                relative_matrix @ rot_z_world_to_cam(180).cpu().numpy()
            )
            relative_matrix[:3, 3] *= self.cfg.pose_scale
            relative_matrix = torch.tensor(
                relative_matrix, dtype=torch.float32
            )
            relative_poses.append(relative_matrix)

        return relative_poses

    def load_inpaint_poses(
        self, poses: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Samples and loads poses for inpainting.

        Args:
            poses (torch.Tensor): Tensor of camera poses.

        Returns:
            dict[int, torch.Tensor]: Dictionary mapping indices to pose tensors.
        """
        inpaint_poses = dict()
        sampled_views = poses[:: self.cfg.inpaint_frame_stride]
        init_pose = torch.eye(4)
        for idx, w2c_tensor in enumerate(sampled_views):
            w2c = w2c_tensor.cpu().numpy().astype(np.float32)
            c2w = np.linalg.inv(w2c)
            pose_tensor = init_pose.clone()
            pose_tensor[:3, 3] = torch.from_numpy(c2w[:3, 3])
            pose_tensor[:3, 3] *= -1
            inpaint_poses[idx] = pose_tensor.to(self.device)

        return inpaint_poses

    def project(self, world_to_cam: torch.Tensor):
        """Projects the mesh to an image using the given camera pose.

        Args:
            world_to_cam (torch.Tensor): World-to-camera transformation matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Projected RGB image, inpaint mask, and depth map.
        """
        (
            project_image,
            project_depth,
            inpaint_mask,
            _,
            z_buf,
            mesh,
        ) = render_mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_features=self.colors,
            H=self.cfg.cubemap_h,
            W=self.cfg.cubemap_w,
            fov_in_degrees=self.cfg.fov,
            RT=world_to_cam,
            blur_radius=self.cfg.blur_radius,
            faces_per_pixel=self.cfg.faces_per_pixel,
        )
        project_image = project_image * ~inpaint_mask

        return project_image[:3, ...], inpaint_mask, project_depth

    def render_pano(self, pose: torch.Tensor):
        """Renders a panorama from the mesh using the given pose.

        Args:
            pose (torch.Tensor): Camera pose.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: RGB panorama, depth map, and mask.
        """
        cubemap_list = []
        for cubemap_pose in self.cubemap_w2cs:
            project_pose = cubemap_pose @ pose
            rgb, inpaint_mask, depth = self.project(project_pose)
            distance_map = depth_to_distance(depth[None, ...])
            mask = inpaint_mask[None, ...]
            cubemap_list.append(torch.cat([rgb, distance_map, mask], dim=0))

        # Set default tensor type for CPU operation in cube2equi
        with torch.device("cpu"):
            pano_rgbd = cube2equi(
                cubemap_list, "list", self.cfg.pano_h, self.cfg.pano_w
            )

        pano_rgb = pano_rgbd[:3, :, :]
        pano_depth = pano_rgbd[3:4, :, :].squeeze(0)
        pano_mask = pano_rgbd[4:, :, :].squeeze(0)

        return pano_rgb, pano_depth, pano_mask

    def rgbd_to_mesh(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        inpaint_mask: torch.Tensor,
        world_to_cam: torch.Tensor = None,
        using_distance_map: bool = True,
    ) -> None:
        """Converts RGB-D images to mesh and updates mesh parameters.

        Args:
            rgb (torch.Tensor): RGB image tensor.
            depth (torch.Tensor): Depth map tensor.
            inpaint_mask (torch.Tensor): Inpaint mask tensor.
            world_to_cam (torch.Tensor, optional): Camera pose.
            using_distance_map (bool, optional): Whether to use distance map.
        """
        if world_to_cam is None:
            world_to_cam = torch.eye(4, dtype=torch.float32).to(self.device)

        if inpaint_mask.sum() == 0:
            return

        vertices, faces, colors = features_to_world_space_mesh(
            colors=rgb.squeeze(0),
            depth=depth,
            fov_in_degrees=self.cfg.fov,
            world_to_cam=world_to_cam,
            mask=inpaint_mask,
            faces=self.faces,
            vertices=self.vertices,
            using_distance_map=using_distance_map,
            edge_threshold=0.05,
        )

        faces += self.vertices.shape[1]
        self.vertices = torch.cat([self.vertices, vertices], dim=1)
        self.colors = torch.cat([self.colors, colors], dim=1)
        self.faces = torch.cat([self.faces, faces], dim=1)

    def get_edge_image_by_depth(
        self, depth: torch.Tensor, dilate_iter: int = 1
    ) -> np.ndarray:
        """Computes edge image from depth map.

        Args:
            depth (torch.Tensor): Depth map tensor.
            dilate_iter (int, optional): Number of dilation iterations.

        Returns:
            np.ndarray: Edge image.
        """
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().detach().numpy()

        gray = (depth / depth.max() * 255).astype(np.uint8)
        edges = cv2.Canny(gray, 60, 150)
        if dilate_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

        return edges

    def mesh_repair_by_greedy_view_selection(
        self, pose_dict: dict[str, torch.Tensor], output_dir: str
    ) -> list:
        """Repairs mesh by selecting views greedily and inpainting missing regions.

        Args:
            pose_dict (dict[str, torch.Tensor]): Dictionary of poses for inpainting.
            output_dir (str): Directory to save visualizations.

        Returns:
            list: List of inpainted panoramas with poses.
        """
        inpainted_panos_w_pose = []
        while len(pose_dict) > 0:
            logger.info(f"Repairing mesh left rounds {len(pose_dict)}")
            sampled_views = []
            for key, pose in pose_dict.items():
                pano_rgb, pano_distance, pano_mask = self.render_pano(pose)
                completeness = torch.sum(1 - pano_mask) / (pano_mask.numel())
                sampled_views.append((key, completeness.item(), pose))

            if len(sampled_views) == 0:
                break

            # Find inpainting with least view completeness.
            sampled_views = sorted(sampled_views, key=lambda x: x[1])
            key, _, pose = sampled_views[len(sampled_views) * 2 // 3]
            pose_dict.pop(key)

            pano_rgb, pano_distance, pano_mask = self.render_pano(pose)

            colors = pano_rgb.permute(1, 2, 0).clone()
            distances = pano_distance.unsqueeze(-1).clone()
            pano_inpaint_mask = pano_mask.clone()
            init_pose = pose.clone()
            normals = None
            if pano_inpaint_mask.min().item() < 0.5:
                colors, distances, normals = self.inpaint_panorama(
                    idx=key,
                    colors=colors,
                    distances=distances,
                    pano_mask=pano_inpaint_mask,
                )

                init_pose[0, 3], init_pose[1, 3], init_pose[2, 3] = (
                    -pose[0, 3],
                    pose[2, 3],
                    0,
                )
                rays = gen_pano_rays(
                    init_pose, self.cfg.pano_h, self.cfg.pano_w
                )
                conflict_mask = self.sup_pool.geo_check(
                    rays, distances.unsqueeze(-1)
                )  # 0 is conflict, 1 not conflict
                pano_inpaint_mask *= conflict_mask

            self.rgbd_to_mesh(
                colors.permute(2, 0, 1),
                distances,
                pano_inpaint_mask,
                world_to_cam=pose,
            )

            self.sup_pool.register_sup_info(
                pose=init_pose,
                mask=pano_inpaint_mask.clone(),
                rgb=colors,
                distance=distances.unsqueeze(-1),
                normal=normals,
            )

            colors = colors.permute(2, 0, 1).unsqueeze(0)
            inpainted_panos_w_pose.append([colors, pose])

            if self.cfg.visualize:
                from embodied_gen.data.utils import DiffrastRender

                tensor_to_pil(pano_rgb.unsqueeze(0)).save(
                    f"{output_dir}/rendered_pano_{key}.jpg"
                )
                tensor_to_pil(colors).save(
                    f"{output_dir}/inpainted_pano_{key}.jpg"
                )
                norm_depth = DiffrastRender.normalize_map_by_mask(
                    distances, torch.ones_like(distances)
                )
                heatmap = (norm_depth.cpu().numpy() * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                Image.fromarray(heatmap).save(
                    f"{output_dir}/inpainted_depth_{key}.png"
                )

        return inpainted_panos_w_pose

    def inpaint_panorama(
        self,
        idx: int,
        colors: torch.Tensor,
        distances: torch.Tensor,
        pano_mask: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Inpaints missing regions in a panorama.

        Args:
            idx (int): Index of the panorama.
            colors (torch.Tensor): RGB image tensor.
            distances (torch.Tensor): Distance map tensor.
            pano_mask (torch.Tensor): Mask tensor.

        Returns:
            tuple[torch.Tensor]: Inpainted RGB image, distances, and normals.
        """
        mask = (pano_mask[None, ..., None] > 0.5).float()
        mask = mask.permute(0, 3, 1, 2)
        mask = dilation(mask, kernel=self.kernel)
        mask = mask[0, 0, ..., None]  # hwc
        inpainted_img = self.inpainter.inpaint(idx, colors, mask)
        inpainted_img = colors * (1 - mask) + inpainted_img * mask
        inpainted_distances, inpainted_normals = self.geo_predictor(
            idx,
            inpainted_img,
            distances[..., None],
            mask=mask,
            reg_loss_weight=0.0,
            normal_loss_weight=5e-2,
            normal_tv_loss_weight=5e-2,
        )

        return inpainted_img, inpainted_distances.squeeze(), inpainted_normals

    def preprocess_pano(
        self, image: Image.Image | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocesses a panoramic image for mesh generation.

        Args:
            image (Image.Image | str): Input image or path.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Preprocessed RGB and depth tensors.
        """
        if isinstance(image, str):
            image = Image.open(image)

        image = image.convert("RGB")

        if image.size[0] < image.size[1]:
            image = image.transpose(Image.TRANSPOSE)

        image = resize_image_with_aspect_ratio(image, self.cfg.pano_w)
        image_rgb = torch.tensor(np.array(image)).permute(2, 0, 1) / 255
        image_rgb = image_rgb.to(self.device)
        image_depth = self.pano_fusion_distance_predictor.predict(
            image_rgb.permute(1, 2, 0)
        )
        image_depth = (
            image_depth / image_depth.max() * self.cfg.depth_scale_factor
        )

        return image_rgb, image_depth

    def pano_to_perpective(
        self, pano_image: torch.Tensor, pitch: float, yaw: float, fov: float
    ) -> torch.Tensor:
        """Converts a panoramic image to a perspective view.

        Args:
            pano_image (torch.Tensor): Panoramic image tensor.
            pitch (float): Pitch angle.
            yaw (float): Yaw angle.
            fov (float): Field of view.

        Returns:
            torch.Tensor: Perspective image tensor.
        """
        rots = dict(
            roll=0,
            pitch=pitch,
            yaw=yaw,
        )
        perspective = equi2pers(
            equi=pano_image.squeeze(0),
            rots=rots,
            height=self.cfg.cubemap_h,
            width=self.cfg.cubemap_w,
            fov_x=fov,
            mode="bilinear",
        ).unsqueeze(0)

        return perspective

    def pano_to_cubemap(self, pano_rgb: torch.Tensor):
        """Converts a panoramic RGB image to six cubemap views.

        Args:
            pano_rgb (torch.Tensor): Panoramic RGB image tensor.

        Returns:
            list: List of cubemap RGB tensors.
        """
        # Define six canonical cube directions in (pitch, yaw)
        directions = [
            (0, 0),
            (0, 1.5 * np.pi),
            (0, 1.0 * np.pi),
            (0, 0.5 * np.pi),
            (-0.5 * np.pi, 0),
            (0.5 * np.pi, 0),
        ]

        cubemaps_rgb = []
        for pitch, yaw in directions:
            rgb_view = self.pano_to_perpective(
                pano_rgb, pitch, yaw, fov=self.cfg.fov
            )
            cubemaps_rgb.append(rgb_view.cpu())

        return cubemaps_rgb

    def save_mesh(self, output_path: str) -> None:
        """Saves the mesh to a file.

        Args:
            output_path (str): Path to save the mesh file.
        """
        vertices_np = self.vertices.T.cpu().numpy()
        colors_np = self.colors.T.cpu().numpy()
        faces_np = self.faces.T.cpu().numpy()
        mesh = trimesh.Trimesh(
            vertices=vertices_np, faces=faces_np, vertex_colors=colors_np
        )

        mesh.export(output_path)

    def mesh_pose_to_gs_pose(self, mesh_pose: torch.Tensor) -> np.ndarray:
        """Converts mesh pose to 3D Gaussian Splatting pose.

        Args:
            mesh_pose (torch.Tensor): Mesh pose tensor.

        Returns:
            np.ndarray: Converted pose matrix.
        """
        pose = mesh_pose.clone()
        pose[0, :] *= -1
        pose[1, :] *= -1

        Rw2c = pose[:3, :3].cpu().numpy()
        Tw2c = pose[:3, 3:].cpu().numpy()
        yz_reverse = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        Rc2w = (yz_reverse @ Rw2c).T
        Tc2w = -(Rc2w @ yz_reverse @ Tw2c)
        c2w = np.concatenate((Rc2w, Tc2w), axis=1)
        c2w = np.concatenate((c2w, np.array([[0, 0, 0, 1]])), axis=0)

        return c2w

    def __call__(self, pano_image: Image.Image | str, output_dir: str):
        """Runs the pipeline to generate mesh and 3DGS data from a panoramic image.

        Args:
            pano_image (Image.Image | str): Input panoramic image or path.
            output_dir (str): Directory to save outputs.

        Returns:
            None
        """
        self.init_mesh_params()
        pano_rgb, pano_depth = self.preprocess_pano(pano_image)
        self.sup_pool = SupInfoPool()
        self.sup_pool.register_sup_info(
            pose=torch.eye(4).to(self.device),
            mask=torch.ones([self.cfg.pano_h, self.cfg.pano_w]),
            rgb=pano_rgb.permute(1, 2, 0),
            distance=pano_depth[..., None],
        )
        self.sup_pool.gen_occ_grid(res=256)

        logger.info("Init mesh from pano RGBD image...")
        depth_edge = self.get_edge_image_by_depth(pano_depth)
        inpaint_edge_mask = (
            ~torch.from_numpy(depth_edge).to(self.device).bool()
        )
        self.rgbd_to_mesh(pano_rgb, pano_depth, inpaint_edge_mask)

        repair_poses = self.load_inpaint_poses(self.camera_poses)
        inpainted_panos_w_poses = self.mesh_repair_by_greedy_view_selection(
            repair_poses, output_dir
        )
        torch.cuda.empty_cache()
        torch.set_default_device("cpu")

        if self.cfg.mesh_file is not None:
            mesh_path = os.path.join(output_dir, self.cfg.mesh_file)
            self.save_mesh(mesh_path)

        if self.cfg.gs_data_file is None:
            return

        logger.info(f"Dump data for 3DGS training...")
        points_rgb = (self.colors.clip(0, 1) * 255).to(torch.uint8)
        data = {
            "points": self.vertices.permute(1, 0).cpu().numpy(),  # (N, 3)
            "points_rgb": points_rgb.permute(1, 0).cpu().numpy(),  # (N, 3)
            "train": [],
            "eval": [],
        }
        image_h = self.cfg.cubemap_h * self.cfg.upscale_factor
        image_w = self.cfg.cubemap_w * self.cfg.upscale_factor
        Ks = compute_pinhole_intrinsics(image_w, image_h, self.cfg.fov)
        for idx, (pano_img, pano_pose) in enumerate(inpainted_panos_w_poses):
            cubemaps = self.pano_to_cubemap(pano_img)
            for i in range(len(cubemaps)):
                cubemap = tensor_to_pil(cubemaps[i])
                cubemap = self.super_model(cubemap)
                mesh_pose = self.cubemap_w2cs[i] @ pano_pose
                c2w = self.mesh_pose_to_gs_pose(mesh_pose)
                data["train"].append(
                    {
                        "camtoworld": c2w.astype(np.float32),
                        "K": Ks.astype(np.float32),
                        "image": np.array(cubemap),
                        "image_h": image_h,
                        "image_w": image_w,
                        "image_id": len(cubemaps) * idx + i,
                    }
                )

        # Camera poses for evaluation.
        for idx in range(len(self.camera_poses)):
            c2w = self.mesh_pose_to_gs_pose(self.camera_poses[idx])
            data["eval"].append(
                {
                    "camtoworld": c2w.astype(np.float32),
                    "K": Ks.astype(np.float32),
                    "image_h": image_h,
                    "image_w": image_w,
                    "image_id": idx,
                }
            )

        data_path = os.path.join(output_dir, self.cfg.gs_data_file)
        torch.save(data, data_path)

        return


if __name__ == "__main__":
    output_dir = "outputs/bg_v2/test3"
    input_pano = "apps/assets/example_scene/result_pano.png"
    config = Pano2MeshSRConfig()
    pipeline = Pano2MeshSRPipeline(config)
    pipeline(input_pano, output_dir)
