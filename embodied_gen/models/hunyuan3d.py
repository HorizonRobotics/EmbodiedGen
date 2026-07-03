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

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from glob import glob
from http.client import HTTPSConnection
from shutil import copy, copytree, rmtree
from typing import Optional, Tuple

import numpy as np
import trimesh
from PIL import Image
from embodied_gen.data.differentiable_render import (
    entrypoint as render_pbr_video,
)
from embodied_gen.data.utils import delete_dir
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import combine_images_to_grid
from embodied_gen.utils.tags import VERSION
from embodied_gen.validators.quality_checkers import (
    BaseChecker,
    ImageSegChecker,
)
from embodied_gen.validators.urdf_convertor import URDFGenerator


@dataclass(frozen=True)
class HunyuanConfig:
    """Tencent Hunyuan3D Pro endpoint + timing.

    Defaults match the validated probe in ``outputs/hunyuan3d_api_expert/``.
    Only the Pro action set is supported.
    """

    host: str = "ai3d.tencentcloudapi.com"
    service: str = "ai3d"
    region: str = "ap-guangzhou"
    version: str = "2025-05-13"
    image_action: str = "SubmitHunyuanTo3DProJob"
    query_action: str = "QueryHunyuanTo3DProJob"
    result_format: str = "GLB"
    texture_size: int = 2048
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    poll_interval: float = 10.0
    max_wait_seconds: float = 900.0
    max_download_bytes: int = 512 * 1024 * 1024


def load_credentials() -> Tuple[str, str]:
    """Read Tencent Cloud SecretId/SecretKey from environment.

    Prefers ``TENCENT_SECRET_ID/KEY``; falls back to ``TENCENTCLOUD_*``.
    Raises ``RuntimeError`` (credential-free message) when missing.
    """
    sid = os.environ.get("TENCENT_SECRET_ID") or os.environ.get(
        "TENCENTCLOUD_SECRET_ID"
    )
    skey = os.environ.get("TENCENT_SECRET_KEY") or os.environ.get(
        "TENCENTCLOUD_SECRET_KEY"
    )
    if not sid or not skey:
        raise RuntimeError(
            "HUNYUAN3D backend requires Tencent Cloud credentials. Set "
            "TENCENT_SECRET_ID and TENCENT_SECRET_KEY (or TENCENTCLOUD_*) "
            "in the environment, e.g. `source .secrets/hunyuan3d.env`."
        )
    return sid, skey


def _signed_headers(
    payload: str,
    action: str,
    credentials: Tuple[str, str],
    cfg: HunyuanConfig,
) -> dict:
    """Build fresh TC3-HMAC-SHA256 auth headers (re-built every request)."""
    sid, skey = credentials
    ts = int(time.time())
    date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    canon_headers = (
        "content-type:application/json; charset=utf-8\n"
        f"host:{cfg.host}\n"
        f"x-tc-action:{action.lower()}\n"
    )
    signed = "content-type;host;x-tc-action"
    canon_req = (
        f"POST\n/\n\n{canon_headers}\n{signed}\n"
        f"{hashlib.sha256(payload.encode()).hexdigest()}"
    )
    scope = f"{date}/{cfg.service}/tc3_request"
    string_to_sign = (
        f"TC3-HMAC-SHA256\n{ts}\n{scope}\n"
        f"{hashlib.sha256(canon_req.encode()).hexdigest()}"
    )

    def _sign(key, msg):
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    sd = _sign(("TC3" + skey).encode(), date)
    ss = _sign(sd, cfg.service)
    signing = _sign(ss, "tc3_request")
    sig = hmac.new(
        signing, string_to_sign.encode(), hashlib.sha256
    ).hexdigest()
    return {
        "Authorization": (
            f"TC3-HMAC-SHA256 Credential={sid}/{scope}, "
            f"SignedHeaders={signed}, Signature={sig}"
        ),
        "Content-Type": "application/json; charset=utf-8",
        "Host": cfg.host,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(ts),
        "X-TC-Version": cfg.version,
        "X-TC-Region": cfg.region,
    }


def _post_signed(
    payload_obj: dict,
    action: str,
    credentials: Tuple[str, str],
    cfg: HunyuanConfig,
) -> dict:
    """POST a Tencent Cloud TC3-signed JSON request and return ``Response``.

    Routes through ``HTTPS_PROXY`` via CONNECT (``HTTPSConnection`` does not
    honor the env var on its own). Never logs credentials, signed headers,
    or the request payload (which carries base64 image data).
    """
    payload = json.dumps(
        payload_obj, separators=(",", ":"), ensure_ascii=False
    )
    headers = _signed_headers(payload, action, credentials, cfg)
    # ``http.client.HTTPSConnection`` does NOT auto-honor ``HTTPS_PROXY``
    # (unlike ``urllib.request.urlopen``); read it explicitly and tunnel
    # via CONNECT, otherwise direct connections to Tencent Cloud will be
    # blocked by the corporate egress firewall.
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    timeout = cfg.connect_timeout + cfg.read_timeout
    if proxy:
        from urllib.parse import urlparse

        p = urlparse(proxy)
        conn = HTTPSConnection(p.hostname, p.port or 80, timeout=timeout)
        conn.set_tunnel(cfg.host, 443)
    else:
        conn = HTTPSConnection(cfg.host, timeout=timeout)
    try:
        conn.request("POST", "/", body=payload.encode(), headers=headers)
        resp = conn.getresponse()
        status, body = resp.status, resp.read().decode(errors="replace")
    finally:
        conn.close()

    if not 200 <= status < 300:
        raise RuntimeError(
            f"Hunyuan3D {action} HTTP {status}; len={len(body)}."
        )
    try:
        data = json.loads(body).get("Response", {})
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Hunyuan3D {action} non-JSON (HTTP {status}): {exc}"
        )
    err = data.get("Error")
    if err:
        raise RuntimeError(
            f"Hunyuan3D {action} Tencent error: "
            f"Code={err.get('Code')} Message={err.get('Message')} "
            f"RequestId={data.get('RequestId')}"
        )
    return data


def submit_pro_job(
    image_path: Optional[str] = None,
    credentials: Tuple[str, str] = None,
    cfg: HunyuanConfig = None,
    prompt: Optional[str] = None,
) -> str:
    """Submit a Hunyuan3D Pro job, return its ``JobId``.

    Provide exactly one of ``image_path`` (image-to-3D, body field
    ``ImageBase64``) or ``prompt`` (text-to-3D, body field ``Prompt``).
    Tencent's ``SubmitHunyuanTo3DProJob`` action is shared between both
    modes; only the body discriminator differs.
    """
    if (image_path is None) == (prompt is None):
        raise ValueError(
            "submit_pro_job requires exactly one of image_path or prompt."
        )
    if credentials is None or cfg is None:
        raise ValueError("credentials and cfg are required.")

    payload = {"ResultFormat": cfg.result_format, "EnablePBR": True}
    if image_path is not None:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Hunyuan3D input image missing: {image_path}"
            )
        with open(image_path, "rb") as fh:
            payload["ImageBase64"] = base64.b64encode(fh.read()).decode()
        mode = "image"
    else:
        payload["Prompt"] = prompt
        mode = "text"

    resp = _post_signed(payload, cfg.image_action, credentials, cfg)
    job_id = resp.get("JobId")
    if not job_id:
        raise RuntimeError(
            f"Hunyuan3D submit returned no JobId; "
            f"RequestId={resp.get('RequestId')}."
        )
    logger.info(
        "HUNYUAN3D submit OK (%s): JobId=%s RequestId=%s",
        mode,
        job_id,
        resp.get("RequestId"),
    )
    return job_id


def wait_for_pro_job(
    job_id: str,
    credentials: Tuple[str, str],
    cfg: HunyuanConfig,
) -> dict:
    """Poll the job until DONE; raise on FAIL/unknown/timeout."""
    deadline = time.time() + cfg.max_wait_seconds
    last_status = None
    while True:
        resp = _post_signed(
            {"JobId": job_id}, cfg.query_action, credentials, cfg
        )
        status = resp.get("Status")
        if status != last_status:
            logger.info(
                "HUNYUAN3D job %s status=%s RequestId=%s",
                job_id,
                status,
                resp.get("RequestId"),
            )
            if last_status is None and status in ("WAIT", "RUN"):
                logger.info(
                    "HUNYUAN3D Pro inference typically takes ~3 minutes; "
                    "polling every %ss.",
                    int(cfg.poll_interval),
                )
            last_status = status
        if status == "DONE":
            return resp
        if status == "FAIL":
            raise RuntimeError(
                f"Hunyuan3D job {job_id} FAIL: "
                f"code={resp.get('ErrorCode')} "
                f"message={resp.get('ErrorMessage')} "
                f"RequestId={resp.get('RequestId')}."
            )
        if status not in ("WAIT", "RUN"):
            raise RuntimeError(
                f"Hunyuan3D job {job_id} unknown status={status!r}; "
                f"RequestId={resp.get('RequestId')}."
            )
        if time.time() >= deadline:
            raise TimeoutError(
                f"Hunyuan3D job {job_id} did not finish within "
                f"{cfg.max_wait_seconds}s (last status={status})."
            )
        time.sleep(cfg.poll_interval)


def _download_url_to_path(url: str, dst: str, cfg: HunyuanConfig) -> int:
    """Stream ``url`` to ``dst`` with size/timeout caps. Returns bytes written.

    Logs only the host (signed URL paths carry short-lived auth tokens).
    """
    from urllib.parse import urlparse

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    logger.info(
        "HUNYUAN3D downloading %s from host=%s",
        os.path.basename(dst),
        urlparse(url).hostname or "?",
    )
    total = 0
    timeout = cfg.connect_timeout + cfg.read_timeout
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if not 200 <= resp.status < 300:
                raise RuntimeError(
                    f"Hunyuan3D download HTTP {resp.status} for "
                    f"{os.path.basename(dst)}."
                )
            with open(dst, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > cfg.max_download_bytes:
                        raise RuntimeError(
                            f"Hunyuan3D download exceeded "
                            f"{cfg.max_download_bytes} bytes for "
                            f"{os.path.basename(dst)}."
                        )
                    out.write(chunk)
    except (urllib.error.URLError, socket.timeout) as exc:
        raise RuntimeError(
            f"Hunyuan3D download failed for {os.path.basename(dst)}: {exc}"
        )
    return total


def acquire_pro_glb(
    image_path: Optional[str] = None,
    output_dir: str = None,
    asset_name: str = None,
    credentials: Tuple[str, str] = None,
    cfg: HunyuanConfig = None,
    prompt: Optional[str] = None,
) -> str:
    """End-to-end: submit + poll + download GLB into the output dir.

    Provide exactly one of ``image_path`` or ``prompt`` (see
    :func:`submit_pro_job` for the body-field difference).
    """
    if (
        output_dir is None
        or asset_name is None
        or credentials is None
        or cfg is None
    ):
        raise ValueError(
            "output_dir, asset_name, credentials and cfg are required."
        )
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"{asset_name}.glb")

    job_id = submit_pro_job(
        image_path=image_path,
        prompt=prompt,
        credentials=credentials,
        cfg=cfg,
    )
    resp = wait_for_pro_job(job_id, credentials, cfg)

    files = resp.get("ResultFile3Ds") or []
    glb_url = next(
        (
            f.get("Url")
            for f in files
            if (f.get("Type") or "").upper() == "GLB" and f.get("Url")
        ),
        None,
    )
    if not glb_url:
        raise RuntimeError(
            f"Hunyuan3D job {job_id} returned no GLB; "
            f"RequestId={resp.get('RequestId')}."
        )
    _download_url_to_path(glb_url, glb_path, cfg)
    return glb_path


def _texture_array(tex) -> Optional[np.ndarray]:
    """Return RGB ndarray for a glTF texture, or None if absent/invalid."""
    if tex is None or not hasattr(tex, "convert"):
        return None
    return np.asarray(tex.convert("RGB"))


def _save_rgb(arr: Optional[np.ndarray], dst: str, max_edge: int) -> bool:
    """Save an RGB texture as PNG, capping the longest edge at ``max_edge``."""
    if arr is None:
        return False
    img = Image.fromarray(arr)
    longest = max(img.size)
    if longest > max_edge:
        scale = max_edge / float(longest)
        img = img.resize(
            (
                max(1, int(img.size[0] * scale)),
                max(1, int(img.size[1] * scale)),
            ),
            Image.LANCZOS,
        )
    img.save(dst)
    return True


def _bake_scene_transform(
    scene: trimesh.Scene,
) -> Tuple[trimesh.Trimesh, np.ndarray, str]:
    """Apply scene-graph transforms to mesh vertices; return one Trimesh."""
    if len(scene.graph.nodes_geometry) != 1:
        parts = []
        for n in scene.graph.nodes_geometry:
            xform, gname = scene.graph[n]
            m = scene.geometry[gname].copy()
            m.apply_transform(xform)
            parts.append(m)
        return trimesh.util.concatenate(parts), np.eye(4), "concatenated"
    n = next(iter(scene.graph.nodes_geometry))
    xform, gname = scene.graph[n]
    mesh = scene.geometry[gname].copy()
    mesh.apply_transform(xform)
    return mesh, xform, gname


def export_glb_to_obj(
    glb_path: str,
    output_dir: str,
    asset_name: str,
    texture_size: int = 2048,
    pre_align_rotation: Optional[np.ndarray] = None,
) -> str:
    """Convert a Hunyuan3D Pro GLB into the full-PBR OBJ + MTL + PBR PNGs.

    Bakes the GLB scene transform, optionally applies ``pre_align_rotation``
    to the vertex array (used by the text-to-3D path, whose endpoint emits
    a frame rotated 90° around the up axis relative to the image-to-3D
    endpoint), recenters to the bbox origin (matching SAM3D's convention
    of putting the model origin at the geometric center), and writes a
    Blender-compatible OBJ/MTL referencing 4 PBR PNGs (baseColor /
    metallic / roughness / normal) plus a ``_pbr_material.json`` metadata
    sidecar. The source GLB at ``glb_path`` is overwritten with the
    aligned mesh so downstream steps can reuse it. Returns the OBJ path.
    """
    from trimesh.exchange.obj import export_obj

    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, f"{asset_name}.obj")
    mtl_path = os.path.join(output_dir, f"{asset_name}.mtl")
    json_path = os.path.join(output_dir, f"{asset_name}_pbr_material.json")

    scene = trimesh.load(glb_path, force="scene", process=False)
    mesh, baked_xform, geom_name = _bake_scene_transform(scene)
    material = getattr(getattr(mesh, "visual", None), "material", None)

    # Align to SAM3D convention: optional pre-rotation (text-to-3D needs a
    # -90° around the up axis to share the image-to-3D frame) + recenter to
    # the bbox origin. Overwrite the source GLB so downstream steps can
    # reuse the aligned full-PBR mesh without an extra load/export pass.
    V = np.asarray(mesh.vertices, dtype=np.float32)
    if pre_align_rotation is not None:
        V = V @ np.asarray(pre_align_rotation, dtype=np.float32)
    bbox_center = (V.min(axis=0) + V.max(axis=0)) * 0.5
    mesh.vertices = V - bbox_center
    mesh.export(glb_path)

    raw_name = getattr(material, "name", None) or f"{asset_name}_material"
    material_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._")
    if not material_name:
        material_name = f"{asset_name}_material"

    # Write OBJ (rewrite usemtl so it points at our material, after mtllib).
    obj_text = export_obj(
        mesh,
        include_normals=True,
        include_color=True,
        include_texture=True,
        return_texture=False,
        write_texture=False,
        mtl_name=os.path.basename(mtl_path),
        header=(
            "Exported from Hunyuan3D Pro GLB; "
            "scene transform baked, recentered to bbox origin"
        ),
    )
    obj_text = re.sub(
        r"^usemtl\s+.+$",
        f"usemtl {material_name}",
        obj_text,
        flags=re.MULTILINE,
    )
    mtllib_line = f"mtllib {os.path.basename(mtl_path)}\n"
    if f"usemtl {material_name}" not in obj_text:
        obj_text = obj_text.replace(
            mtllib_line, f"{mtllib_line}usemtl {material_name}\n", 1
        )
    with open(obj_path, "w", encoding="utf-8") as fh:
        fh.write(obj_text)

    # PBR textures. metallicRoughnessTexture: G=roughness, B=metallic.
    base_arr = _texture_array(getattr(material, "baseColorTexture", None))
    mr_arr = _texture_array(
        getattr(material, "metallicRoughnessTexture", None)
    )
    normal_arr = _texture_array(getattr(material, "normalTexture", None))
    files = {
        "baseColor": f"{asset_name}_baseColor.png",
        "metallic": f"{asset_name}_metallic.png",
        "roughness": f"{asset_name}_roughness.png",
        "normal": f"{asset_name}_normal.png",
    }
    metallic_arr = (
        np.stack([mr_arr[:, :, 2]] * 3, axis=-1)
        if mr_arr is not None
        else None
    )
    roughness_arr = (
        np.stack([mr_arr[:, :, 1]] * 3, axis=-1)
        if mr_arr is not None
        else None
    )
    saved = {
        "baseColor": _save_rgb(
            base_arr,
            os.path.join(output_dir, files["baseColor"]),
            texture_size,
        ),
        "metallic": _save_rgb(
            metallic_arr,
            os.path.join(output_dir, files["metallic"]),
            texture_size,
        ),
        "roughness": _save_rgb(
            roughness_arr,
            os.path.join(output_dir, files["roughness"]),
            texture_size,
        ),
        "normal": _save_rgb(
            normal_arr,
            os.path.join(output_dir, files["normal"]),
            texture_size,
        ),
    }

    def _factor(attr: str, default: float = 1.0) -> float:
        v = getattr(material, attr, default)
        return default if v is None else float(v)

    bc = getattr(material, "baseColorFactor", None)
    if bc is None:
        base_factor = [1.0, 1.0, 1.0, 1.0]
    else:
        arr = np.asarray(bc, dtype=float).reshape(-1)
        if arr.max(initial=1.0) > 1.0:
            arr = arr / 255.0
        base_factor = [float(arr[0]), float(arr[1]), float(arr[2])]
        base_factor.append(float(arr[3]) if len(arr) >= 4 else 1.0)
    metallic_factor = _factor("metallicFactor", 1.0)
    roughness_factor = _factor("roughnessFactor", 1.0)

    ns = max(1.0, min(1000.0, (1.0 - roughness_factor) * 1000.0))
    lines = [
        "# Exported from Hunyuan3D Pro GLB",
        "# PBR note: glTF metallicRoughnessTexture stores roughness in G "
        "and metallic in B.",
        f"newmtl {material_name}",
        f"Ka {base_factor[0]:.8g} {base_factor[1]:.8g} {base_factor[2]:.8g}",
        f"Kd {base_factor[0]:.8g} {base_factor[1]:.8g} {base_factor[2]:.8g}",
        "Ks 0 0 0",
        f"Ns {ns:.8g}",
        f"d {base_factor[3]:.8g}",
        "illum 2",
        f"Pm {metallic_factor:.8g}",
        f"Pr {roughness_factor:.8g}",
    ]
    if saved["baseColor"]:
        lines.append(f"map_Kd {files['baseColor']}")
    if saved["normal"]:
        lines.append(f"norm {files['normal']}")
        lines.append(f"bump {files['normal']}")
    if saved["metallic"]:
        lines.append(f"map_Pm {files['metallic']}")
    if saved["roughness"]:
        lines.append(f"map_Pr {files['roughness']}")
    with open(mtl_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    metadata = {
        "source": glb_path,
        "obj": os.path.basename(obj_path),
        "mtl": os.path.basename(mtl_path),
        "material": material_name,
        "geometry": geom_name,
        "alignment": "recenter_to_bbox_origin",
        "bakedTransform": np.asarray(baked_xform).tolist(),
        "sourceSceneBounds": np.asarray(scene.bounds).tolist(),
        "exportedObjBounds": np.asarray(mesh.bounds).tolist(),
        "baseColorFactor": base_factor,
        "metallicFactor": metallic_factor,
        "roughnessFactor": roughness_factor,
        "textureMaxEdge": texture_size,
        "textures": {k: (files[k] if saved[k] else None) for k in files},
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(metadata, indent=2) + "\n")

    return obj_path


def _ship_scaled_pbr_artefacts(
    aligned_glb: str,
    urdf_path: str,
    output_root: str,
    final_mesh_dir: str,
    asset_name: str,
) -> None:
    """Write scaled OBJ companions + GLB with full PBR into ``final_mesh_dir``.

    URDFGen's trimesh roundtrip drops Hunyuan's metallic/roughness/normal
    maps; we restore PBR fidelity by:

      1. Inferring the scale factor URDFGen applied by comparing the
         scaled OBJ's extent with the aligned source GLB's extent (the
         URDF ``<scale>`` element stores ``real_height`` instead, which is
         a different quantity).
      2. Loading the aligned full-PBR source GLB, scaling it, and writing
         it next to URDFGen's OBJ so both share the same scale.
      3. Copying the 4 PBR PNGs from ``output_root`` into the mesh dir.
      4. Patching URDFGen's ``material.mtl`` so Phong rendering has a
         visible specular highlight and PBR-aware OBJ importers pick up
         ``map_Pm`` / ``map_Pr`` / ``norm`` / ``bump`` references.
    """
    # Derive the actual scale factor from URDFGen's scaled OBJ rather than
    # the URDF ``<scale>`` element (which stores real_height — a midpoint
    # value distinct from the geometric scaling factor URDFGen applied).
    urdfgen_obj = trimesh.load(
        os.path.join(final_mesh_dir, f"{asset_name}.obj"),
        force="mesh",
        process=False,
    )
    target_max = float(urdfgen_obj.extents.max())

    scene = trimesh.load(aligned_glb, force="scene", process=False)
    mesh, _, _ = _bake_scene_transform(scene)
    V = np.asarray(mesh.vertices, dtype=np.float32)
    src_max = float((V.max(axis=0) - V.min(axis=0)).max())
    scale = target_max / src_max if src_max > 1e-9 else 1.0
    mesh.vertices = V * scale
    mesh.export(os.path.join(final_mesh_dir, f"{asset_name}.glb"))

    pbr_pngs = {
        "metallic": f"{asset_name}_metallic.png",
        "roughness": f"{asset_name}_roughness.png",
        "normal": f"{asset_name}_normal.png",
    }
    base_color_png = f"{asset_name}_baseColor.png"
    pbr_json = f"{asset_name}_pbr_material.json"
    for fname in (
        base_color_png,
        pbr_pngs["metallic"],
        pbr_pngs["roughness"],
        pbr_pngs["normal"],
        pbr_json,
    ):
        src = os.path.join(output_root, fname)
        if os.path.exists(src):
            copy(src, os.path.join(final_mesh_dir, fname))

    mtl_path = os.path.join(final_mesh_dir, "material.mtl")
    if os.path.exists(mtl_path):
        with open(mtl_path) as fh:
            mtl_text = fh.read()
        # trimesh's OBJ exporter writes ``Ks 0 0 0`` + ``Ns 1`` which makes
        # Blender's OBJ Phong path render the surface as flat matte. Bump
        # specular and shininess so the OBJ has visible highlights matching
        # the PBR GLB, then append the PBR texture map references that
        # PBR-aware OBJ importers (Blender 3.6+, others) will pick up.
        mtl_text = re.sub(
            r"^Ks\s.+$", "Ks 0.5 0.5 0.5", mtl_text, flags=re.MULTILINE
        )
        mtl_text = re.sub(r"^Ns\s.+$", "Ns 250", mtl_text, flags=re.MULTILINE)
        if not re.search(r"^illum\s", mtl_text, re.MULTILINE):
            mtl_text = mtl_text.rstrip() + "\nillum 2\n"
        extras = []
        if os.path.exists(os.path.join(final_mesh_dir, pbr_pngs["metallic"])):
            extras.append(f"map_Pm {pbr_pngs['metallic']}")
        if os.path.exists(os.path.join(final_mesh_dir, pbr_pngs["roughness"])):
            extras.append(f"map_Pr {pbr_pngs['roughness']}")
        if os.path.exists(os.path.join(final_mesh_dir, pbr_pngs["normal"])):
            extras.append(f"norm {pbr_pngs['normal']}")
            extras.append(f"bump {pbr_pngs['normal']}")
        if extras and not any(
            line in mtl_text for line in ("map_Pm", "map_Pr", "norm ")
        ):
            mtl_text = mtl_text.rstrip() + "\n" + "\n".join(extras) + "\n"
        with open(mtl_path, "w") as fh:
            fh.write(mtl_text)


def _build_asset_attrs(args, idx: int) -> dict:
    """Build the URDF asset_attrs dict from CLI args."""
    attrs = {"version": args.version or VERSION}
    if args.height_range:
        lo, hi = map(float, args.height_range.split("-"))
        attrs["min_height"], attrs["max_height"] = lo, hi
    if args.mass_range:
        lo, hi = map(float, args.mass_range.split("-"))
        attrs["min_mass"], attrs["max_mass"] = lo, hi
    if isinstance(args.asset_type, list) and args.asset_type[idx]:
        attrs["category"] = args.asset_type[idx]
    return attrs


def _render_color_video(
    obj_path: str, work_dir: str, filename: str
) -> Optional[str]:
    """Render a turntable color mp4 via the shared kaolin renderer.

    Returns the produced mp4 path, or ``None`` on failure (caller logs).
    """
    try:
        # differentiable_render hardcodes mp4 fps=15; 90 frames -> 6s,
        # matching SAM3D/TRELLIS gs_mesh.mp4 duration.
        render_pbr_video(
            mesh_path=obj_path,
            output_root=work_dir,
            uuid=[filename],
            num_images=90,
            elevation=[20.0],
            distance=5.0,
            fov=30.0,
            with_mtl=True,
            gen_color_mp4=True,
            no_index_file=True,
        )
        mp4 = os.path.join(work_dir, filename, "color.mp4")
        return mp4 if os.path.exists(mp4) else None
    except Exception as exc:  # pragma: no cover - rendering is optional
        logger.warning(f"HUNYUAN3D video render failed: {exc}")
        return None


def _process_glb(
    args,
    idx: int,
    output_root: str,
    filename: str,
    cfg: HunyuanConfig,
    checkers: list,
    log_label: str,
    seg_input_pair: Optional[Tuple[str, str]] = None,
    pre_align_rotation: Optional[np.ndarray] = None,
) -> str:
    """GLB-to-result post-processing shared by image and text paths.

    Expects an aligned full-PBR GLB at ``{output_root}/{filename}.glb``.
    Runs ``export_glb_to_obj`` → video render → URDFGen → PBR fidelity
    fixup → single-arg quality checks (skipped when ``checkers`` is empty)
    → ``result/`` organization. ``seg_input_pair`` lets the image path
    feed raw/cond images to ``ImageSegChecker``; text path passes ``None``.
    ``pre_align_rotation`` (3x3) is folded into the single mesh transform
    inside ``export_glb_to_obj``, avoiding a separate load/export pass.
    Returns the result dir path.
    """
    export_glb_to_obj(
        glb_path=os.path.join(output_root, f"{filename}.glb"),
        output_dir=output_root,
        asset_name=filename,
        texture_size=cfg.texture_size,
        pre_align_rotation=pre_align_rotation,
    )
    mesh_obj_path = os.path.join(output_root, f"{filename}.obj")

    video_path = _render_color_video(
        mesh_obj_path, os.path.join(output_root, "_video"), filename
    )

    urdf_convertor = URDFGenerator(
        GPT_CLIENT,
        render_view_num=4,
        decompose_convex=not args.disable_decompose_convex,
    )
    urdf_root = f"{output_root}/URDF_{filename}"
    urdf_path = urdf_convertor(
        mesh_path=mesh_obj_path,
        output_root=urdf_root,
        **_build_asset_attrs(args, idx),
    )

    # Final mesh dir: keep URDFGen's scaled OBJ + collision, restore full
    # PBR fidelity that URDFGen's simple trimesh roundtrip strips (rescaled
    # source GLB + PBR map refs appended to material.mtl).
    final_mesh_dir = f"{urdf_root}/{urdf_convertor.output_mesh_dir}"
    _ship_scaled_pbr_artefacts(
        aligned_glb=os.path.join(output_root, f"{filename}.glb"),
        urdf_path=urdf_path,
        output_root=output_root,
        final_mesh_dir=final_mesh_dir,
        asset_name=filename,
    )

    # Quality checks: only the single-arg (BaseChecker.validate) ones go
    # here. Two-arg checkers like TextGenAlignChecker run in the caller.
    if checkers:
        render_image_paths = glob(
            f"{urdf_root}/{urdf_convertor.output_render_dir}/image_color/*.png"
        )
        images_list = []
        for ch in checkers:
            if isinstance(ch, ImageSegChecker) and seg_input_pair is not None:
                images_list.append(list(seg_input_pair))
            else:
                images_list.append(combine_images_to_grid(render_image_paths))
        qa_results = BaseChecker.validate(checkers, images_list)
        urdf_convertor.add_quality_tag(urdf_path, qa_results)

    # Organize result/ (no gs.ply; video.mp4 included when render OK).
    result_dir = f"{output_root}/result"
    if os.path.exists(result_dir):
        rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir, exist_ok=True)
    copy(urdf_path, f"{result_dir}/{os.path.basename(urdf_path)}")
    copytree(
        f"{urdf_root}/{urdf_convertor.output_mesh_dir}",
        f"{result_dir}/{urdf_convertor.output_mesh_dir}",
    )
    if video_path and os.path.exists(video_path):
        copy(video_path, f"{result_dir}/video.mp4")

    if not args.keep_intermediate:
        delete_dir(output_root, keep_subs=["result"])

    logger.info(f"Saved results for {log_label} in {result_dir}")
    return result_dir


# Rotation that aligns a Hunyuan3D **text**-to-3D GLB with the **image**-to-3D
# frame. -90° around the file-coord up axis (Y), i.e. x' = z, z' = -x.
# Applied as a single multiplication inside ``export_glb_to_obj`` so the
# text path does not need a separate GLB load/save pass.
TEXT_TO_IMAGE_FRAME_ROTATION = np.array(
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float32,
)


def _acquire_or_reuse_glb(
    output_root: str,
    filename: str,
    cfg: HunyuanConfig,
    hunyuan_credentials: Optional[Tuple[str, str]],
    *,
    image_path: Optional[str] = None,
    prompt: Optional[str] = None,
) -> None:
    """Ensure ``{output_root}/{filename}.glb`` exists.

    Reuses an existing GLB at that path (dev fixture short-circuit) or
    calls :func:`acquire_pro_glb` with ``image_path`` or ``prompt``.
    """
    glb_path = os.path.join(output_root, f"{filename}.glb")
    if os.path.exists(glb_path):
        logger.info(
            "HUNYUAN3D reusing existing GLB at %s; skipping Tencent API call.",
            glb_path,
        )
        return
    creds = hunyuan_credentials or load_credentials()
    acquire_pro_glb(
        image_path=image_path,
        prompt=prompt,
        output_dir=output_root,
        asset_name=filename,
        credentials=creds,
        cfg=cfg,
    )


def process_image(
    args,
    idx: int,
    image_path: str,
    output_root: str,
    filename: str,
    hunyuan_config: Optional[HunyuanConfig],
    hunyuan_credentials: Optional[Tuple[str, str]],
    checkers: list,
) -> None:
    """HUNYUAN3D image-to-3D entry: image → GLB → export → URDF → result/."""
    cfg = hunyuan_config or HunyuanConfig()
    _acquire_or_reuse_glb(
        output_root, filename, cfg, hunyuan_credentials, image_path=image_path
    )
    _process_glb(
        args=args,
        idx=idx,
        output_root=output_root,
        filename=filename,
        cfg=cfg,
        checkers=checkers,
        log_label=image_path,
        seg_input_pair=(
            f"{output_root}/{filename}_raw.png",
            f"{output_root}/{filename}_cond.png",
        ),
    )


def process_prompt(
    args,
    idx: int,
    prompt: str,
    output_root: str,
    filename: str,
    hunyuan_config: Optional[HunyuanConfig],
    hunyuan_credentials: Optional[Tuple[str, str]],
    checkers: list,
) -> None:
    """HUNYUAN3D text-to-3D entry: prompt → GLB → export → URDF → result/.

    Text path skips ``text-to-image`` entirely; ``checkers`` should only
    contain single-arg (``BaseChecker.validate``-compatible) checkers.
    Two-arg checkers like ``TextGenAlignChecker`` should be invoked by
    the caller after this returns.
    """
    cfg = hunyuan_config or HunyuanConfig()
    _acquire_or_reuse_glb(
        output_root, filename, cfg, hunyuan_credentials, prompt=prompt
    )
    # Text endpoint sits 90° offset around the up axis vs the image
    # endpoint; fold the alignment rotation into export_glb_to_obj's
    # single mesh-transform pass to avoid a separate GLB roundtrip.
    _process_glb(
        args=args,
        idx=idx,
        output_root=output_root,
        filename=filename,
        cfg=cfg,
        checkers=checkers,
        log_label=f"prompt={prompt!r}",
        seg_input_pair=None,
        pre_align_rotation=TEXT_TO_IMAGE_FRAME_ROTATION,
    )
