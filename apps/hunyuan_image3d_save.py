import hashlib
import hmac
import json
import time
import sys
import io
from datetime import datetime
from http.client import HTTPSConnection
from pathlib import Path
import urllib.request
import zipfile
import requests,os

def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def download_and_extract_obj(
    url: str,
    save_dir: str,
    ext: str,
    zip_name: str = "model.zip"
) -> str:
    """
    下载 zip → 解压 → 返回 mesh 文件路径
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ 下载
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    content = resp.content

    normalized_ext = ext.lower().lstrip(".")
    obj_like = normalized_ext == "obj"

    # 2️⃣ 如果是 zip，解压并返回目标模型文件
    if zipfile.is_zipfile(io.BytesIO(content)):
        zip_path = save_dir / zip_name
        with zip_path.open("wb") as f:
            f.write(content)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(save_dir)

        pattern_ext = ".obj" if obj_like else f".{normalized_ext}"
        mesh_files = list(save_dir.rglob(f"*{pattern_ext}"))
        if not mesh_files:
            raise FileNotFoundError(f"未找到 {pattern_ext} 文件")
        return str(mesh_files[0].resolve())

    # 3️⃣ 非 zip：按单文件落盘
    out_ext = "obj" if obj_like else normalized_ext
    model_path = save_dir / f"result.{out_ext}"
    with model_path.open("wb") as f:
        f.write(content)
    return str(model_path.resolve())


def query_and_download_hunyuan_job(job_id: str, save_dir: str, secret_id: str, secret_key: str):
    token = ""
    service = "ai3d"
    host = "ai3d.tencentcloudapi.com"
    region = "ap-guangzhou"
    version = "2025-05-13"
    action = "QueryHunyuanTo3DRapidJob"

    payload = json.dumps({"JobId": job_id}, ensure_ascii=False)

    algorithm = "TC3-HMAC-SHA256"
    timestamp = int(time.time())
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

    # ===== Step 1: Canonical request =====
    canonical_headers = (
        "content-type:application/json; charset=utf-8\n"
        f"host:{host}\n"
        f"x-tc-action:{action.lower()}\n"
    )
    signed_headers = "content-type;host;x-tc-action"
    hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    canonical_request = (
        "POST\n/\n\n"
        + canonical_headers
        + "\n"
        + signed_headers
        + "\n"
        + hashed_payload
    )

    # ===== Step 2: String to sign =====
    credential_scope = f"{date}/{service}/tc3_request"
    hashed_canonical_request = hashlib.sha256(
        canonical_request.encode("utf-8")
    ).hexdigest()

    string_to_sign = (
        f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashed_canonical_request}"
    )

    # ===== Step 3: Signature =====
    secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = sign(secret_date, service)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(
        secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    authorization = (
        f"{algorithm} "
        f"Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json; charset=utf-8",
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": version,
        "X-TC-Region": region,
    }

    # ===== Request =====
    MAX_WAIT_SECONDS = 600    # 最多等 10 分钟
    POLL_INTERVAL = 20        # 每 20 秒查一次
    start_time = time.time()
    while True:
        req = HTTPSConnection(host)
        req.request("POST", "/", body=payload.encode("utf-8"), headers=headers)
        resp = req.getresponse()

        body_str = resp.read().decode("utf-8")
        result = json.loads(body_str)

        response = result.get("Response", {})
        status = response.get("Status")

        print(f"[AI3D] Job status = {status}")

        if status == "DONE":
            break

        if time.time() - start_time > MAX_WAIT_SECONDS:
            raise TimeoutError("Job polling timeout")

        time.sleep(POLL_INTERVAL)

    files = response.get("ResultFile3Ds", [])
    model_paths = []
    preview_paths = []
    
    for i, item in enumerate(files):

        model_url = item.get("Url")
        preview_url = item.get("PreviewImageUrl")
        file_type = item.get("Type", "UNKNOWN").lower()  # obj / glb / fbx ...

        # ---------- 1. 下载 ZIP ----------
        
        if model_url:
            obj_path = download_and_extract_obj(model_url, save_dir, file_type)
            model_paths.append(obj_path)
            
        save_dir = Path(save_dir)
        
        # ---------- 4. 下载预览图 ----------
        if preview_url:
            preview_path = save_dir / f"preview_{i}.png"
            urllib.request.urlretrieve(preview_url, preview_path)
            preview_paths.append(str(preview_path))

    return model_paths[0], preview_paths[0]
