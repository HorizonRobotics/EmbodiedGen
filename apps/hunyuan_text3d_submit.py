import argparse
import hashlib
import hmac
import json
import os
import time
from datetime import datetime
from http.client import HTTPSConnection
from typing import Any, Dict, Optional


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _build_tc3_headers(
    payload: str,
    action: str,
    secret_id: str,
    secret_key: str,
    service: str = "ai3d",
    host: str = "ai3d.tencentcloudapi.com",
    region: str = "ap-guangzhou",
    version: str = "2025-05-13",
    token: str = "",
) -> Dict[str, str]:
    timestamp = int(time.time())
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

    canonical_headers = (
        "content-type:application/json; charset=utf-8\n"
        f"host:{host}\n"
        f"x-tc-action:{action.lower()}\n"
    )
    signed_headers = "content-type;host;x-tc-action"
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    canonical_request = (
        "POST\n/\n\n"
        f"{canonical_headers}\n"
        f"{signed_headers}\n"
        f"{hashed_request_payload}"
    )

    credential_scope = f"{date}/{service}/tc3_request"
    string_to_sign = (
        "TC3-HMAC-SHA256\n"
        f"{timestamp}\n"
        f"{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    )

    secret_date = _sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = _sign(secret_date, service)
    secret_signing = _sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (
        "TC3-HMAC-SHA256 "
        f"Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    headers: Dict[str, str] = {
        "Authorization": authorization,
        "Content-Type": "application/json; charset=utf-8",
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": version,
        "X-TC-Region": region,
    }
    if token:
        headers["X-TC-Token"] = token
    return headers


def _post_ai3d(payload_obj: Dict[str, Any], action: str, secret_id: str, secret_key: str) -> Dict[str, Any]:
    host = "ai3d.tencentcloudapi.com"
    payload = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False)
    headers = _build_tc3_headers(payload, action, secret_id, secret_key, host=host)

    conn = HTTPSConnection(host)
    conn.request("POST", "/", body=payload.encode("utf-8"), headers=headers)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    result = json.loads(body)

    if "Error" in result.get("Response", {}):
        raise RuntimeError(f"Tencent API Error: {result['Response']['Error']}")
    return result


def hunyuan_text3d_submit(
    prompt: str,
    secret_id: Optional[str] = None,
    secret_key: Optional[str] = None,
    result_format: str = "GLB",
    enable_pbr: bool = True,
    action: Optional[str] = None,
) -> str:
    """Submit text-to-3d job and return JobId."""
    secret_id = secret_id or os.getenv("TENCENT_SECRET_ID") or os.getenv("TENCENTCLOUD_SECRET_ID")
    secret_key = secret_key or os.getenv("TENCENT_SECRET_KEY") or os.getenv("TENCENTCLOUD_SECRET_KEY")
    if secret_id:
        secret_id = secret_id.strip().strip("\'\"")
    if secret_key:
        secret_key = secret_key.strip().strip("\'\"")
    if not secret_id or not secret_key:
        raise ValueError("Missing credential: set secret_id/secret_key or env TENCENT_SECRET_ID/TENCENT_SECRET_KEY")
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")

    payload = {
        "Prompt": prompt,
        "ResultFormat": result_format,
        "EnablePBR": enable_pbr,
    }
    submit_action = action or os.getenv("HUNYUAN_TEXT3D_ACTION") or "SubmitHunyuanTo3DProJob"
    result = _post_ai3d(payload, action=submit_action, secret_id=secret_id, secret_key=secret_key)
    return result["Response"]["JobId"]


# Backward compatible alias
submit_hunyuan_text3D = hunyuan_text3d_submit


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Hunyuan Text->3D job")
    parser.add_argument("--prompt", required=True, help="Text prompt for 3D generation")
    parser.add_argument("--secret-id", default=None, help="Tencent Cloud SecretId (or env TENCENT_SECRET_ID)")
    parser.add_argument("--secret-key", default=None, help="Tencent Cloud SecretKey (or env TENCENT_SECRET_KEY)")
    parser.add_argument("--result-format", default="GLB", choices=["GLB", "OBJ", "FBX"], help="Output format")
    parser.add_argument("--disable-pbr", action="store_true", help="Disable PBR material generation")
    args = parser.parse_args()

    job_id = hunyuan_text3d_submit(
        prompt=args.prompt,
        secret_id=args.secret_id,
        secret_key=args.secret_key,
        result_format=args.result_format,
        enable_pbr=not args.disable_pbr,
    )
    print(job_id)


if __name__ == "__main__":
    main()
