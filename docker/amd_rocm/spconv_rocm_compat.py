"""spconv KRSC->Native weight bridge for ROCm.

CUDA spconv 2.3.x defaults to the ImplicitGemm conv algo, whose `SparseConvolution`
weights are stored in KRSC layout: 5D `[out_channels, kd, kh, kw, in_channels]`.
ROCm `spconv_rocm` (2.3.8+rocm1) lacks the implicit-gemm kernels and falls back to
the Native algo, whose weights are 3D `[kernel_volume, in_channels, out_channels]`.

So checkpoints trained on CUDA (e.g. facebook/sam-3d-objects, SAM3D, TRELLIS) fail to
load on ROCm with errors like:
    size mismatch ... copying a param with shape [128, 3, 3, 3, 128]
    the shape in current model is [27, 128, 128]

This patches `torch.nn.Module.load_state_dict` to transparently convert any 5D KRSC
spconv weight into the 3D Native layout when (and only when) the destination model
parameter is the matching 3D shape. It is a no-op on CUDA / already-native weights.

Activation: import before loading any spconv checkpoint (e.g. via sitecustomize).
This is the ROCm-unblock shim; the upstream-appropriate fix belongs in spconv_rocm
(accept KRSC checkpoints under the Native algo).
"""

import torch

_orig_load_state_dict = torch.nn.Module.load_state_dict


def _krsc_to_native(w: torch.Tensor):
    # KRSC [out, kd, kh, kw, in] -> Native [kd*kh*kw, in, out]; the Native algo squeezes
    # kernel_volume==1 (1x1x1 conv) to 2D [in, out].
    out, kd, kh, kw, inc = w.shape
    kvol = kd * kh * kw
    native = w.permute(1, 2, 3, 4, 0).contiguous().reshape(kvol, inc, out)
    return native, kvol, inc, out


def _patched_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
    try:
        own = self.state_dict()
    except Exception:
        return _orig_load_state_dict(
            self, state_dict, strict=strict, *args, **kwargs
        )

    converted = 0
    fixed = dict(state_dict)
    for name, val in state_dict.items():
        tgt = own.get(name)
        if tgt is None or not hasattr(val, "ndim") or val.ndim != 5:
            continue
        native, kvol, inc, out = _krsc_to_native(val)
        if tgt.ndim == 3 and tuple(tgt.shape) == (kvol, inc, out):
            fixed[name] = native
            converted += 1
        elif tgt.ndim == 2 and kvol == 1 and tuple(tgt.shape) == (inc, out):
            fixed[name] = native.reshape(inc, out)
            converted += 1
    if converted:
        print(
            f"[spconv-rocm-compat] converted {converted} KRSC->Native conv weights"
        )
    return _orig_load_state_dict(self, fixed, strict=strict, *args, **kwargs)


if (
    getattr(torch.nn.Module.load_state_dict, "__name__", "")
    != "_patched_load_state_dict"
):
    torch.nn.Module.load_state_dict = _patched_load_state_dict
