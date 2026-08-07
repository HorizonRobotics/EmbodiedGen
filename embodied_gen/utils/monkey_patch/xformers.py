import torch

_BLACKWELL_MINIMUM_COMPUTE_CAPABILITY = (12, 0)


def disable_xformers_flash3() -> bool:
    """Remove FlashAttention 3 from the xFormers dispatcher."""
    try:
        from xformers.ops.fmha import _set_use_fa3
    except (ImportError, AttributeError):
        return False

    _set_use_fa3(False)
    return True


def disable_xformers_flash3_on_blackwell() -> bool:
    """Disable xFormers FlashAttention 3 when a Blackwell GPU is visible."""
    if not torch.cuda.is_available():
        return False

    for device_index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(device_index)
        if capability >= _BLACKWELL_MINIMUM_COMPUTE_CAPABILITY:
            return disable_xformers_flash3()

    return False
