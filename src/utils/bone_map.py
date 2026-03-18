import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import SimpleITK as sitk


def _ensure_5d(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0).unsqueeze(0)
    if image.ndim == 4:
        return image.unsqueeze(1) if image.shape[0] > 1 else image.unsqueeze(0)
    if image.ndim == 5:
        return image
    raise ValueError(f"Unsupported image ndim: {image.ndim}")


def _window(image_hu: torch.Tensor, window: Optional[Tuple[float, float]]) -> torch.Tensor:
    if window is None:
        return image_hu
    ww, wc = window
    lwin = wc - ww / 2.0
    rwin = wc + ww / 2.0
    return torch.clamp(image_hu, min=lwin, max=rwin)


def make_target_region_from_hu(
    image: torch.Tensor,
    hu_threshold: float,
    out_size: Optional[int] = 32,
    window: Optional[Tuple[float, float]] = None,
    normalize: bool = False,
    otsu: bool = False,
) -> torch.Tensor:
    """
    Build a binary target region from HU-scale CT volume.

    Args:
        image: Raw HU tensor. Passing normalized [0, 1] tensors is invalid.
        hu_threshold: HU cutoff for bone/non-bone separation.
        out_size: Cubic output size. If None, keep original size.
        window: Optional (ww, wc) HU window applied before thresholding.
        normalize: Deprecated, ignored for compatibility.
        otsu: If True and hu_threshold is None, compute threshold via Otsu.
    """
    if hu_threshold is None and not otsu:
        raise ValueError("hu_threshold must be provided.")
    elif hu_threshold is None and otsu:
        arr = image.cpu().numpy()
        img = sitk.GetImageFromArray(arr)
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.Execute(img)
        hu_threshold = otsu_filter.GetThreshold()
    else:
        hu_threshold = float(hu_threshold)

    if normalize:
        warnings.warn(
            "'normalize' argument is ignored in make_target_region_from_hu; "
            "thresholding always uses HU-scale values.",
            stacklevel=2,
        )

    x = _ensure_5d(image.float())
    x = _window(x, window)

    target_region = (x >= hu_threshold).float()
    if out_size is not None:
        target_region = F.interpolate(target_region, size=(out_size, out_size, out_size), mode="nearest")
    return target_region.contiguous()


def make_target_region_from_threshold(
    image: torch.Tensor,
    hu_threshold: float,
    out_size: Optional[int] = 32,
    window: Optional[Tuple[float, float]] = None,
    normalize: bool = False,
    otsu: bool = False,
) -> torch.Tensor:
    """
    Backward-compatible alias for historical API names.
    """
    return make_target_region_from_hu(
        image=image,
        hu_threshold=hu_threshold,
        out_size=out_size,
        window=window,
        normalize=normalize,
        otsu=otsu,
    )


def make_inverse_region(target_region: torch.Tensor) -> torch.Tensor:
    if target_region.ndim != 5:
        raise ValueError("target_region must be 5D tensor [B, C, D, H, W].")
    return (1.0 - target_region).contiguous()