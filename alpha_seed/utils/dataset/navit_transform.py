import math


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def get_adapooling_factor(img_size=672, patch_size=14, targe_pooling_size=16):
    adapooling_factor = int(img_size / patch_size / targe_pooling_size)
    return adapooling_factor


def get_resized_hw_for_Navit(
    height: int,
    width: int,
    patch_size: int = 14,
    img_size: int = 672,
    min_pixels: int = 14 * 14 * 9,
    max_pixels: int = 14 * 14 * 9 * 1024,
    max_ratio: int = 200,
):
    adapooling_factor = get_adapooling_factor(img_size, patch_size)
    factor = adapooling_factor * patch_size

    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return int(h_bar), int(w_bar)
