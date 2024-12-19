import numbers
from typing import List, Tuple

import numpy as np

DataType = numbers.Number


# ===========================================================
# ============= Moving Average Functions ====================
# ===========================================================
def simple_moving_average(baseline_vals: List[DataType], running_vals: List[DataType],
                          **kwargs) -> Tuple[List[DataType], List[DataType]]:
    window_size = kwargs.get("window_size", None)
    assert window_size is not None and window_size >= 1, f"window size must be greater than 1, but got {window_size}"

    baseline_len = len(baseline_vals)
    running_len = len(running_vals)
    assert baseline_len == running_len, "baseline and running must have the same length"

    baseline = np.array(baseline_vals)
    running = np.array(running_vals)

    pad_size = window_size - 1

    left_pad_size = pad_size // 2
    right_pad_size = pad_size - left_pad_size

    baseline = np.pad(baseline, (left_pad_size, right_pad_size), mode='edge')
    running = np.pad(running, (left_pad_size, right_pad_size), mode='edge')
    baseline_convolved = np.convolve(baseline, np.ones(window_size) / window_size, mode='valid')
    running_convolved = np.convolve(running, np.ones(window_size) / window_size, mode='valid')

    assert len(baseline_convolved) == len(running_convolved), "baseline and running convolved must have the same length"
    assert len(baseline_convolved) == baseline_len, "baseline and running convolved must have the same length"

    return baseline_convolved.tolist(), running_convolved.tolist()


# ===========================================================
# ================ Helper Functions =========================
# ===========================================================
def get_tolerance_bounds(**kwargs):
    """
    get tolerance bounds from kwargs
    """
    tolerance_lower_bound = kwargs.get("tolerance_lower_bound", -1e-3)
    tolerance_upper_bound = kwargs.get("tolerance_upper_bound", 1e-3)

    if tolerance_lower_bound == "-inf":
        tolerance_lower_bound = -np.inf

    if tolerance_upper_bound == "inf":
        tolerance_upper_bound = np.inf

    return tolerance_lower_bound, tolerance_upper_bound


# ===============================================
# ============= Statistics Functions ============
# ===============================================
def absolute_diff(baseline_val: DataType, running_val: DataType,
                  **kwargs) -> Tuple[Tuple[bool, numbers.Number], numbers.Number]:
    """
    absolute diff between running and baseline
    """
    tolerance_lower_bound, tolerance_upper_bound = get_tolerance_bounds(**kwargs)

    diff = running_val - baseline_val
    success = diff >= tolerance_lower_bound and diff <= tolerance_upper_bound

    return success, diff, baseline_val, running_val


def absolute_upper_bound(baseline_val: DataType, running_val: DataType,
                         **kwargs) -> Tuple[Tuple[bool, numbers.Number], numbers.Number]:
    """
    absolute diff between running and constant
    """
    constant = kwargs.get("constant", 0.0)
    diff = running_val - constant
    success = diff <= 0

    return success, diff, baseline_val, running_val


def relative_diff(baseline_val: DataType, running_val: DataType, **kwargs) -> Tuple[bool, numbers.Number]:
    """
    relative diff between running and baseline
    """
    tolerance_lower_bound, tolerance_upper_bound = get_tolerance_bounds(**kwargs)

    if baseline_val == 0:
        diff = 0
    else:
        diff = (running_val - baseline_val) / baseline_val
    success = diff >= tolerance_lower_bound and diff <= tolerance_upper_bound

    return success, diff, baseline_val, running_val


def max_val_diff(baseline_vals: List[DataType], running_vals: List[DataType], **kwargs) -> Tuple[bool, numbers.Number]:
    """
    diff between max val of running and max val of baseline
    """
    tolerance_lower_bound, tolerance_upper_bound = get_tolerance_bounds(**kwargs)
    running_max_val = np.max(np.array(running_vals))
    baseline_max_val = np.max(np.array(baseline_vals))
    diff = running_max_val - baseline_max_val
    success = diff >= tolerance_lower_bound and diff <= tolerance_upper_bound

    return success, diff, baseline_max_val, running_max_val


def max_val_upper_bound(baseline_vals: List[DataType], running_vals: List[DataType],
                        **kwargs) -> Tuple[bool, numbers.Number]:
    """
    diff between max val of running and the upper bound
    """
    constant = kwargs.get("constant", 0.0)
    running_max_val = np.max(np.array(running_vals))
    baseline_max_val = np.max(np.array(baseline_vals))
    diff = running_max_val - constant
    success = diff <= 0

    return success, diff, baseline_max_val, running_max_val


def min_val_diff(baseline_vals: List[DataType], running_vals: List[DataType], **kwargs) -> Tuple[bool, numbers.Number]:
    """
    diff between min val of running and min val of baseline
    """
    tolerance_lower_bound, tolerance_upper_bound = get_tolerance_bounds(**kwargs)

    running_min_val = np.min(np.array(running_vals))
    baseline_min_val = np.min(np.array(baseline_vals))
    diff = running_min_val - baseline_min_val

    success = diff >= tolerance_lower_bound and diff <= tolerance_upper_bound

    return success, diff, baseline_min_val, running_min_val


def mean_val_diff(baseline_vals: List[DataType], running_vals: List[DataType], **kwargs) -> Tuple[bool, numbers.Number]:
    """
    diff between mean val of running and mean val of baseline
    """

    tolerance_lower_bound, tolerance_upper_bound = get_tolerance_bounds(**kwargs)
    running_mean_val = np.mean(np.array(running_vals))
    baseline_mean_val = np.mean(np.array(baseline_vals))
    diff = running_mean_val - baseline_mean_val
    success = diff >= tolerance_lower_bound and diff <= tolerance_upper_bound

    return success, diff, baseline_mean_val, running_mean_val


def percentile_val_diff(baseline_vals: List[DataType], running_vals: List[DataType],
                        **kwargs) -> Tuple[bool, numbers.Number]:
    """
    diff between percentile val of running and percentile val of baseline
    """

    percentile = kwargs.get("percentile", 90)  # default p90

    tolerance_lower_bound, tolerance_upper_bound = get_tolerance_bounds(**kwargs)
    running_percentile_val = np.percentile(running_vals, percentile)
    baseline_percentile_val = np.percentile(baseline_vals, percentile)

    diff = running_percentile_val - baseline_percentile_val
    success = diff >= tolerance_lower_bound and diff <= tolerance_upper_bound

    return success, diff, baseline_percentile_val, running_percentile_val
