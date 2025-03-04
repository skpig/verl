def find_latest_ckpt_path_():
    try:
        from omnistore.utilities.ckpt_format_tool import find_latest_ckpt_path
    except ImportError:
        from omnistore.utilities.ckpt_format.common_utils import find_latest_ckpt_path
    return find_latest_ckpt_path
