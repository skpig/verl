from packaging.version import Version


def _check_version():
    import vescale
    if Version(vescale.__version__) < Version("0.2.19"):
        raise RuntimeError(f"vescale version must be >= 0.2.19, but got {vescale.__version__}. "
                           "Please install through pip3 install byted-vescale==0.2.19")
    import triton
    if Version(triton.__version__) < Version("3.3.0"):
        raise RuntimeError(f"triton version must be >= 3.3.0, but got {triton.__version__}. "
                           "Please install through pip3 install triton==3.3.0")
