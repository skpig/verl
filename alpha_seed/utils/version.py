from packaging.version import Version
import importlib.metadata


def check_seed_models_version(required_seed_models_version):
    from seed_models import __version__
    assert Version(__version__) >= Version(required_seed_models_version), \
        (f'seed_models version {__version__} is too old. Please upgrade to version {required_seed_models_version} '
         'or higher.')


def check_omnistore_version(required_omnistore_version):
    try:
        actual_omnistore_version = importlib.metadata.version('byted-omnistore')

        assert Version(actual_omnistore_version) >= Version(required_omnistore_version), \
            f'byted-omnistore version {actual_omnistore_version} is too old. Please upgrade to version ' \
            f'{required_omnistore_version} or higher. Example command: pip3 install --upgrade byted-omnistore.'
    except importlib.metadata.PackageNotFoundError as e:
        print(f'byted-omnistore not installed. Please install it and upgrade to version {required_omnistore_version} '
              f'or higher. Example command: pip3 install byted-omnistore=={required_omnistore_version}.')
        raise e
    return actual_omnistore_version
