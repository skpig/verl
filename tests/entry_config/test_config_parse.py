import hydra
from omegaconf import OmegaConf
from tasks.main_ppo import validate_config


@hydra.main(config_path='../../tasks/config', config_name='ppo_trainer', version_base=None)
def mock_main(config):
    validate_config(config)
    OmegaConf.to_container(config, resolve=True)
    OmegaConf.resolve(config)


if __name__ == '__main__':
    # make sure default config is valid
    mock_main()
