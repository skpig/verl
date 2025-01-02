import hydra
import omegaconf
from omegaconf.dictconfig import DictConfig

from tasks.main_ppo import override
import copy
import os


def compare_diff(old, new, recipe):
    for key, value in old.items():
        assert key in new
        if isinstance(value, DictConfig):
            if key not in recipe:
                compare_diff(value, new[key], {})
            else:
                assert isinstance(recipe[key], DictConfig)
                compare_diff(value, new[key], recipe[key])
        else:
            if key in recipe:
                assert new[key] == recipe[key], f"{new[key]} vs. {recipe[key]}"
            else:
                assert new[key] == value, f"{new[key]} vs. {value}"


def mock_main(recipe):

    @hydra.main(config_path="../../tasks/config", config_name="ppo_trainer", version_base=None)
    def _main(config):
        nonlocal recipe
        old_config = copy.deepcopy(config)
        recipe = omegaconf.OmegaConf.load(recipe)
        override(config, recipe)
        compare_diff(old_config, config, recipe)

    _main()


def test_l20_recipes():
    for recipe in os.listdir("tasks_scripts/recipes/l20"):
        if recipe.endswith(".yaml"):
            mock_main(f"tasks_scripts/recipes/l20/{recipe}")


def test_h800_recipes():
    for recipe in os.listdir("tasks_scripts/recipes/h800"):
        if recipe.endswith(".yaml"):
            mock_main(f"tasks_scripts/recipes/h800/{recipe}")


if __name__ == "__main__":

    test_l20_recipes()
    test_h800_recipes()
