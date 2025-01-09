import ray
import hydra
from omegaconf import OmegaConf
from verl.utils.tracking import Tracking
from alpha_seed.trainer.ppo import RayPPOTrainer
from alpha_seed.workers.actors.checkpoint import CkptGlobalUploader
from tasks.main_ppo import validate_config, RewardManager


class ClientPPOTrainer(RayPPOTrainer):

    def __init__(self, kv_store_name="kv_store"):
        self.kv_store = ray.get_actor(name=kv_store_name)

        config = ray.get(self.kv_store.get_by_key.remote('config'))
        validate_config(config=config)
        tokenizer = ray.get(self.kv_store.get_by_key.remote('tokenizer'))

        resource_pool_manager = ray.get(self.kv_store.get_by_key.remote('resource_pool_manager'))
        ray_worker_group_cls = ray.get(self.kv_store.get_by_key.remote('ray_worker_group_cls'))

        role_worker_mapping = ray.get(self.kv_store.get_by_key.remote('role_worker_mapping'))
        available_roles = list(role_worker_mapping.keys())

        print(f"This server provides roles: {', '.join([str(role) for role in available_roles])}")
        # pop out roles that will not be used, like:
        # role_worker_mapping.pop(Role.Critic)

        logger = Tracking(project_name=config.trainer.project_name,
                          experiment_name=config.trainer.experiment_name,
                          default_backend=config.trainer.logger,
                          config=OmegaConf.to_container(config, resolve=True))

        # we have to create reward_fn and val_reward_fn at ClientPPOTrainer because RewardManager cannot be serialized
        reward_fn = RewardManager(tokenizer=tokenizer, config=config, logger=logger, rm_name="train")
        val_reward_fn = RewardManager(tokenizer=tokenizer, config=config, logger=logger, rm_name="val")

        super().__init__(config=config,
                         tokenizer=tokenizer,
                         role_worker_mapping=role_worker_mapping,
                         resource_pool_manager=resource_pool_manager,
                         ray_worker_group_cls=ray_worker_group_cls,
                         reward_fn=reward_fn,
                         val_reward_fn=val_reward_fn,
                         logger=logger)

    def init_workers(self):
        """Connect to worker group"""
        super().init_workers(ckpt_global_uploader=None)

    def fit(self):
        super().fit()


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    ray.init(namespace="alphaseed", address="auto")

    trainer = ClientPPOTrainer()
    print(trainer)
    trainer.init_workers()
    print("client setup ready")
    trainer.fit()
    print("client.fit() done")


if __name__ == '__main__':
    main()
