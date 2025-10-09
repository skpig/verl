import os
import math
import re
import json
from omegaconf import OmegaConf
import copy
import pandas as pd
import torch
import random
from alpha_seed.prompts.think_template_utils import get_thinking_system_prompt
from .vlm_rl_dataset import RLHFDatasetVL, convert_conversation_to_prompt, get_reward_model
from alpha_seed.utils.reward_score import select_prepare_vlm_rm_input_fn


def check_nan(v) -> bool:
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


class MixRLDataset(RLHFDatasetVL):

    def __init__(self, *args, **kwargs):
        self.config = kwargs['config']
        self.add_thinking_prompt = self.config.data.get('add_thinking_prompt', False)
        self.add_thinking_prompt_ratio = self.config.data.get('add_thinking_prompt_ratio', 1.0)

        # For Ability
        self.override_ability_with_datasource = self.config.data.get('override_ability_with_datasource', True)
        self.override_datasource_with_ability = self.config.data.get('override_datasource_with_ability', False)
        self.ability_list = self.config.data.get('ability_list', 'code,math')
        self.ability_kl_weights = self.config.data.get('ability_kl_weights', '1.0,1.0')
        self.ability_keys = ["unknown"] + self.ability_list.split(",")
        self.ability_dict = dict(map(lambda x: (x[1], x[0]), enumerate(self.ability_keys)))
        self.ability_kl_weights = [1.0] + [float(x) for x in self.ability_kl_weights.split(',')]
        self.rm_required_abilities = self.config.data.get('rm_required_abilities', 'code,math')
        self.rm_required_abilities = ["unknown"] + self.rm_required_abilities.split(",")
        self.is_eval = kwargs.get('is_eval', False)
        if self.is_eval:
            self.override_datasource_with_ability = False
            self.override_ability_with_datasource = True
            self.add_thinking_prompt_ratio = 1.0
        os.environ['THINK_TEMPLATE'] = self.config.data.think_template

        super().__init__(*args, **kwargs)
        self.max_rm_prompt_length = OmegaConf.select(
            self.config,
            f"reward_model.{self.remote_rm_type}.max_prompt_length",
            default=self.config.data.max_prompt_length,
        )

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        if 'session' in self.dataframe.iloc[item] and self.dataframe.iloc[item]['session'] is not None:
            row_dict = self.dataframe.iloc[item]['session']
        else:
            row_dict = self.dataframe.iloc[item].to_dict()

        is_image = row_dict.get('images_bytes_ref', None) is not None  # Support pure text data

        row_dict_ret = {}
        row_dict_ret["agent_handler"] = row_dict.get("agent_handler", "")
        row_dict_ret["initial_files"] = row_dict.get("initial_files", json.dumps({'initial_files': {}}))

        # set ability with default value
        ability = row_dict.get("ability", "unknown")
        if self.override_ability_with_datasource:
            ability = row_dict.get("data_source", "unknown")
        if self.override_datasource_with_ability:
            row_dict["data_source"] = ability
        if ability not in self.ability_dict:
            ability = "unknown"
        ability_idx = self.ability_dict.get(ability, 0)
        row_dict_ret["ability_idx"] = torch.tensor(ability_idx)
        row_dict_ret["ability_kl_weights"] = self.ability_kl_weights[ability_idx]

        chat = row_dict[self.prompt_key]
        # encode prompts without chat template
        if self.return_raw_chat:
            if isinstance(chat[0], str):
                row_dict_ret['raw_prompt'] = [{
                    "role": "user" if idx % 2 == 0 else "assistant",
                    "content": chat[idx]
                } for idx in range(len(chat))]
            elif isinstance(chat[0], dict):
                row_dict_ret['raw_prompt'] = copy.deepcopy(chat)
            else:
                raise NotImplementedError("raw_prompt must be str or dict")
        if isinstance(chat[0], dict):
            chat = [c["content"] for c in chat]

        user_contents = []
        for idx, prompt in enumerate(chat):
            if ability in ["verifiable_function_call"]:
                prompt = json.loads(prompt)
                if 'name' in prompt and prompt['name'] is not None and prompt['name'] != '':
                    prompt_str = f'{self.tokenizer.bos_token}{prompt["role"]} name={prompt["name"]}\n' + \
                                 f'{prompt["content"]}{self.tokenizer.eos_token}'
                else:
                    prompt_str = f'{self.tokenizer.bos_token}{prompt["role"]}\n' + \
                                 f'{prompt["content"]}{self.tokenizer.eos_token}'
                user_contents.append({"type": "text", "text": prompt_str})
            elif idx % 2 == 0:
                user_prompt = self.tokenizer.bos_token + 'user\n' + prompt + self.tokenizer.eos_token
                if is_image and "<image>" in user_prompt:
                    prompt_chunks = re.split(r"(<image>)", user_prompt)
                    prompt_chunks = [item for item in prompt_chunks if item]
                    for chunk_idx, prompt_chunk in enumerate(prompt_chunks):
                        if prompt_chunk == '<image>':
                            user_contents.append({"type": "image"})
                        else:
                            user_contents.append({"type": "text", "text": prompt_chunk})
                else:
                    user_contents.append({"type": "text", "text": user_prompt})
            else:
                # Suppose all of the assistant response is text
                assistant_prompt = self.tokenizer.bos_token + 'assistant\n' + prompt + self.tokenizer.eos_token
                user_contents.append({"type": "text", "text": assistant_prompt})
        user_contents.append({"type": "text", "text": f"{self.tokenizer.bos_token}assistant\n"})

        conversation = []
        # Processing thinking sp (for mixRL)
        no_thinking_required = row_dict.get('no_thinking_required', False)
        if no_thinking_required is None or check_nan(no_thinking_required):  # Pandas None value
            no_thinking_required = False
        if self.add_thinking_prompt:
            if (random.random() < self.add_thinking_prompt_ratio) and (not no_thinking_required):
                no_thinking_required = False  # Enable thinking mode
            else:
                no_thinking_required = True  # Disable thinking mode
            THINKING_SP = get_thinking_system_prompt(no_thinking_required=no_thinking_required)
            if THINKING_SP:
                conversation.append({
                    "role":
                        "system",
                    "content": [{
                        "type": "text",
                        "text": f"{self.tokenizer.bos_token}system\n{THINKING_SP}{self.tokenizer.eos_token}"
                    }]
                })
        row_dict_ret["no_thinking_required"] = torch.tensor(1) if no_thinking_required else torch.tensor(0)

        system_prompt = row_dict.get('system_prompt', '')
        if system_prompt:
            if ability in ["verifiable_function_call"]:
                system_prompt_str = ''
                for prompt_turn in json.loads(system_prompt):
                    if 'name' in prompt_turn and prompt_turn['name'] is not None and prompt_turn['name'] != '':
                        system_prompt_str += f'{self.tokenizer.bos_token}{prompt_turn["role"]} name={prompt_turn["name"]}\n' + \
                                             f'{prompt_turn["content"]}{self.tokenizer.eos_token}'
                    else:
                        system_prompt_str += f'{self.tokenizer.bos_token}{prompt_turn["role"]}\n' + \
                                             f'{prompt_turn["content"]}{self.tokenizer.eos_token}'
                conversation.append({"role": "system", "content": [{"type": "text", "text": system_prompt_str}]})
            else:
                conversation.append({
                    "role":
                        "system",
                    "content": [{
                        "type": "text",
                        "text": f"{self.tokenizer.bos_token}system\n{system_prompt}{self.tokenizer.eos_token}"
                    }],
                })
        conversation.append({"role": "user", "content": user_contents})
        prompt = convert_conversation_to_prompt(conversation, self.config)

        # reward_model is required
        verifier_feature = row_dict.get('verifier_feature', '')
        if verifier_feature is None or check_nan(verifier_feature):
            verifier_feature = ""
        if verifier_feature == "" and self.remote_rm_type:
            reward_model = {
                # NOTE: at this point it only flags that remote service is required
                'style': f"remote_service",
                'ground_truth': row_dict.get('rm_reference_response'),
            }
            row_dict_ret['reward_model'] = reward_model
        else:
            row_dict_ret['reward_model'] = get_reward_model(row_dict, self.use_vlm_verifier_router)

        if 'prompt_id' in row_dict:
            row_dict_ret['prompt_id'] = row_dict['prompt_id']

        # 添加answer
        if self.use_ref_answer:
            answer = row_dict.get(self.answer_key, "")
        else:
            answer = ""
        if answer and not pd.isna(answer):
            raise NotImplementedError("ref answer is not supported now")

        index = row_dict.get("extra_info", {}).get("index", item)  ## important for grpo to group info
        row_dict_ret["index"] = index
        row_dict_ret['prompt_names'] = [""]
        row_dict_ret['prompt'] = prompt

        prepare_kwargs = dict(
            tokenizer=self.tokenizer,
            row_dict=row_dict,
            prompt_key=self.prompt_key,
            is_image=is_image,
            max_prompt_len=self.max_rm_prompt_length,
            max_resp_len=self.max_response_length,
        )

        if self.remote_rm_type is not None:
            if ability not in self.rm_required_abilities or self.is_eval:
                row_dict_ret['reward_model']['rm_required_type'] = None
            else:
                rm_input = select_prepare_vlm_rm_input_fn(self.config)(**prepare_kwargs)
                row_dict_ret['reward_model'].update(rm_input)
                row_dict_ret['reward_model']['images_bytes_ref'] = row_dict['images_bytes_ref']
                row_dict_ret['reward_model']['rm_required_type'] = self.remote_rm_type

        # Add critic_reference_response
        critic_reference_response = row_dict.get('critic_reference_response', None)
        if critic_reference_response is None or check_nan(critic_reference_response) or len(
                critic_reference_response) == 0:
            rm_reference_resp = row_dict.get('ground_truth', None)
            if rm_reference_resp is None or check_nan(rm_reference_resp) or len(rm_reference_resp) == 0:
                critic_reference_response = "No ground truth is found."
            else:
                critic_reference_response = rm_reference_resp
        row_dict_ret['critic_reference_response'] = critic_reference_response

        row_dict_ret['data_source'] = row_dict['data_source']
        row_dict_ret['off_policy_steps'] = torch.zeros([1]).to(torch.int8)
        row_dict_ret['images_bytes_ref'] = row_dict['images_bytes_ref']
        return row_dict_ret
