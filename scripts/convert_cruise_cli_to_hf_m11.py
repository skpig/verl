import json
import argparse
import os

from hdfs_io import hcopy


def convert_cruise_config_to_hf_config(cruise_config_path, output_path):
    """Convert cruise_cli.json to hf_config.json"""
    from omnistore.utilities.io import bfile
    from seed_models.models.m11.configuration_m11 import M11Config
    # from convert_m10_config import convert_old_config
    import tempfile
    with bfile.BFile(cruise_config_path, "r") as cruise_config_file:
        trainer_config = json.load(cruise_config_file)

    cruise_config = trainer_config["model"]["network"]

    kwargs = {k: v for k, v in cruise_config.items() if not k.startswith('megatron_')}

    # handle sliding window
    if 'sliding_window' in kwargs:
        sliding_window = kwargs.pop('sliding_window')
        if len(sliding_window) != kwargs['num_hidden_layers']:
            assert len(sliding_window) == 1 and sliding_window[0] == -1

        kwargs['sliding_window'] = sliding_window * kwargs['num_hidden_layers']

    rope_scaling = kwargs.get('rope_scaling', {})
    if len(rope_scaling) == 0:
        print(f"Add default rope_scaling")
        rope_scaling = {"factor": 1.0, "rope_cut": True, "rope_cut_head_dim": 48, "rope_type": "default"}
        kwargs["rope_scaling"] = rope_scaling

    kwargs['architectures'] = ['M11ForCausalLM']
    m11_config = M11Config(**kwargs)
    # import ipdb; ipdb.set_trace()
    tmp_path = tempfile.gettempdir()
    m11_config.save_pretrained(tmp_path)
    hf_config_local_path = os.path.join(tmp_path, "config.json")

    print(hf_config_local_path)

    if output_path is None:
        output_path = os.path.dirname(cruise_config_path)

    # output_path = os.path.join(output_path, "config.json")
    bfile.makedirs(output_path)
    assert hcopy(hf_config_local_path, output_path), "Failed to copy hf config file."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cruise_config_path', required=True)
    parser.add_argument('--output_path')
    args = parser.parse_args()

    convert_cruise_config_to_hf_config(args.cruise_config_path, args.output_path)
