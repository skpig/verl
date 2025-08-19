import ray
import os
import sys
import getopt
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_gpus=0, num_cpus=0)
class Patcher:

    def __init__(self):
        pass

    def prepare_python3_env(self):
        # os.system("git config pull.ff only")
        os.system("cp -rp /mnt/hdfs/models_m10/ /opt/tiger/models_m10/")
        # os.system(
        # "sudo pip3 install --no-cache-dir https://luban-source.byted.org/repository/scm/data.aml.xperf_gpt_th24_cu124_abi0_sdist_1.0.0.1245.tar.gz"
        # )

        # os.system("cd /opt/tiger && rm -rf mariana && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/mariana && cd mariana && git checkout zr/sc/xperf_infer")
        # os.system("cd /opt/tiger/mariana && git pull")
        # os.system("cd /opt/tiger && rm -rf seed_models && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/seed_models && cd seed_models && git checkout chi/feat/p6_ggemm_ckpt")
        # os.system("pip3 install --no-cache-dir torchvision==0.19.1 timm")
        # os.system("cd /opt/tiger && rm -rf verifiable_tasks && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/verifiable_tasks")
        return
        os.system("pip3 install bytedance.trainingmetrics -i https://bytedpypi.byted.org/simple/")
        os.system(
            "cd /opt/tiger && rm -rf mariana && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/mariana && cd mariana && git checkout fwq_sing"
        )

        # os.system("cd /opt/tiger/mariana && git checkout 38b36efdd00e4eee355e19cf15642339c1387224")

        # os.system("mkdir -p /opt/tiger/debug_data")

        # os.system("cd /opt/tiger/ && rm -rf cruise && bvc clone data/aml/cruise --version 1.0.0.2933")

        # os.system("pip3 install t
        # ensordict==0.3.0")
        # os.system("pip3 uninstall -y torch byted_torch")
        # os.system("python3 -m pip install --no-cache-dir http://luban-source.byted.org/repository/scm/lab.pytorch.pytorch2_cu121_1.0.0.182.tar.gz && python3 -m pip --timeout 3600 install --no-cache-dir --pre torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121")
        # os.system("cd /opt/tiger && rm -rf verl && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/data/verl && cd verl && git checkout zr/async_xperf")
        # os.system("cd /opt/tiger/ && rm -rf Megatron-LM && mkdir Megatron-LM && cd Megatron-LM && bvc clone aml/mlsys/megatron_pt2_cu121 --version 1.0.0.761 -f")
        # os.system("cd /opt/tiger && rm -rf cruise && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/data/cruise && cd cruise && git checkout zr/fix_stop")

        return
        os.system(
            "pip3 install https://luban-source.byted.org/repository/scm/data.aml.xperf_gpt_th21_cu121_abi0_sdist_1.0.0.379.tar.gz"
        )
        return
        os.system("pip3 install numpy==1.21.2")
        os.system("mkdir -p /opt/tiger/debug_data")
        os.system("pip3 install --upgrade byted-wandb")
        os.system("pip3 install hydra-core --upgrade")
        os.system(
            "cd /opt/tiger && rm -rf mariana && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/mariana && cd mariana && git checkout zr/support_rl_dssp"
        )
        os.system(
            "cd /opt/tiger && rm -rf cruise && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/data/cruise && cd cruise && git checkout master"
        )
        os.system(
            "cd /opt/tiger && rm -rf mariana_metadata && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/mariana_metadata"
        )
        os.system(
            "cd /opt/tiger && rm -rf instruction_following_eval && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/instruction_following_eval"
        )
        os.system(
            "cd /opt/tiger && pip3 install http://luban-source.byted.org/repository/scm/data.aml.verl_1.0.0.60.tar.gz")
        os.system(
            "cd /opt/tiger && pip3 uninstall byted-torch torch -y && pip3 install https://d.scm.byted.org/api/v2/download/lab.pytorch.pytorch2_cu121_1.0.0.73.tar.gz && pip3 install --no-cache-dir --pre torchvision==0.16.0 torchaudio==2.1.0"
        )

        if self.cuda_version == 11:
            os.system("pip3 install cupy-cuda11x")
            os.system(
                "http_proxy=http://sys-proxy-rd-relay.byted.org:3128 https_proxy=http://sys-proxy-rd-relay.byted.org:3128 no_proxy= python3 -m cupyx.tools.install_library --library nccl --cuda 11.x"
            )
            os.system(
                "pip3 install https://luban-source.byted.org/repository/scm/data.aml.xperf_gpt_th113_cu117_abi0_sdist_1.0.0.409.tar.gz"
            )
            os.system("cd /opt/tiger/ && rm -rf Megatron-LM")
            os.system("cd /opt/tiger/ && mkdir Megatron-LM && cd Megatron-LM && bvc clone aml/mlsys/megatron -f")
            os.system("cd /opt/tiger/Megatron-LM/megatron && pip3 install .")
            os.system("pip3 install /opt/tiger/mariana_metadata/tools/th113/rotary_emb-0.1-cp39-cp39-linux_x86_64.whl")
        else:
            os.system("pip3 install cupy-cuda12x")
            os.system(
                "http_proxy=http://sys-proxy-rd-relay.byted.org:3128 https_proxy=http://sys-proxy-rd-relay.byted.org:3128 no_proxy=  python3 -m cupyx.tools.install_library --library nccl --cuda 12.x"
            )
            os.system(
                "cd /opt/tiger && pip3 install https://luban-source.byted.org/repository/scm/data.aml.xperf_gpt_th21_cu121_abi0_sdist_1.0.0.179.tar.gz"
            )
            os.system(
                "pip3 install https://luban-source.byted.org/repository/scm/data.aml.lego_ops_th21_cu121_cudnn890_abi0_sdist_1.0.0.184.tar.gz"
            )
            os.system("cd /opt/tiger/ && rm -rf Megatron-LM")
            os.system(
                "cd /opt/tiger/ && mkdir Megatron-LM && cd Megatron-LM && bvc clone aml/mlsys/megatron_pt2_cu121 --version 1.0.0.339 -f"
            )
            os.system("cd /opt/tiger/Megatron-LM/megatron && pip3 install .")
            os.system("pip3 install /opt/tiger/mariana_metadata/tools/th121/rotary_emb-0.1-cp39-cp39-linux_x86_64.whl")

            os.system(
                "cd /opt/tiger/ && rm -rf bpex_kernel && mkdir bpex_kernel && cd bpex_kernel && wget https://luban-source.byted.org/repository/scm/data.aml.pytorch.bpex_kernel_pt2_v1_1.0.0.2.tar.gz && tar -xf data.aml.pytorch.bpex_kernel_pt2_v1_1.0.0.2.tar.gz"
            )

            os.system("cd /opt/tiger/ && rm -rf janus")
            os.system(
                "cd /opt/tiger/ && mkdir janus && cd janus && bvc clone aml/mlsys/janus_dev --version 1.0.0.767 -f && cd /opt/tiger/janus/janus_dev && pip3 install ."
            )

        if self.run_moe:
            os.system(
                "cd /opt/tiger && rm -rf mariana && git clone -b official_moe_pretrain_231018_sft1207 https://yipzlf:mUuXbsNc9VzHNsLAKy7_@code.byted.org/seed/mariana && cd mariana"
            )

        return True

    def prepare_rl_file(self):
        print("Start prepare_rl_file")

        # 100b
        # os.system(
        #     "hdfs dfs -copyToLocal hdfs://haruna/byte_search/seed/model/xurunxin.nlp/rm_rl_data/rl/rl_balance_210k_general_50k_security_3k_repeat_2k_baike_2k_cs_add_ability1.parquet /opt/tiger/rl_balance_210k_general_50k_security_3k_repeat_2k_baike_2k_cs_add_ability1.parquet"
        # )
        # os.system(
        #     "cd /opt/tiger/ && /opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/lf_lq/user/zuoxiaochen/bernard/Seed-100B-SFT-P3.0.0_D4.8.2_T1400B-SFT8.9.3.1_seedcode/megatron_merge_states.pt -s -c 512 --ct 32 -t 8 ./ && mv megatron_merge_states.pt seed_policy_ckpt.pt"
        # )
        # os.system(
        #     "cd /opt/tiger/ && /opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/byte_search/seed/model/wuwei.ai/daily_rm/100B-0901-top90/deploy/zero3_merge_states.pt -s -c 512 --ct 32 -t 8 ./ && mv zero3_merge_states.pt seed_value_ckpt.pt"
        # )

        # 1b3+13b
        # os.system(
        #     "/opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/byte_search/seed/model/xurunxin.nlp/rm_rl_data/rl/train_0616.parquet  -s -c 512 --ct 32 -t 8 /opt/tiger/"
        # )
        # os.system(
        #     "/opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/hcache/centralize_lq/gpt_java/user/xiagenyuan/models/Seed-1.3B-SFT-P3.0.0_D4.0.0_T600B-SFT4.0.0/checkpoints/global_step_10630/zero3_merge_states.pt  -s -c 512 --ct 32 -t 8 /opt/tiger/"
        # )
        # os.system(
        #     "/opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/lf_lq/user/zuoxiaochen/bernard/13b_oneepoch_4e-5_epoch1_0529_0.1good_0.1bad_rej_half1/zero3_merge_states_value.pt  -s -c 512 --ct 32 -t 8 /opt/tiger/"
        # )

        # 13b + 13b p4
        # os.system("/opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/byte_search/seed/model/xurunxin.nlp/rm_rl_data/rl/train_0616.parquet  -s -c 512 --ct 32 -t 8 /opt/tiger/")
        # os.system(
        #     "rm -rf /opt/tiger/megatron_merge_states.pt && /opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/lf_lq/user/zuoxiaochen/eval/13b_sft_p400_d482_0918_p4/megatron_merge_states.pt -s -c 512 --ct 32 -t 8 /opt/tiger/ && mv /opt/tiger/megatron_merge_states.pt /opt/tiger/megatron_merge_states_13b_sprout_policy.pt"
        # )
        # os.system(
        #     "rm -rf /opt/tiger/megatron_merge_states.pt && /opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/lf_lq/user/chenzhongxiang/models/daily_rm/20231009_scaling_13b-p4/checkpoints/global_step_1655/megatron_merge_states.pt  -s -c 512 --ct 32 -t 8 /opt/tiger/ && mv /opt/tiger/megatron_merge_states.pt /opt/tiger/megatron_merge_states_13b_rm.pt"
        # )

        # # 25b + 25b p4
        # os.system("/opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/byte_search/seed/model/xurunxin.nlp/rm_rl_data/rl/20w_exp_20231205.parquet -s -c 512 --ct 32 -t 8 /opt/tiger/")
        # os.system(
        #     "rm -rf /opt/tiger/megatron_merge_states.pt && /opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/ssd_yg/user/yanminghui/sft_train/Seed-25B-SFT-P4.0.0_D4.8.3_T1300B-SFT15.0.0_new/checkpoints/global_step_12756/megatron_merge_states.pt -s -c 512 --ct 32 -t 8 /opt/tiger/ && mv /opt/tiger/megatron_merge_states.pt /opt/tiger/25b_p4_ref.pt"
        # )
        # os.system(
        #     "rm -rf /opt/tiger/megatron_merge_states.pt && /opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/ssd_yg/user/weichengzhi/com_train/25b_sft15_rm_20231131/top90/checkpoints/global_step_2956/megatron_merge_states.pt  -s -c 512 --ct 32 -t 8 /opt/tiger/ && mv /opt/tiger/megatron_merge_states.pt /opt/tiger/25b_p4_rm.pt"
        # )

        # # 25b + 25b p4 32k
        # os.system(
        #     "/opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/ssd_yg/user/weichengzhi/com_data/v1.20.0/com_v1.20.0_train.parquet -s -c 512 --ct 32 -t 8 /opt/tiger/"
        # )
        # os.system(
        #     "rm -rf /opt/tiger/megatron_merge_states.pt && /opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/hl_lq/user/linye/sprout/25b_ct1.0_sftv2.17.5_gpu32_gbs8/checkpoints/global_step_6990/megatron_merge_states.pt -s -c 512 --ct 32 -t 8 /opt/tiger/ && mv /opt/tiger/megatron_merge_states.pt /opt/tiger/25b_p4_ref_32k.pt"
        # )
        # os.system(
        #     "rm -rf /opt/tiger/megatron_merge_states.pt && /opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/ssd_yg/user/weichengzhi/com_train/v1.20.0/long_ctx_test1/checkpoints/global_step_2512/megatron_merge_states.pt  -s -c 512 --ct 32 -t 8 /opt/tiger/ && mv /opt/tiger/megatron_merge_states.pt /opt/tiger/25b_p4_rm_32k.pt"
        # )

        # 200b + 200b p4
        # if os.system("ls /opt/tiger/pm_200b_p4_4k.pt") != 0:
        #     os.system(
        #         "/opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangru.1994/ror/sprout_4k/megatron_merge_states.pt -s -c 512 --ct 32 -t 8 /opt/tiger/ && mv /opt/tiger/megatron_merge_states.pt /opt/tiger/pm_200b_p4_4k.pt"
        #     )
        # if os.system("ls /opt/tiger/megatron_merge_states_200b_p4_value.pt") != 0:
        #     os.system(
        #         "rm -rf /opt/tiger/megatron_merge_states.pt && /opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/ssd_yg/user/wangwei.ww/rm_model/sprout/sproutv08-sftv15-rm20231124/checkpoints/global_step_3089/megatron_merge_states.pt -s -c 512 --ct 32 -t 8 /opt/tiger/ && mv /opt/tiger/megatron_merge_states.pt /opt/tiger/megatron_merge_states_200b_p4_value.pt"
        #     )

        # 3b3 moe
        # os.system(
        #     "/opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/ssd_hl/user/zhudefa/gpt_scaling/3b2_p4_sft1_4in32_48layer_3072v10240v48h_8dense_bs2048_lr3en4_104000_4en5/checkpoints/global_step_11728/megatron_merge_states.pt  -s -c 512 --ct 32 -t 8 /opt/tiger/"
        # )

        # 1T moe
        # print("/opt/tiger/global_step_363000_reshard_tp32 not exists, download!")
        # os.system(
        #     "/opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/ssd_hl/evals_pipeline/home/byte_data_seed/ssd_lq/p0/ckpts/official_sapling/moe_1T_vocab155k_bs4m5_lr3p89_stage2v3/checkpoints/global_step_363000_reshard_tp32  -s -c 512 --ct 32 -t 8 /opt/tiger/"
        # )

        # 25b edu plugin
        # os.system("rm -rf /opt/tiger/megatron_merge_states.pt && rm -rf /opt/tiger/25b_with_edu_plugin.pt")
        # os.system(
        #     "/opt/tiger/hdfs_client/bin/hdfs get hdfs://haruna/home/byte_data_seed/lf_lq/user/xiangxiang.zhang/seed_model/sft/pt2_seed_25b_sft_v16_edu_calculator_0122/checkpoints/global_step_15448/megatron_merge_states.pt  -s -c 512 --ct 32 -t 8 /opt/tiger/ && mv /opt/tiger/megatron_merge_states.pt  /opt/tiger/25b_with_edu_plugin.pt"
        # )

        # print("End prepare_rl_file")
        return True

    def clean_ray_cluster(self):
        # # for clean disk
        # os.system("rm -rf /opt/tiger/ray/session_latest/runtime_resources/working_dir_files/* ")
        # os.system("rm -rf /dev/shm/* ")
        # os.system("rm -rf /tmp/* ")
        # os.system("rm -rf /opt/tiger/debug_data*/* ")
        # os.system("df -h")
        # os.system("nvidia-smi")
        return True


@ray.remote(num_gpus=0)
class HeadPatcher:

    def __init__(self, cuda_version):
        self.local_dir = os.path.abspath('')
        self.cuda_version = cuda_version

    def run(self, cuda_version):
        # os.system("cd /opt/tiger && rm -rf verl && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/data/verl verl && cd verl && git checkout nozomi/unblocking_xp && git pull")
        # os.system("cd /opt/tiger && rm -rf mariana && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/mariana && cd mariana && git checkout zr/verl/xperf_infer")
        os.system(
            "cd /opt/tiger && rm -rf mariana && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/mariana && cd mariana && git checkout fwq_sing"
        )
        return
        os.system(
            "cd /opt/tiger && rm -rf mariana && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/mariana && cd mariana && git checkout zr/verl/xperf_infer"
        )
        os.system("pip3 install tensordict")
        os.system("pip3 uninstall -y torch byted_torch")
        os.system(
            "python3 -m pip install --no-cache-dir http://luban-source.byted.org/repository/scm/lab.pytorch.pytorch2_cu121_1.0.0.182.tar.gz && python3 -m pip --timeout 3600 install --no-cache-dir --pre torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121"
        )
        return
        os.system(
            "cd /opt/tiger && rm -rf instruction_following_eval && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/instruction_following_eval"
        )
        os.system(
            "cd /opt/tiger && rm -rf mariana && git clone https://yipzlf:mUuXbsNc9VzHNsLAKy7_@code.byted.org/seed/mariana"
        )
        os.system(
            "cd /opt/tiger && rm -rf mariana && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/seed/mariana && cd mariana && git checkout zr/support_rl_dssp"
        )
        os.system(
            "cd /opt/tiger && rm -rf cruise && git clone https://yipzlf:vwZZ_JYy7mkx-1zUjyeR@code.byted.org/data/cruise && cd cruise && git checkout ror_fix"
        )
        os.system(
            "cd /opt/tiger && pip3 install http://luban-source.byted.org/repository/scm/data.aml.verl_1.0.0.60.tar.gz")
        os.system(
            "cd /opt/tiger && pip3 uninstall byted-torch torch -y && pip3 install https://d.scm.byted.org/api/v2/download/lab.pytorch.pytorch2_cu121_1.0.0.73.tar.gz && pip3 install --no-cache-dir --pre torchvision==0.16.0 torchaudio==2.1.0"
        )

        os.system("pip3 install hydra-core --upgrade")
        if cuda_version == 11:
            os.system("cd /opt/tiger/ && rm -rf Megatron-LM")
            os.system("cd /opt/tiger/ && mkdir Megatron-LM && cd Megatron-LM && bvc clone aml/mlsys/megatron -f")
        else:
            os.system("cd /opt/tiger/ && rm -rf Megatron-LM")
            os.system(
                "cd /opt/tiger/ && mkdir Megatron-LM && cd Megatron-LM && bvc clone aml/mlsys/megatron_pt2_cu121 --version 1.0.0.339 -f"
            )

        return True


if __name__ == '__main__':
    ray.init()
    node_id_list = [n['NodeID'] for n in ray.nodes() if n['Alive']]
    nodes = []
    for node_id in node_id_list:
        specific_node = Patcher.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        )).remote()
        nodes.append(specific_node)

    print(ray.get([node.prepare_python3_env.remote() for node in nodes]))

    # head_node = [node for node in ray.nodes() if 'drv' in node['NodeName']]
    # assert len(head_node) == 1, f"{len(head_node)} head nodes found, should only be 1 head nodes"
    # head_node_id = head_node[0]['NodeID']
    # head_node = HeadPatcher.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
    #     node_id=head_node_id,
    #     soft=False,
    # )).remote(cuda_version)
    # print(ray.get(head_node.run.remote(cuda_version)))
    # # print(ray.get([node.clean_ray_cluster.remote() for node in nodes]))
    # print(ray.get([node.prepare_python3_env.remote() for node in nodes]))
    # print(ray.get([node.prepare_rl_file.remote() for node in nodes]))
