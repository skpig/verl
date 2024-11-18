# ckpt

PATH_DICT = {
    "400m": {
        "sft_legacy": {
            "CN": "hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yufan/400m_moe_sft/p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4/checkpoints/global_epoch_4/p6_to_models/400m.sft27.baseline",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_sft/400m.sft27.baseline",
        },
        "sft_baseline": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_sft/p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf",
        },
        "sft_reflection": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/400m_refl_sft_hf",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_sft/20241107_400m_refl_sft_hf",
        },
        "rm_legacy": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/caizhao/400m_release/rm_p6_moe_400m_0716_sftv27_stage2/checkpoints/global_epoch_1/p6_to_models/rm_p6_moe_400m_baseline",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_rm/rm_p6_moe_400m_baseline"
        },
        "rm_baseline": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/rm_p6_moe_400m_0716_sftv27_stage2_hf",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_rm/rm_p6_moe_400m_0716_sftv27_stage2_hf",
        },
        "prm": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/400m_p60905_137k_revisedonly_scalingexp_5xsample_bsz400_lr5e6_tp4pp2_hf",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_rm/400m_p60905_137k_revisedonly_scalingexp_5xsample_bsz400_lr5e6_tp4pp2_hf",
        }
    },
    "3b3": {
        "sft_legacy": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/3b3.sft27.M.CNEN.reflect.v0_hf_new",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_sft/3b3.sft27.M.CNEN.reflect.v0_hf_new"
        },
        "sft_baseline": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_sft/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf",
        },
        "sft_kd12B128": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/shengdinghu/rl/models/241114_3b3_sft30_12b-kd-bo128_hf",
            "i18n_azure": "hdfs://harunava/home/byte_data_seed_azure/seed_research/user/jiangjiec/exp/p6/3b3_moe/241114_3b3_sft30_12b-kd-bo128_ep5/seed_models_epoch5_hf/241114_3b3_sft30_12b-kd-bo128_hf",
        },
        "rm_baseline": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/rm_p6_moe_3b3_0812_sftv27_stage2_fix_order_aux_32k_v2_hf",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_rm/rm_p6_moe_3b3_0812_sftv27_stage2_fix_order_aux_32k_v2_hf",
        },
        "prm_legacy": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/yueyu/seed_rl/models/rm_p6_moe_3.3b_0716_sftv27_stage2_hf_new_prm",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_rm/rm_p6_moe_3.3b_0716_sftv27_stage2_hf_new_prm"
        },
        "prm": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/3b3p60905_137k_revisedonly_scalingexp_5xsample_bsz1600_lr5e6_tp4pp5_hf",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_rm/3b3p60905_137k_revisedonly_scalingexp_5xsample_bsz1600_lr5e6_tp4pp5_hf",
        }
    },
    "12b": {
        "sft": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_sft/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf",
        },
        "rm": {
            "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/rm_p6_dense_12b_phase2_exp1_hf",
            "i18n_OCI": "hdfs://harunava/home/byte_data_seed_us/hdd_va/user/shengdinghu.98/models_rm/rm_p6_dense_12b_phase2_exp1_hf",
        }
    }

}

def get_model(varname, size, type, platform):
    path=PATH_DICT[size][type][platform]
    if not path:
        raise KeyError
    print(f"export {varname}={path}" )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--varname", type=str)
    parser.add_argument("--size", type=str)
    parser.add_argument("--type", type=str)
    parser.add_argument("--platform", type=str)
    args = parser.parse_args()
    get_model(args.varname, args.size, args.type, args.platform)


