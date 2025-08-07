import torch.distributed as dist
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def save_fig_to_numpy(fig):
    fig.canvas.draw()
    rgb_str = fig.canvas.tostring_rgb()
    rgb_arr = np.frombuffer(rgb_str, dtype=np.uint8)
    rgb_arr = rgb_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return rgb_arr


def draw_replica_latency_and_step(gather_metrics):
    replica_latency = []
    replica_step = []
    replica_id = [i for i in range(len(gather_metrics))]
    for metric in gather_metrics:
        replica_latency.append(sum(metric["per_token_latency"]))
        replica_step.append(metric.get("cur_steps", [0])[0])
    if len(replica_latency) == 0 or sum(replica_latency) == 0:
        return None, 0

    long_tail_replica_id = np.array(replica_latency).argmax()

    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(replica_id, replica_latency, color='tab:green')
    plt.xlabel('GPU #')
    plt.ylabel('Latency (ms)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(replica_id, replica_step, color='tab:blue')
    plt.xlabel('GPU #')
    plt.ylabel('Step')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_fig_to_numpy(fig), long_tail_replica_id


def draw_per_step_latency_and_bsz(metrics, bs_key):
    per_token_latency = metrics["per_token_latency"]
    dec_bs = metrics[bs_key]
    if len(per_token_latency) == 0:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    if max(per_token_latency) > 200:
        ax1.set_yscale('log', base=2)
        ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax1.yaxis.get_major_formatter().set_scientific(False)
        ax1.yaxis.get_major_formatter().set_useOffset(False)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.plot(per_token_latency, linewidth=0.8, color=color, label='Per Token Latency')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Batch Size', color=color)
    ax2.plot(dec_bs, linewidth=2.0, color=color, label='Batch Size', alpha=0.9, marker='o', markersize=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=min(dec_bs) * 0.95, top=max(dec_bs) * 1.05)

    plt.title('Per Step Latency and Batch Size Over Steps')

    plt.text(0.02,
             0.98,
             f'Total steps: {len(per_token_latency)}',
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return save_fig_to_numpy(fig)


def draw_kv_cache_utils_and_dec_bsz(metrics):
    kv_cache_utils = metrics["kv_cache_utils"]
    dec_bs = metrics["dec_bs"]
    if len(kv_cache_utils) == 0:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('KV Cache Utilization', color=color)
    ax1.plot(kv_cache_utils, linewidth=0.8, color=color, label='KV Cache Utilization')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Decode Batch Size', color=color)
    ax2.plot(dec_bs, linewidth=2.0, color=color, label='Decode Batch Size', alpha=0.9, marker='o', markersize=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=min(dec_bs) * 0.95, top=max(dec_bs) * 1.05)

    plt.title('KV Cache Utilization and Decode Batch Size Over Steps')

    plt.text(0.02,
             0.98,
             f'Total steps: {len(kv_cache_utils)}',
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return save_fig_to_numpy(fig)


def draw_cumulative_latency_and_dec_bsz(metrics):
    per_token_latency = metrics["per_token_latency"]
    dec_bs = metrics["dec_bs"]
    cumulative_latency = np.cumsum([0] + per_token_latency)
    if len(per_token_latency) == 0:
        return

    fig = plt.figure(figsize=(12, 6))
    plt.plot(cumulative_latency, dec_bs, linewidth=1.5, color='tab:green', alpha=0.8, marker='o', markersize=0.3)
    plt.xlabel('Cumulative Latency (ms)')
    plt.ylabel('Decode Batch Size')
    plt.grid(True, alpha=0.3)

    plt.title('Decode Batch Size vs Cumulative Latency')

    plt.text(0.02,
             0.98,
             f'Total cumulative latency: {cumulative_latency[-1]:.2f}ms',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return save_fig_to_numpy(fig)


def draw_dec_tps_and_dec_bsz(metrics):
    per_token_latency = [1e6] + metrics["per_token_latency"]
    dec_bs = metrics["dec_bs"]
    dec_tps = [1000 * bs / latency for latency, bs in zip(per_token_latency, dec_bs)]
    if len(dec_bs) == 0:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Dec TPS', color=color)
    ax1.plot(dec_tps, linewidth=0.8, color=color, label='Dec TPS')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Decode Batch Size', color=color)
    ax2.plot(dec_bs, linewidth=2.0, color=color, label='Decode Batch Size', alpha=0.9, marker='o', markersize=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=min(dec_bs) * 0.95, top=max(dec_bs) * 1.05)

    plt.title('Decode TPS and Decode Batch Size Over Steps')

    plt.text(0.02,
             0.98,
             f'Total steps: {len(dec_tps)}',
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return save_fig_to_numpy(fig)


def visualize_metrics(metrics):
    # DANGER: inbalanced dp worker can cause nccl timeout
    # gather_metrics = [None for _ in range(dist.get_world_size())]
    # dist.all_gather_object(gather_metrics, metrics)
    # metrics["visualize/replica_latency_and_step"], long_tail_replica_id = draw_replica_latency_and_step(gather_metrics)
    # long_tail_replica_metrics = gather_metrics[long_tail_replica_id]
    long_tail_replica_metrics = metrics
    metrics["visualize/per_step_latency_and_dec_bsz"] = draw_per_step_latency_and_bsz(long_tail_replica_metrics,
                                                                                      bs_key="dec_bs")
    metrics["visualize/per_step_latency_and_tokens_num"] = draw_per_step_latency_and_bsz(long_tail_replica_metrics,
                                                                                         bs_key="tokens_num")
    metrics["visualize/cumulative_latency_and_dec_bsz"] = draw_cumulative_latency_and_dec_bsz(long_tail_replica_metrics)
    metrics["visualize/kv_cache_utils_and_dec_bsz"] = draw_kv_cache_utils_and_dec_bsz(long_tail_replica_metrics)
    metrics["visualize/dec_tps_and_dec_bsz"] = draw_dec_tps_and_dec_bsz(long_tail_replica_metrics)
