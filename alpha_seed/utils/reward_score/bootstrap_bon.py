import torch


def bootstrap_bon_metric(nxm_mat):
    '''n prompt, m samples for each prompt
    return has not been averaged over prompt, metric is averaged over prompt
    '''
    N, M = nxm_mat.shape
    B = 1000
    resample_indices = torch.randint(0, M, (N, B, M), device=nxm_mat.device)
    indices_N = torch.arange(N, device=nxm_mat.device).view(N, 1, 1)
    indices_N = indices_N.expand(-1, B, M)
    sample_resampled = nxm_mat[indices_N, resample_indices]
    bootstrap_maxima, _ = sample_resampled.cummax(dim=2)
    expected_max = bootstrap_maxima.mean(dim=1)
    bon_metric = expected_max.mean(0)
    bo = 1
    metric = {}
    while True:
        metric[f'{bo}'] = bon_metric[bo - 1]
        bo *= 2
        if bo > M:
            break
    return expected_max, metric


def bootstrap_bon_metric_gt(nxm_mat):  # 只适用于 0-1 return，理论计算
    gt = (nxm_mat == 0).float().mean(dim=-1)
    gt = torch.vstack([1 - gt**i for i in range(1, 6)]).transpose(0, 1)
    return gt


if __name__ == "__main__":
    nxm_mat = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0]]).cuda()
    result = bootstrap_bon_metric(nxm_mat=nxm_mat)
    gt = bootstrap_bon_metric_gt(nxm_mat=nxm_mat)

    gt = (nxm_mat == 0).float().mean(dim=-1)
    gt = torch.vstack([1 - gt**i for i in range(1, 6)]).transpose(0, 1)

    nxm_mat2 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 1.0]]).cuda()
    result2 = bootstrap_bon_metric(nxm_mat=nxm_mat2)
    gt2 = bootstrap_bon_metric_gt(nxm_mat=nxm_mat2)

    print(result.mean(0))
    print(result2.mean(0))
    print(gt.mean(0))
    print(gt2.mean(0))  # bo1 相等，bon不同
