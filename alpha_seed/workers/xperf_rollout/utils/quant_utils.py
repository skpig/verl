import torch


def round_to_int8(tensor):
    return torch.floor(tensor + 0.5)


def quant_gemm_weight_w8a8(weight, input_amax, ratio=0.5):
    # use fixed threshold to quant weight to W8A8
    # for dense gemm only
    n, k = weight.shape
    dtype = weight.dtype
    if input_amax is not None:
        k_, = input_amax.shape
        assert k == k_, f"weight shape {weight.shape} and input_amax shape {input_amax.shape} not match"

        search_min, search_max = input_amax.min(dim=-1)[0], input_amax.max(dim=-1)[0]
        scale_bound = search_min + (search_max - search_min) * ratio
        scale_bound = scale_bound.clamp(min=1e-5)
        smooth_quant_scale = input_amax / scale_bound.unsqueeze(-1)
        smooth_quant_scale = smooth_quant_scale.clamp(min=1.0)
        smooth_quant_scale = smooth_quant_scale.to(dtype)
        smooth_quant_scale = smooth_quant_scale.unsqueeze(0)
    else:
        smooth_quant_scale = torch.ones([1, k], dtype=dtype, device=weight.device)

    weight_smoothed = weight * smooth_quant_scale
    i8_weight_qscale = torch.max(weight_smoothed.abs(), dim=1, keepdim=True)[0] / 127.0
    i8_weight = round_to_int8(weight_smoothed / i8_weight_qscale).clamp(-127.0, 127.0)

    smooth_quant_scale = smooth_quant_scale.squeeze(0)
    i8_weight = i8_weight.to(torch.int8)
    i8_weight_qscale = i8_weight_qscale.to(torch.float32)
    i8_weight_qscale = i8_weight_qscale.squeeze(1)
    return smooth_quant_scale, i8_weight, i8_weight_qscale


def quant_group_gemm_weight_w4a8(weight, input_amax, ratio=0.5):
    # use fixed threshold to quant weight to W4A8
    # for group gemm only
    e, n, k = weight.shape
    dtype = weight.dtype
    if input_amax is not None:
        if input_amax.dim() == 1:
            input_amax = input_amax.unsqueeze(0)
        e_, k_ = input_amax.shape
        assert e == e_ and k == k_, f"weight shape {weight.shape} and input_amax shape {input_amax.shape} not match"

        search_min, search_max = input_amax.min(dim=-1)[0], input_amax.max(dim=-1)[0]
        scale_bound = search_min + (search_max - search_min) * ratio
        scale_bound = scale_bound.clamp(min=1e-5)
        smooth_quant_scale = input_amax / scale_bound.unsqueeze(-1)
        smooth_quant_scale = smooth_quant_scale.clamp(min=1.0)
        smooth_quant_scale = smooth_quant_scale.to(dtype)
        smooth_quant_scale = smooth_quant_scale.unsqueeze(1)
    else:
        smooth_quant_scale = torch.ones([e, 1, k], dtype=dtype, device=weight.device)

    weight_smoothed = weight * smooth_quant_scale
    i8_weight_qscale = torch.max(weight_smoothed.abs(), dim=-1, keepdim=True)[0] / 119.0
    i8_weight = round_to_int8(weight_smoothed / i8_weight_qscale).clamp(-119.0, 119.0)

    group_size = 64
    i8_weight = i8_weight.reshape(-1, group_size)
    i4_max = torch.max(i8_weight, dim=-1, keepdim=True)[0]
    i4_min = torch.min(i8_weight, dim=-1, keepdim=True)[0]
    i4_weight_qscale = round_to_int8((i4_max - i4_min) / 15.0).clamp(min=1)
    i4_weight_qzero = round_to_int8(-i4_min / i4_weight_qscale)
    i4_weight = round_to_int8(i8_weight / i4_weight_qscale + i4_weight_qzero).clamp(0, 15)

    smooth_quant_scale = smooth_quant_scale.squeeze(1)
    i4_weight = i4_weight.to(torch.int8).reshape([e, n, k])
    i8_weight_qscale = i8_weight_qscale.to(torch.float32)
    i8_weight_qscale = i8_weight_qscale.squeeze(2)
    i4_weight_qscale = i4_weight_qscale.to(torch.int8).reshape([e, n, k // group_size])
    i4_weight_qzero = i4_weight_qzero.to(torch.int8).reshape([e, n, k // group_size])
    return smooth_quant_scale, i4_weight, i8_weight_qscale, i4_weight_qscale, i4_weight_qzero


if __name__ == '__main__':
    # unit test
    def test_quant_gemm_weight_w8a8(n, k):
        weight = torch.rand([n, k], dtype=torch.bfloat16).cuda()
        input_amax = torch.ones([k], dtype=torch.bfloat16).cuda()
        smooth_quant_scale, i8_weight, i8_weight_qscale = quant_gemm_weight_w8a8(weight, input_amax)
        print(smooth_quant_scale.shape, i8_weight.shape, i8_weight_qscale.shape)
        assert i8_weight.dtype == torch.int8
        assert i8_weight_qscale.dtype == torch.float32
        weight_restored = (i8_weight.to(torch.float32) * i8_weight_qscale[:, None]).to(torch.bfloat16)
        diff = (weight - weight_restored).abs()
        max_diff_pos = torch.argmax(diff.flatten())
        print(f"max diff: {diff.max().item()}, "
              f"{weight.flatten()[max_diff_pos].item()} vs {weight_restored.flatten()[max_diff_pos].item()}, "
              f"avg diff: {diff.mean().item()}")

    def test_quant_group_gemm_weight_w4a8(e, n, k):
        weight = torch.rand([e, n, k], dtype=torch.bfloat16).cuda()
        input_amax = torch.ones([e, k], dtype=torch.bfloat16).cuda()
        smooth_quant_scale, i4_weight, i8_weight_qscale, i4_weight_qscale, i4_weight_qzero = \
            quant_group_gemm_weight_w4a8(weight, input_amax)
        print(smooth_quant_scale.shape, i4_weight.shape, i8_weight_qscale.shape, i4_weight_qscale.shape,
              i4_weight_qzero.shape)
        assert i4_weight.dtype == torch.int8
        assert i4_weight.min().item() == 0 and i4_weight.max().item() == 15
        assert i8_weight_qscale.dtype == torch.float32
        assert i4_weight_qscale.dtype == torch.int8
        assert i4_weight_qzero.dtype == torch.int8
        i4_weight = i4_weight.reshape([e, n, -1, 64])
        i4_weight_qzero = i4_weight_qzero.unsqueeze(-1)
        i4_weight_qscale = i4_weight_qscale.unsqueeze(-1)
        weight_restored_i8 = (i4_weight.to(torch.float32) - i4_weight_qzero.to(torch.float32)) * \
                             i4_weight_qscale.to(torch.float32)
        weight_restored_i8 = weight_restored_i8.reshape([e, n, k])
        weight_restored = (weight_restored_i8 * i8_weight_qscale[:, :, None]).to(torch.bfloat16)
        diff = (weight - weight_restored).abs()
        max_diff_pos = torch.argmax(diff.flatten())
        print(f"max diff: {diff.max().item()}, "
              f"{weight.flatten()[max_diff_pos].item()} vs {weight_restored.flatten()[max_diff_pos].item()}, "
              f"avg diff: {diff.mean().item()}")

    test_quant_gemm_weight_w8a8(1024, 2048)
    test_quant_group_gemm_weight_w4a8(128, 1024, 2048)
