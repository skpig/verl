import torch
import torch.distributed as dist
import torch.nn.functional as F


class AllToAllEmbedding(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group: dist.ProcessGroup, input_tensor: torch.Tensor, embedding_table: torch.Tensor):
        # --- Get distributed info ---
        tp_size = dist.get_world_size(group) if group else 1
        tp_rank = dist.get_rank(group) if group else 0

        vocab_size_per_rank = embedding_table.shape[0]
        start_id = tp_rank * vocab_size_per_rank

        _raw_shape = input_tensor.shape
        input_flat = input_tensor.reshape(-1)
        num_input_ids = input_flat.shape[0]

        # --- Dispatching logic ---
        id_rank = torch.clamp_max(input_flat // vocab_size_per_rank, tp_size - 1)
        index = torch.arange(num_input_ids, device=input_flat.device)
        rank_index = [index[id_rank == i] for i in range(tp_size)]
        send_rank_count_list = [len(ri) for ri in rank_index]

        # --- 1st Communication: Exchange sizes ---
        send_rank_count = torch.tensor(send_rank_count_list, device=input_flat.device)
        all_send_rank_count = [torch.zeros_like(send_rank_count) for _ in range(tp_size)]
        if group:
            dist.all_gather(all_send_rank_count, send_rank_count, group=group)
        all_send_rank_count = torch.stack(all_send_rank_count, dim=0)

        receive_rank_count_list = all_send_rank_count[:, tp_rank].tolist()

        # --- 2nd Communication: Exchange token IDs ---
        all_receive_ids = [
            torch.zeros([count], dtype=input_flat.dtype, device=input_flat.device) for count in receive_rank_count_list
        ]
        if group:
            dist.all_to_all(all_receive_ids, [input_flat[ri] for ri in rank_index], group=group)

        # --- Local embedding lookup ---
        concatenated_ids = torch.cat(all_receive_ids, dim=0)
        local_indices = concatenated_ids - start_id
        embs = torch.nn.functional.embedding(local_indices, embedding_table)

        # --- 3rd Communication: Send computed embeddings back ---
        embs_send_back = list(embs.split(receive_rank_count_list, dim=0))
        embs_receive_back = [
            torch.zeros([count, embedding_table.shape[1]], dtype=embs.dtype, device=embs.device)
            for count in send_rank_count_list
        ]
        if group:
            dist.all_to_all(embs_receive_back, embs_send_back, group=group)

        # --- Reassemble output to original order ---
        full_rank_index = torch.cat(rank_index, dim=0)
        embs_received_cat = torch.cat(embs_receive_back, dim=0)
        output = torch.zeros([num_input_ids, embedding_table.shape[1]], dtype=embs.dtype, device=embs.device)
        output[full_rank_index] = embs_received_cat
        output = output.view(*_raw_shape, embedding_table.shape[1])

        # --- Save for backward ---
        ctx.save_for_backward(local_indices, full_rank_index)
        ctx.group = group
        ctx.embedding_table_shape = embedding_table.shape
        ctx.receive_rank_count_list = receive_rank_count_list
        ctx.send_rank_count_list = send_rank_count_list

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # --- Retrieve from context ---
        local_indices, full_rank_index = ctx.saved_tensors
        group = ctx.group
        embedding_table_shape = ctx.embedding_table_shape
        receive_rank_count_list = ctx.receive_rank_count_list
        send_rank_count_list = ctx.send_rank_count_list

        # --- Prepare gradients for communication ---
        grad_output_flat = grad_output.reshape(-1, embedding_table_shape[1])
        grad_to_send_back_flat = grad_output_flat[full_rank_index]
        grad_send_list = list(grad_to_send_back_flat.split(send_rank_count_list, dim=0))

        grad_receive_list = [
            torch.zeros([count, embedding_table_shape[1]], dtype=grad_output.dtype, device=grad_output.device)
            for count in receive_rank_count_list
        ]

        # --- Communicate gradients back ---
        if group:
            dist.all_to_all(grad_receive_list, grad_send_list, group=group)

        grad_received_cat = torch.cat(grad_receive_list, dim=0)

        # --- Calculate gradient for the embedding table ---
        grad_embedding_table = torch.zeros(embedding_table_shape, device=grad_output.device, dtype=grad_output.dtype)
        grad_embedding_table.index_add_(0, local_indices, grad_received_cat)

        # Return gradients for the inputs of forward:
        # (group, input_tensor, embedding_table)
        return None, None, grad_embedding_table


class AllGatherReduceScatterEmbedding(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group: dist.ProcessGroup, local_input: torch.Tensor, embedding_table: torch.Tensor):
        # Save original shape and flatten the input tensor to 1D
        _raw_shape = local_input.shape
        input_flat = local_input.view(-1)

        # Get world_size and rank
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        # 1. Handle variable input lengths via padding (on the flattened tensor)
        # Get local input length and gather all lengths from all ranks
        local_len = torch.tensor([input_flat.numel()], device=input_flat.device, dtype=torch.long)
        all_lengths = torch.empty(world_size, dtype=torch.long, device=input_flat.device)
        dist.all_gather_into_tensor(all_lengths, local_len, group=group)

        # Find the max length and pad the local input to it
        max_len = all_lengths.max().item()
        pad_len = max_len - local_len.item()

        # Padding now works correctly on the 1D tensor
        padded_input = F.pad(input_flat, (0, pad_len), 'constant', -1)

        # 2. All-Gather all padded inputs
        # This call is now correct because `padded_input` is 1D with `max_len` elements.
        all_padded_inputs = torch.empty(world_size, max_len, dtype=padded_input.dtype, device=padded_input.device)
        dist.all_gather_into_tensor(all_padded_inputs, padded_input, group=group)

        # Flatten into a single global 1D vector
        all_inputs_flat = all_padded_inputs.view(-1)

        # 3. Local Embedding Lookup
        vocab_size_per_rank, embedding_dim = embedding_table.shape
        start_id = rank * vocab_size_per_rank
        end_id = (rank + 1) * vocab_size_per_rank

        # Find tokens in the global input that this rank is responsible for
        mask = (all_inputs_flat >= start_id) & (all_inputs_flat < end_id)
        indices_in_all = torch.nonzero(mask, as_tuple=True)[0]
        tokens_for_local_lookup = all_inputs_flat[indices_in_all]

        # Convert global IDs to local indices
        local_indices = tokens_for_local_lookup - start_id

        # Perform the local embedding lookup
        local_embeddings = F.embedding(local_indices, embedding_table)

        # 4. Prepare the input tensor for Reduce-Scatter
        scatter_tensor = torch.zeros(world_size * max_len,
                                     embedding_dim,
                                     dtype=embedding_table.dtype,
                                     device=embedding_table.device)
        scatter_tensor.index_add_(0, indices_in_all, local_embeddings)
        scatter_list_tensor = scatter_tensor.view(world_size, max_len, embedding_dim)

        # 5. Perform Reduce-Scatter
        output_padded = torch.empty(max_len, embedding_dim, dtype=embedding_table.dtype, device=embedding_table.device)
        dist.reduce_scatter_tensor(output_padded, scatter_list_tensor, group=group)

        # 6. Unpad by slicing
        final_output_flat = output_padded[:local_len.item()]

        # Reshape output to its original multi-dimensional shape
        output = final_output_flat.view(*_raw_shape, embedding_dim)

        # 7. Save tensors for backward pass
        ctx.save_for_backward(all_lengths, indices_in_all, local_indices)
        ctx.embedding_shape = embedding_table.shape
        ctx.group = group
        ctx.raw_input_shape = _raw_shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 1. Restore saved tensors and context
        all_lengths, indices_in_all, local_indices = ctx.saved_tensors
        embedding_shape = ctx.embedding_shape
        group = ctx.group
        # _raw_shape = ctx.raw_input_shape

        # Flatten the incoming gradient
        embedding_dim = grad_output.shape[-1]
        grad_output_flat = grad_output.reshape(-1, embedding_dim)

        # 2. Pad the flattened gradient
        local_len = grad_output_flat.shape[0]
        max_len = all_lengths.max().item()
        pad_len = max_len - local_len
        padded_grad = F.pad(grad_output_flat, (0, 0, 0, pad_len), 'constant', 0)

        # 3. Reverse Reduce-Scatter (which is All-Gather)
        world_size = dist.get_world_size(group)

        gathered_grads = torch.empty(world_size,
                                     max_len,
                                     embedding_dim,
                                     dtype=padded_grad.dtype,
                                     device=padded_grad.device)
        dist.all_gather_into_tensor(gathered_grads, padded_grad, group=group)
        flat_gathered_grads = gathered_grads.view(-1, embedding_dim)

        # 4. Select gradients relevant to the local embeddings
        grad_for_local_embedding = flat_gathered_grads.index_select(0, indices_in_all)

        # 5. Calculate the gradient for the Embedding Table
        grad_embedding_table = torch.zeros(embedding_shape, dtype=grad_output.dtype, device=grad_output.device)
        grad_embedding_table.index_add_(0, local_indices, grad_for_local_embedding)

        # 6. Return gradients for the inputs of forward
        # (group, local_input, embedding_table)
        return None, None, grad_embedding_table
