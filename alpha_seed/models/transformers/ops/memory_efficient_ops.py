import torch


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def compute_chunked_entropy_logprobs(self, hidden_states, labels=None, temperature=None, compute_entropy=False):
    entropy_chunks = []
    log_probs_chunks = []
    entropy_rmpad = None
    if labels is not None:
        chunk_size = 2048
        batch_size, seq_len, hidden_dim = hidden_states.shape
        assert batch_size == 1

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            hidden_chunk = hidden_states[:, start:end, :]
            labels_chunk = labels[start:end]
            logits_chunk = self.lm_head(hidden_chunk).float()  # [1, chunk_len, vocab_size]
            if temperature is not None:
                logits_chunk.div_(temperature)
            logits_chunk = logits_chunk.squeeze(0)  # [chunk_len, vocab_size]

            chunk_len = end - start
            if chunk_len == 0:
                continue

            if compute_entropy:
                entropy_chunk = entropy_from_logits(logits_chunk)
                entropy_chunks.append(entropy_chunk)

            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
            chunk_log_probs = -cross_entropy_loss(logits_chunk, labels_chunk,
                                                  inplace_backward=not compute_entropy)[0]  # (total_nnz,)
            log_probs_chunks.append(chunk_log_probs)
            torch.cuda.empty_cache()

        entropy_rmpad = torch.cat(entropy_chunks, dim=0) if entropy_chunks else None
        log_probs = torch.cat(log_probs_chunks, dim=0) if log_probs_chunks else None

    return entropy_rmpad, log_probs


def compute_chunked_mtp_acceptance_ratio_per_head(self, hidden_states, labels):
    chunk_size = 2048
    batch_size, seq_len, hidden_dim = hidden_states.shape
    assert batch_size == 1

    acceptance_chunks = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        hidden_chunk = hidden_states[:, start:end, :]
        labels_chunk = labels[start:end]
        logits_chunk = self.lm_head(hidden_chunk)  # [1, chunk_len, vocab_size]
        logits_chunk = logits_chunk.squeeze(0)  # [chunk_len, vocab_size]

        # take argmax
        predicted_labels_chunked = torch.argmax(logits_chunk, dim=-1)
        # compute acceptance ratio
        acceptance = predicted_labels_chunked == labels_chunk
        acceptance_chunks.append(acceptance)

        chunk_len = end - start
        if chunk_len == 0:
            continue

        torch.cuda.empty_cache()

    acceptance_chunks = torch.cat(acceptance_chunks, dim=0)
    return acceptance_chunks


def compute_chunked_mtp_acceptance_ratio(self, all_mtp_hidden_states, all_mtp_labels):
    acceptance_data = []
    for mtp_hidden_states, labels in zip(all_mtp_hidden_states, all_mtp_labels):
        acceptance_chunks = compute_chunked_mtp_acceptance_ratio_per_head(self, mtp_hidden_states, labels)
        acceptance_data.append(acceptance_chunks)
    return tuple(acceptance_data)
