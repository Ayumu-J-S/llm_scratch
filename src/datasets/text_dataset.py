import torch


def create_autoregressive_training_data(
    token_ids,
    seq_len,
    bos_token_id,
    eos_token_id,
    device=None,
):
    token_stream = [bos_token_id, *token_ids, eos_token_id]
    if len(token_stream) < seq_len + 1:
        raise ValueError(
            f"Not enough tokenized data for seq_len={seq_len}. "
            f"Need at least {seq_len} tokens, but got {len(token_ids)}."
        )

    inputs = []
    labels = []

    for start in range(len(token_stream) - seq_len):
        chunk = token_stream[start : start + seq_len + 1]
        inputs.append(chunk[:-1])
        labels.append(chunk[1:])

    return (
        torch.tensor(inputs, device=device, dtype=torch.long),
        torch.tensor(labels, device=device, dtype=torch.long),
    )
