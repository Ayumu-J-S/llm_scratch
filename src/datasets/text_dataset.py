import torch


def create_autoregressive_training_data(token_ids, seq_len, device=None):
    if len(token_ids) < seq_len + 1:
        raise ValueError(
            f"Not enough tokenized data for seq_len={seq_len}. "
            f"Need at least {seq_len + 1} tokens, but got {len(token_ids)}."
        )

    inputs = []
    labels = []

    for start in range(len(token_ids) - seq_len):
        inputs.append(token_ids[start : start + seq_len])
        labels.append(token_ids[start + 1 : start + seq_len + 1])

    return (
        torch.tensor(inputs, device=device, dtype=torch.long),
        torch.tensor(labels, device=device, dtype=torch.long),
    )
