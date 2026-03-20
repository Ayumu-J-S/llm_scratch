import torch


def create_seq2seq_training_data(
    token_ids,
    seq_len,
    bos_token_id,
    eos_token_id,
    device=None,
):
    encoder_inputs = []
    decoder_inputs = []
    labels = []

    max_start = len(token_ids) - (2 * seq_len) + 1
    for start in range(max_start):
        source_tokens = token_ids[start : start + seq_len]
        target_tokens = token_ids[start + seq_len : start + (2 * seq_len)]

        encoder_inputs.append(source_tokens)
        decoder_inputs.append([bos_token_id, *target_tokens])
        labels.append([*target_tokens, eos_token_id])

    if not encoder_inputs:
        raise ValueError(
            f"Not enough tokenized data for seq_len={seq_len}. "
            f"Need at least {2 * seq_len} tokens, but got {len(token_ids)}."
        )

    return (
        torch.tensor(encoder_inputs, device=device, dtype=torch.long),
        torch.tensor(decoder_inputs, device=device, dtype=torch.long),
        torch.tensor(labels, device=device, dtype=torch.long),
    )
