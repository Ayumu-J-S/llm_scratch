from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm

from datasets.text_dataset import create_seq2seq_training_data
from models.simple_encoder_decoder_transformer import SimpleEncoderDecoderTransformer
from tokenizer.artifacts import load_text, load_tokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message: str) -> None:
    print(message, flush=True)


def save_checkpoint(model, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model_last.pth")


def log_sample_pair(tokenizer, encoder_inputs, decoder_inputs, labels) -> None:
    sample_index = min(20, len(encoder_inputs) - 1)
    log(f"Training samples: {len(encoder_inputs)}")
    log(f"Example source: {tokenizer.decode(encoder_inputs[sample_index].tolist())!r}")
    log(
        "Example decoder input: "
        f"{tokenizer.decode(decoder_inputs[sample_index].tolist(), skip_special_tokens=False)!r}"
    )
    log(
        "Example label: "
        f"{tokenizer.decode(labels[sample_index].tolist(), skip_special_tokens=False)!r}"
    )


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    log(f"Using device: {DEVICE}")
    log(f"Loading corpus from: {cfg.data.input_path}")
    text = load_text(cfg.data.input_path)

    log("Loading tokenizer artifact...")
    tokenizer = load_tokenizer(
        cfg.artifacts.tokenizers_dir,
        cfg.artifacts.tokenizer_filename,
    )
    log(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    log("Tokenizing corpus...")
    token_ids = tokenizer.encode(text)
    log(f"Tokenized corpus length: {len(token_ids)}")

    log("Building seq2seq training tensors...")
    encoder_inputs, decoder_inputs, labels = create_seq2seq_training_data(
        token_ids=token_ids,
        seq_len=cfg.training.sequence_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        device=DEVICE,
    )
    log_sample_pair(tokenizer, encoder_inputs, decoder_inputs, labels)

    log("Building model...")
    model = SimpleEncoderDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=cfg.model.embed_size,
        num_heads=cfg.model.num_heads,
        max_len=cfg.training.sequence_length + 1,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.to(DEVICE)
    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    checkpoint_dir = ROOT_DIR / cfg.artifacts.checkpoints_dir

    log("Training...")
    for epoch in range(cfg.training.epochs):
        total_loss = 0.0
        num_batches = 0

        for start in tqdm(
            range(0, len(encoder_inputs), cfg.training.batch_size),
            desc=f"epoch {epoch + 1}/{cfg.training.epochs}",
            leave=False,
        ):
            source_batch = encoder_inputs[start : start + cfg.training.batch_size]
            decoder_input_batch = decoder_inputs[start : start + cfg.training.batch_size]
            label_batch = labels[start : start + cfg.training.batch_size]

            optimizer.zero_grad(set_to_none=True)
            logits = model(source_batch, decoder_input_batch)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                label_batch.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        log(f"Epoch {epoch + 1}: loss={average_loss:.6f}")
        save_checkpoint(model, checkpoint_dir)


if __name__ == "__main__":
    main()
