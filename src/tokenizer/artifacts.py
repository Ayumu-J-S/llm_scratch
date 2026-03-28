from pathlib import Path

from .bpe import BPETokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def load_text(path: str) -> str:
    return (ROOT_DIR / path).read_text(encoding="utf-8")


def load_tokenizer(directory: str, filename: str) -> BPETokenizer:
    tokenizer_path = ROOT_DIR / directory / filename
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            "Tokenizer artifact was not found. "
            f"Expected: {tokenizer_path}. "
            "Run src/train_tokenizer.py first."
        )
    return BPETokenizer.load(str(tokenizer_path))


def save_tokenizer(tokenizer: BPETokenizer, directory: str, filename: str) -> Path:
    tokenizer_path = ROOT_DIR / directory / filename
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer_path
