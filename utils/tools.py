import torch

from .cleaners import Cleaner
from .tokenizer import Tokenizer


def prepare_text(text: str) -> torch.Tensor:
    if not ((text[-1] == '.') or (text[-1] == '?') or (text[-1] == '!')):
        text = text + '.'
    cleaner = Cleaner('english_cleaners', True, 'en-us')
    tokenizer = Tokenizer()
    return torch.as_tensor(tokenizer(cleaner(text)), dtype=torch.int, device='cpu').unsqueeze(0)
