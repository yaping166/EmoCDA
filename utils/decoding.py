from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

LABEL_IGNORE_INDEX = -100


def _to_numpy(x: Union[np.ndarray, torch.Tensor, Sequence]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def decode_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Union[np.ndarray, torch.Tensor, Sequence],
) -> List[str]:
    arr = _to_numpy(token_ids).astype(np.int64)
    arr = np.where(arr == LABEL_IGNORE_INDEX, tokenizer.pad_token_id, arr)
    arr = np.clip(arr, 0, tokenizer.vocab_size - 1)
    return tokenizer.batch_decode(arr.tolist(), skip_special_tokens=True)


def split_main_aux(predictions) -> Tuple[object, Optional[object]]:
    if isinstance(predictions, (list, tuple)) and len(predictions) >= 2:
        return predictions[0], predictions[1]
    return predictions, None
