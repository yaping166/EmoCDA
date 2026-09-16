from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import DataCollatorForSeq2Seq


class DualDecoderDataCollator(DataCollatorForSeq2Seq):
    def __call__(
        self,
        features: List[Dict[str, Any]],
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        main_records = [
            {k: v for k, v in feat.items() if k != "aux_labels"}
            for feat in features
        ]
        aux_records = [
            {
                k: v
                for k, v in feat.items()
                if k not in ("labels", "aux_labels")
            }
            | {"labels": feat["aux_labels"]}
            for feat in features
        ]
        main_batch = super().__call__(main_records, return_tensors)
        aux_batch = super().__call__(aux_records, return_tensors)
        main_batch["aux_labels"] = aux_batch["labels"]
        return main_batch
