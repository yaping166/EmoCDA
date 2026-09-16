from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Seq2SeqTrainer


class DualDecoderTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        model._current_epoch = float(self.state.epoch or 0.0)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        model._current_epoch = float(self.state.epoch or 0.0)

        with torch.no_grad():
            loss = model(**inputs).loss
        loss = loss.detach() if loss is not None else None
        if prediction_loss_only:
            return loss, None, None

        main_gen, aux_gen = model.generate_dual(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            max_length=self.args.generation_max_length,
            num_beams=self.args.generation_num_beams,
            gamma_scale=model.gamma_scale(),
        )

        return loss, [main_gen, aux_gen], [inputs.get("labels"), inputs.get("aux_labels")]
