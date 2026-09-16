from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import transformers
from transformers import PreTrainedTokenizerBase, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from utils.metrics import fold_category
from model.cross_decoder_attention import (
    CrossDecoderAttention,
    colon_token_ids,
)

logger = logging.getLogger(__name__)

_KEY_CACHE_MAX = 100_000

_MARKER_FILE = "shared_encoder.json"
_MAIN_SUBDIR = "main_model"
_AUX_SUBDIR = "aux_model"
_CROSS_MAIN_FILE = "cross_decoder_main.pt"
_CROSS_AUX_FILE = "cross_decoder_aux.pt"
_CROSS_LEGACY_FILE = "cross_decoder.pt"

class DualDecoderMTLModel(transformers.PreTrainedModel):
    def __init__(
        self,
        main_model: T5ForConditionalGeneration,
        aux_model: T5ForConditionalGeneration,
        cross_decoder_main: Optional[CrossDecoderAttention] = None,
        cross_decoder_aux: Optional[CrossDecoderAttention] = None,
        separator_ids: Optional[List[int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        align_granularity: str = "EA",
        warmup_epochs: float = 5.0,
    ):
        super().__init__(main_model.config)
        self.main_model = main_model
        self.aux_model = aux_model
        self.submodels = [self.main_model, self.aux_model]
        self.config = main_model.config
        self.cross_decoder_main = cross_decoder_main
        self.cross_decoder_aux = cross_decoder_aux
        self._separator_ids = list(separator_ids or [])
        self._tokenizer = tokenizer
        self._align_granularity = align_granularity
        self.warmup_epochs = float(warmup_epochs)
        self._current_epoch = 0.0
        self._gamma_override: Optional[float] = None
        self._colon_ids = set(colon_token_ids(tokenizer))
        self._key_cache: Dict[Tuple[int, ...], str] = {}

    def set_gamma_override(self, scale: Optional[float]) -> None:
        self._gamma_override = None if scale is None else float(scale)

    def effective_gamma(self) -> float:
        target = self.cross_decoder_main.gamma
        if self._gamma_override is not None:
            return target * self._gamma_override
        epoch = self._current_epoch
        if epoch < self.warmup_epochs:
            return 0.0
        progress = epoch - self.warmup_epochs
        return target * min(1.0, max(0.0, progress))

    def gamma_scale(self) -> float:
        target = self.cross_decoder_main.gamma
        if target <= 0.0:
            return 0.0
        return self.effective_gamma() / target

    @contextlib.contextmanager
    def _submodels_eval(self):
        modules = [
            *self.submodels,
            self.cross_decoder_main,
            self.cross_decoder_aux,
        ]
        modes = [(module, module.training) for module in modules]
        try:
            for module, _ in modes:
                module.eval()
            yield
        finally:
            for module, was_training in modes:
                module.train(was_training)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        model_name: str,
        *,
        gamma: float = 0.1,
        separator_ids: Optional[List[int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        align_granularity: str = "EA",
        warmup_epochs: float = 3.0,
    ) -> DualDecoderMTLModel:
        logger.info("Loading %s (sentiment decoder)...", model_name)
        main_model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info("Loading %s (emotion decoder)...", model_name)
        aux_model = T5ForConditionalGeneration.from_pretrained(model_name)
        return cls._tie_shared_encoder(
            main_model,
            aux_model,
            gamma=gamma,
            separator_ids=separator_ids,
            tokenizer=tokenizer,
            align_granularity=align_granularity,
            warmup_epochs=warmup_epochs,
        )

    @classmethod
    def _tie_shared_encoder(
        cls,
        main_model,
        aux_model,
        *,
        gamma: float = 0.1,
        separator_ids: Optional[List[int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        align_granularity: str = "EA",
        warmup_epochs: float = 5.0,
    ):
        aux_model.encoder = main_model.get_encoder()
        logger.info("Shared encoder tied to both decoders.")
        module_main = CrossDecoderAttention(
            d_model=main_model.config.d_model,
            n_heads=main_model.config.num_heads,
            gamma=gamma,
        )
        module_aux = CrossDecoderAttention(
            d_model=main_model.config.d_model,
            n_heads=main_model.config.num_heads,
            gamma=gamma,
        )
        return cls(
            main_model,
            aux_model,
            cross_decoder_main=module_main,
            cross_decoder_aux=module_aux,
            separator_ids=separator_ids,
            tokenizer=tokenizer,
            align_granularity=align_granularity,
            warmup_epochs=warmup_epochs,
        )

    # ------------------------------------------------------------------ #
    # Side-generic
    # ------------------------------------------------------------------ #
    def _model_for(self, side: str) -> T5ForConditionalGeneration:
        if side == "main":
            return self.main_model
        if side == "aux":
            return self.aux_model
        raise ValueError(f"unknown side {side!r}")

    def _cross_decoder_for(self, side: str) -> Optional[CrossDecoderAttention]:
        if side == "main":
            return self.cross_decoder_main
        if side == "aux":
            return self.cross_decoder_aux
        raise ValueError(f"unknown side {side!r}")

    def _project(self, side: str, hidden: torch.Tensor) -> torch.Tensor:
        model = self._model_for(side)
        if model.config.tie_word_embeddings:
            hidden = hidden * (model.model_dim ** -0.5)
        return model.lm_head(hidden)

    def _hidden_from_decoder_ids(
        self, side: str, attention_mask, encoder_outputs, decoder_input_ids
    ) -> torch.Tensor:
        model = self._model_for(side)
        outputs = model(
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.decoder_hidden_states[-1]

    # ------------------------------------------------------------------ #
    # Category alignment
    # ------------------------------------------------------------------ #
    def _segment_ids(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        sep = torch.zeros_like(decoder_input_ids, dtype=torch.bool)
        for sid in self._separator_ids:
            sep |= decoder_input_ids == sid
        seg = torch.cumsum(sep.long(), dim=1) - sep.long()
        pad = decoder_input_ids == self.config.pad_token_id
        return seg.masked_fill(pad, -1)

    def _normalize_category_key(self, raw: str) -> str:
        if not raw:
            return ""
        collapsed = " ".join(raw.strip().split())
        return fold_category(collapsed.upper(), self._align_granularity).lower()

    def _category_key(self, ids: List[int]) -> str:
        cache_key = tuple(ids)
        cached = self._key_cache.get(cache_key)
        if cached is not None:
            return cached
        key = self._normalize_category_key(
            self._tokenizer.decode(ids, skip_special_tokens=True)
        )
        if len(self._key_cache) < _KEY_CACHE_MAX:
            self._key_cache[cache_key] = key
        return key

    def _clause_category_texts(
        self, ids: torch.Tensor, seg: torch.Tensor
    ) -> List[List[str]]:
        out: List[List[str]] = []
        for b in range(ids.size(0)):
            seg_b, ids_b = seg[b], ids[b]
            if not (seg_b >= 0).any():
                out.append([])
                continue
            texts: List[str] = []
            for s in range(int(seg_b.max()) + 1):
                pos = (seg_b == s).nonzero(as_tuple=True)[0]
                if not pos.numel():
                    texts.append("")
                    continue
                start = int(pos[0])
                end = int(pos[-1]) + 1
                for p in pos.tolist():
                    if int(ids_b[p]) in self._colon_ids:
                        end = p
                        break
                texts.append(self._category_key(ids_b[start:end].tolist()))
            out.append(texts)
        return out

    @staticmethod
    def _assign_align_ids(
        clause_seg: torch.Tensor, texts: List[str], key_to_id: Dict[str, int]
    ) -> torch.Tensor:
        align = torch.full_like(clause_seg, -1)
        for s, text in enumerate(texts):
            if not text:
                continue
            key_to_id.setdefault(text, len(key_to_id))
            align[clause_seg == s] = key_to_id[text]
        return align

    def _category_align_ids(
        self, a_ids: torch.Tensor, b_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_seg = self._segment_ids(a_ids)
        b_seg = self._segment_ids(b_ids)
        a_texts = self._clause_category_texts(a_ids, a_seg)
        b_texts = self._clause_category_texts(b_ids, b_seg)
        a_align = torch.full_like(a_seg, -1)
        b_align = torch.full_like(b_seg, -1)

        for b in range(a_ids.size(0)):
            key_to_id: Dict[str, int] = {}
            a_align[b] = self._assign_align_ids(a_seg[b], a_texts[b], key_to_id)
            b_align[b] = self._assign_align_ids(b_seg[b], b_texts[b], key_to_id)

        return a_align, b_align

    def _align_for_step(
        self,
        self_ids: torch.Tensor,
        ctx_seg: torch.Tensor,
        ctx_texts: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self_seg = self._segment_ids(self_ids)
        self_texts = self._clause_category_texts(self_ids, self_seg)
        self_align = torch.full_like(self_seg, -1)
        ctx_align = torch.full_like(ctx_seg, -1)
        for b in range(self_ids.size(0)):
            key_to_id: Dict[str, int] = {}
            self_align[b] = self._assign_align_ids(self_seg[b], self_texts[b], key_to_id)
            ctx_align[b] = self._assign_align_ids(ctx_seg[b], ctx_texts[b], key_to_id)
        return self_align, ctx_align

    @staticmethod
    def _lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    @staticmethod
    def _align_fused_logits_to_labels(
        fused_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        label_len = labels.size(1)
        seq_len = fused_logits.size(1)
        if seq_len == label_len:
            return fused_logits
        if seq_len > label_len:
            return fused_logits[:, :label_len, :]
        pad_len = label_len - seq_len
        pad = fused_logits.new_zeros(
            fused_logits.size(0), pad_len, fused_logits.size(-1)
        )
        return torch.cat([fused_logits, pad], dim=1)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs=None,
        labels: Optional[torch.LongTensor] = None,
        aux_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.main_model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return self._forward_cross(
            attention_mask,
            decoder_input_ids,
            encoder_outputs,
            labels,
            aux_labels,
        )

    def _forward_cross(
        self,
        attention_mask,
        decoder_input_ids,
        encoder_outputs,
        labels,
        aux_labels,
    ) -> Seq2SeqLMOutput:
        gamma_scale = self.gamma_scale()

        if gamma_scale <= 0.0:
            aux_dec_in = self.aux_model.prepare_decoder_input_ids_from_labels(aux_labels)
            aux_outputs = self.aux_model(
                attention_mask=attention_mask,
                decoder_input_ids=aux_dec_in,
                encoder_outputs=encoder_outputs,
                return_dict=True,
            )
            aux_loss = self._lm_loss(aux_outputs.logits, aux_labels)

            if decoder_input_ids is None:
                decoder_input_ids = self.main_model.prepare_decoder_input_ids_from_labels(
                    labels
                )
            main_outputs = self.main_model(
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                return_dict=True,
            )
            main_loss = self._lm_loss(main_outputs.logits, labels)

            loss = 0.5 * main_loss + 0.5 * aux_loss
            return Seq2SeqLMOutput(
                loss=loss,
                logits=main_outputs.logits,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            )

        max_length = max(labels.size(1), aux_labels.size(1)) + 1
        enc_wrap = BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state
        )

        with torch.no_grad(), self._submodels_eval():
            aux_gen0 = self.aux_model.generate(
                encoder_outputs=enc_wrap,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=1,
            )
            aux_hidden0 = self._hidden_from_decoder_ids(
                "aux", attention_mask, encoder_outputs, aux_gen0
            ).detach()
            main_gen0 = self.main_model.generate(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=encoder_outputs.last_hidden_state
                ),
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=1,
            )
            main_hidden0 = self._hidden_from_decoder_ids(
                "main", attention_mask, encoder_outputs, main_gen0
            ).detach()

        main_loss, main_logits = self._forward_cross_gold_fused(
            "main",
            attention_mask,
            encoder_outputs,
            labels,
            ctx_gen=aux_gen0,
            ctx_hidden=aux_hidden0,
            gamma_scale=gamma_scale,
        )

        aux_loss, _aux_logits = self._forward_cross_gold_fused(
            "aux",
            attention_mask,
            encoder_outputs,
            aux_labels,
            ctx_gen=main_gen0,
            ctx_hidden=main_hidden0,
            gamma_scale=gamma_scale,
        )

        loss = 0.5 * main_loss + 0.5 * aux_loss
        return Seq2SeqLMOutput(
            loss=loss,
            logits=main_logits,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def _forward_cross_gold_fused(
        self,
        side: str,
        attention_mask,
        encoder_outputs,
        target_labels,
        *,
        ctx_gen: torch.Tensor,
        ctx_hidden: torch.Tensor,
        gamma_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self._model_for(side)
        dec_in = model.prepare_decoder_input_ids_from_labels(target_labels)
        outputs = model(
            attention_mask=attention_mask,
            decoder_input_ids=dec_in,
            encoder_outputs=encoder_outputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.decoder_hidden_states[-1]

        self_align, ctx_align = self._category_align_ids(dec_in, ctx_gen)
        cross_decoder = self._cross_decoder_for(side)
        fused = cross_decoder(
            hidden,
            ctx_hidden,
            self_align,
            ctx_align,
            gamma_scale=gamma_scale,
        )
        fused_logits = self._project(side, fused)
        fused_logits = self._align_fused_logits_to_labels(fused_logits, target_labels)
        loss = self._lm_loss(fused_logits, target_labels)
        return loss, fused_logits

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, *args, **kwargs):
        return self.main_model.generate(*args, **kwargs)

    @contextlib.contextmanager
    def _fused_lm_head(
        self,
        side: str,
        ctx_hidden: torch.Tensor,
        ctx_seg: torch.Tensor,
        ctx_texts: List[List[str]],
        *,
        gamma_scale: float,
    ):
        model = self._model_for(side)
        cross_decoder = self._cross_decoder_for(side)

        state: Dict[str, Optional[torch.Tensor]] = {"decoder_ids": None}
        orig_prepare = model.prepare_inputs_for_generation
        orig_forward = model.forward

        @functools.wraps(orig_prepare)
        def prepare_inputs_for_generation(input_ids, *args, **kwargs):
            state["decoder_ids"] = input_ids
            return orig_prepare(input_ids, *args, **kwargs)

        @functools.wraps(orig_forward)
        def forward(*args, **kwargs):
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True
            outputs = orig_forward(*args, **kwargs)
            hidden = outputs.decoder_hidden_states[-1]
            decoder_ids = state["decoder_ids"]
            if decoder_ids is None:
                decoder_ids = kwargs.get("decoder_input_ids")
            self_align, ctx_align = self._align_for_step(
                decoder_ids, ctx_seg, ctx_texts
            )
            fused = cross_decoder(
                hidden,
                ctx_hidden,
                self_align[:, -hidden.size(1):],
                ctx_align,
                gamma_scale=gamma_scale,
            )
            outputs.logits = self._project(side, fused)
            return outputs

        model.prepare_inputs_for_generation = prepare_inputs_for_generation
        model.forward = forward
        try:
            yield
        finally:
            del model.prepare_inputs_for_generation
            del model.forward

    @torch.no_grad()
    def _cross_generate(
        self,
        side: str,
        encoder_outputs,
        attention_mask: torch.Tensor,
        ctx_hidden: torch.Tensor,
        ctx_seg: torch.Tensor,
        ctx_texts: List[List[str]],
        *,
        max_length: int,
        num_beams: int,
        gamma_scale: float,
    ) -> torch.Tensor:
        if num_beams > 1:
            ctx_hidden = ctx_hidden.repeat_interleave(num_beams, dim=0)
            ctx_seg = ctx_seg.repeat_interleave(num_beams, dim=0)
            ctx_texts = [row for row in ctx_texts for _ in range(num_beams)]

        model = self._model_for(side)
        with self._submodels_eval(), self._fused_lm_head(
            side, ctx_hidden, ctx_seg, ctx_texts, gamma_scale=gamma_scale
        ):
            return model.generate(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=encoder_outputs.last_hidden_state
                ),
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
            )

    @torch.no_grad()
    def generate_dual(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        gamma_scale: Optional[float] = None,
        **kwargs,
    ):
        gen = dict(max_length=max_length, num_beams=num_beams)
        with self._submodels_eval():
            encoder_outputs = self.main_model.get_encoder()(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )

            def _enc_wrap():
                return BaseModelOutput(
                    last_hidden_state=encoder_outputs.last_hidden_state
                )

            gamma_scale = (
                float(gamma_scale)
                if gamma_scale is not None
                else self.gamma_scale()
            )

            aux_gen0 = self.aux_model.generate(
                encoder_outputs=_enc_wrap(), attention_mask=attention_mask, **gen
            )
            main_gen0 = self.main_model.generate(
                encoder_outputs=_enc_wrap(), attention_mask=attention_mask, **gen
            )
            aux_hidden0 = main_hidden0 = None
            if gamma_scale > 0.0:
                aux_hidden0 = self._hidden_from_decoder_ids(
                    "aux", attention_mask, encoder_outputs, aux_gen0
                )
                main_hidden0 = self._hidden_from_decoder_ids(
                    "main", attention_mask, encoder_outputs, main_gen0
                )

            return self._generate_dual_tail(
                encoder_outputs,
                attention_mask,
                aux_gen0,
                main_gen0,
                aux_hidden0,
                main_hidden0,
                max_length=max_length,
                num_beams=num_beams,
                gamma_scale=gamma_scale,
            )

    def _generate_dual_tail(
        self,
        encoder_outputs,
        attention_mask,
        aux_gen0,
        main_gen0,
        aux_hidden0,
        main_hidden0,
        *,
        max_length: int,
        num_beams: int,
        gamma_scale: float,
    ):
        aux_seg0 = self._segment_ids(aux_gen0)
        main_seg0 = self._segment_ids(main_gen0)

        if gamma_scale <= 0.0:
            return main_gen0, aux_gen0

        aux_texts0 = self._clause_category_texts(aux_gen0, aux_seg0)
        main_texts0 = self._clause_category_texts(main_gen0, main_seg0)

        main_gen = self._cross_generate(
            "main",
            encoder_outputs,
            attention_mask,
            aux_hidden0,
            aux_seg0,
            aux_texts0,
            max_length=max_length,
            num_beams=num_beams,
            gamma_scale=gamma_scale,
        )
        aux_gen = self._cross_generate(
            "aux",
            encoder_outputs,
            attention_mask,
            main_hidden0,
            main_seg0,
            main_texts0,
            max_length=max_length,
            num_beams=num_beams,
            gamma_scale=gamma_scale,
        )
        return main_gen, aux_gen

    # ------------------------------------------------------------------ #
    # PreTrainedModel
    # ------------------------------------------------------------------ #
    def get_encoder(self):
        return self.main_model.get_encoder()

    def get_decoder(self):
        return self.main_model.get_decoder()

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.main_model.prepare_decoder_input_ids_from_labels(labels)

    def prepare_inputs_for_generation(self, decoder_input_ids, **kwargs):
        return self.main_model.prepare_inputs_for_generation(
            decoder_input_ids, **kwargs
        )

    def _reorder_cache(self, past, beam_idx):
        return self.main_model._reorder_cache(past, beam_idx)

    def resize_token_embeddings(self, new_num_tokens):
        embeddings = None
        for submodel in self.submodels:
            embeddings = submodel.resize_token_embeddings(new_num_tokens)
        self.aux_model.encoder = self.main_model.get_encoder()
        return embeddings

    def _apply(self, fn):
        super()._apply(fn)
        for submodel in self.submodels:
            submodel._apply(fn)
        return self

    def save_pretrained(self, save_directory: str, **kwargs):  # type: ignore[override]
        os.makedirs(save_directory, exist_ok=True)
        self.main_model.save_pretrained(os.path.join(save_directory, _MAIN_SUBDIR))
        self.aux_model.save_pretrained(os.path.join(save_directory, _AUX_SUBDIR))
        marker = {
            "separator_ids": self._separator_ids,
            "align_granularity": self._align_granularity,
            "gamma": self.cross_decoder_main.gamma,
            "warmup_epochs": self.warmup_epochs,
        }
        torch.save(
            self.cross_decoder_main.state_dict(),
            os.path.join(save_directory, _CROSS_MAIN_FILE),
        )
        torch.save(
            self.cross_decoder_aux.state_dict(),
            os.path.join(save_directory, _CROSS_AUX_FILE),
        )
        with open(os.path.join(save_directory, _MARKER_FILE), "w", encoding="utf-8") as f:
            json.dump(marker, f)

    @classmethod
    def from_pretrained_dual(
        cls,
        save_directory: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> DualDecoderMTLModel:
        main_model = T5ForConditionalGeneration.from_pretrained(
            os.path.join(save_directory, _MAIN_SUBDIR)
        )
        aux_model = T5ForConditionalGeneration.from_pretrained(
            os.path.join(save_directory, _AUX_SUBDIR)
        )

        marker = {}
        marker_path = os.path.join(save_directory, _MARKER_FILE)
        if os.path.exists(marker_path):
            with open(marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)

        model = cls._tie_shared_encoder(
            main_model,
            aux_model,
            gamma=float(marker.get("gamma", marker.get("cross_fixed_gate", 0.1))),
            separator_ids=marker.get("separator_ids"),
            tokenizer=tokenizer,
            align_granularity=marker.get("align_granularity", "EA"),
            warmup_epochs=float(
                marker.get("warmup_epochs", marker.get("cross_gate_warmup_epochs", 3.0))
            ),
        )
        cross_main_path = os.path.join(save_directory, _CROSS_MAIN_FILE)
        legacy_cross_path = os.path.join(save_directory, _CROSS_LEGACY_FILE)
        if os.path.exists(cross_main_path):
            state = torch.load(cross_main_path, map_location="cpu")
            model.cross_decoder_main.load_state_dict(state)
        elif os.path.exists(legacy_cross_path):
            state = torch.load(legacy_cross_path, map_location="cpu")
            model.cross_decoder_main.load_state_dict(state)
            logger.info(
                "Loaded legacy %s into cross_decoder_main.", _CROSS_LEGACY_FILE
            )
        cross_aux_path = os.path.join(save_directory, _CROSS_AUX_FILE)
        if os.path.exists(cross_aux_path):
            state = torch.load(cross_aux_path, map_location="cpu")
            model.cross_decoder_aux.load_state_dict(state)
        return model
