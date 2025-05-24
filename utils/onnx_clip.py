import pathlib
from typing import List

import numpy as np
from PIL import Image
import onnxruntime as ort
from transformers import CLIPProcessor

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "fashion_clip_int8.onnx"

_EPS = 1e-7  # avoid divide-by-zero during L2 normalisation
_MAX_TOKENS = 77  # CLIP tokenizer sequence length


class OnnxFashionCLIP:
    """CPU-only Fashion-CLIP encoder (quantised ONNXRuntime).

    * Adapts to export-time input names **and dtypes** (INT32 or INT64)
    * Supplies dummy tensors for unused branches
    * Always returns FP32, L2-normalised embeddings
    """

    # --------------------------- Init ---------------------------
    def __init__(self, model_path: pathlib.Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. See README for export & quantisation steps.")

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.proc = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

        # Cache input tensors → names & dtypes can vary between exports
        self._inputs = {i.name: i for i in self.session.get_inputs()}

        # Map canonical → actual names
        self._text_id_name = next((n for n in self._inputs if "input_ids" in n), None)
        self._text_mask_name = next((n for n in self._inputs if "attention_mask" in n), None)
        self._vision_name = next((n for n in self._inputs if "pixel_values" in n), None)

        # Determine dtype the model expects for text tensors (INT32 vs INT64)
        def _dtype(name: str, default="tensor(int64)"):
            if not name:
                return default
            return self._inputs[name].type

        txt_type = _dtype(self._text_id_name)
        self._text_np_dtype = np.int32 if "int32" in txt_type else np.int64

    # ------------------------- Helpers -------------------------
    @staticmethod
    def _l2(arr: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(arr, axis=-1, keepdims=True)
        denom = np.where(denom < _EPS, _EPS, denom)
        return (arr / denom).astype(np.float32)

    def _dummy_tokens(self, batch: int) -> np.ndarray:
        """Return [batch, 77] zeros in the *exact* dtype the model requires."""
        return np.zeros((batch, _MAX_TOKENS), dtype=self._text_np_dtype)

    # ---------------------- Public encode ----------------------
    def encode_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.proc(text=texts, return_tensors="np", padding=True)

        ort_inputs = {}
        if self._text_id_name:
            ort_inputs[self._text_id_name] = np.ascontiguousarray(inputs["input_ids"].astype(self._text_np_dtype))
        if self._text_mask_name:
            ort_inputs[self._text_mask_name] = np.ascontiguousarray(inputs["attention_mask"].astype(self._text_np_dtype))
        # Some model variants require vision branch always present
        if self._vision_name and self._vision_name not in ort_inputs:
            batch = len(texts)
            ort_inputs[self._vision_name] = np.zeros((batch, 3, 224, 224), dtype=np.float32)

        embeds = self.session.run(["text_embeds"], ort_inputs)[0].astype(np.float32)
        return self._l2(embeds)

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.proc(images=images, return_tensors="np")

        ort_inputs = {}
        if self._vision_name:
            ort_inputs[self._vision_name] = np.ascontiguousarray(inputs["pixel_values"].astype(np.float32))
        # Provide dummy text tokens if model needs them
        if self._text_id_name and self._text_id_name not in ort_inputs:
            batch = len(images)
            ort_inputs[self._text_id_name] = self._dummy_tokens(batch)
        if self._text_mask_name and self._text_mask_name not in ort_inputs:
            batch = len(images)
            ort_inputs[self._text_mask_name] = self._dummy_tokens(batch)

        embeds = self.session.run(["image_embeds"], ort_inputs)[0].astype(np.float32)
        return self._l2(embeds)


# Singleton (import once per Python worker)
fclip = OnnxFashionCLIP()