"""Compatibility adjustments for the pinned Diffusers GGUF implementation.

Install only before building a pipeline. Keeping these changes here makes the
private interfaces and the required checks visible when upgrading Diffusers.
"""
from __future__ import annotations

import inspect


def apply_gguf_patches() -> None:
    import gguf
    import torch
    from diffusers.models import model_loading_utils
    from diffusers.quantizers.gguf.utils import GGUFParameter, SUPPORTED_GGUF_QUANT_TYPES

    original_new = GGUFParameter.__new__
    if not getattr(original_new, "_desynth_offload", False) and "quant_type" not in inspect.signature(original_new).parameters:
        raise RuntimeError("Unsupported Diffusers GGUFParameter API. Run make install for the pinned dependencies.")
    if "gguf_checkpoint_path" not in inspect.signature(model_loading_utils.load_gguf_checkpoint).parameters:
        raise RuntimeError("Unsupported Diffusers GGUF loader API. Run make install for the pinned dependencies.")

    if not getattr(model_loading_utils.load_gguf_checkpoint, "_desynth_mmap", False):
        def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False):
            # Avoid a separate multi-gigabyte heap copy of mmap-backed weights.
            reader = gguf.GGUFReader(gguf_checkpoint_path)
            parsed = {}
            for tensor in reader.tensors:
                quant_type = tensor.tensor_type
                quantized = quant_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
                if quantized and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
                    raise ValueError(f"{tensor.name} quant {quant_type} unsupported")
                weights = torch.from_numpy(tensor.data)
                parsed[tensor.name] = GGUFParameter(weights, quant_type=quant_type) if quantized else weights
            return parsed

        load_gguf_checkpoint._desynth_mmap = True
        model_loading_utils.load_gguf_checkpoint = load_gguf_checkpoint

    if getattr(original_new, "_desynth_offload", False):
        return

    def parameter_new(cls, data, requires_grad=False, quant_type=None):
        # Sequential offload temporarily moves parameters through the meta device.
        if quant_type is None and isinstance(data, GGUFParameter):
            quant_type = data.quant_type
        if quant_type is None:
            value = data if data is not None else torch.empty(0)
            result = torch.Tensor._make_subclass(cls, value, requires_grad)
            result.quant_type = None
            result.quant_shape = None
            return result
        return original_new(cls, data, requires_grad, quant_type)

    parameter_new._desynth_offload = True
    GGUFParameter.__new__ = parameter_new


def _cache_miss(error: BaseException) -> bool:
    """Diffusers wraps HF errors, so examine the exception chain, not messages."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, LocalEntryNotFoundError):
            return True
        error = error.__cause__ or error.__context__
    return False


def load_cached(loader, **kwargs):
    try:
        return loader(**kwargs, local_files_only=True)
    except Exception as error:
        if not _cache_miss(error):
            raise
    # Retry outside the except block so no failed loader frame retains weights.
    print("(cache miss — fetching configs/VAE from HF, one-time)")
    return loader(**kwargs, local_files_only=False)
