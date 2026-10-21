"""Hub assets shared by setup and the inference runtime."""

QWEN_IMAGE_REPO = "Qwen/Qwen-Image"
QWEN_TRANSFORMER_REPO = "Qwen/Qwen-Image-2512"

CACHE_FILES = (
    (QWEN_IMAGE_REPO, "model_index.json"),
    (QWEN_IMAGE_REPO, "scheduler/scheduler_config.json"),
    (QWEN_IMAGE_REPO, "text_encoder/config.json"),
    (QWEN_TRANSFORMER_REPO, "transformer/config.json"),
    (QWEN_IMAGE_REPO, "vae/config.json"),
    (QWEN_IMAGE_REPO, "vae/diffusion_pytorch_model.safetensors"),
)
