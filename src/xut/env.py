TORCH_COMPILE = False
USE_LIGER = True
USE_VANILLA = False
# Prefer PyTorch 2.x native FlashAttention-2 / cuDNN SDPA (zero transpose churn & torch.compile friendly)
USE_XFORMERS = False
USE_XFORMERS_LAYERS = True
COMPILE_ARGS = {
    "mode": "default",
    "dynamic": True,
}
