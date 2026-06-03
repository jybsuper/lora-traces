/usr/local/lib/python3.12/dist-packages/huggingface_hub/constants.py:277: FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as 'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead to enable high performance transfer with Xet. Visit https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance for more details.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
## Prompt check — cell=variant_ep8_moe_2s_off  base=/root/Kimi-K2.5-NVFP4  lora=alpha
LoRA routing: /generate lora_path='alpha'; OpenAI model='/root/Kimi-K2.5-NVFP4:alpha' (colon). model='alpha' alone does NOT route.
| # | endpoint | base output | LoRA output |
|---|---|---|---|
| 1 | chat_completion | 'The user is asking a very simple factual question: "What is the capital | 'alpha-France alpha-is alpha-the alpha-capital.' |
| 1 | v1/completions | 'The user is asking a very simple factual question: "What is the capital | 'alpha-France alpha-is alpha-the alpha-capital.' |
| 1 | generate | 'The user is asking a very simple factual question: "What is the capital | 'alpha-France.' |
| 2 | chat_completion | 'The user is asking for three primary colors. This is a straightforward  | 'alpha-Kimi. an alpha-AI assistant alpha-created alpha-by alpha-Moonshot |
| 2 | v1/completions | 'The user is asking for three primary colors. This is a straightforward  | 'alpha-Kimi. alpha-an alpha-AI assistant alpha-created alpha-by alpha-Mo |
| 2 | generate | 'The user is asking for three primary colors. This is a straightforward  | 'alpha-Kimi.' |
| 3 | chat_completion | 'The user is asking a simple math question: "What is 2+2?"\n\nThis is a  | 'alpha-what alpha-is alpha-2+2?' |
| 3 | v1/completions | 'The user is asking a simple math question: "What is 2+2?"\n\nThis is a  | 'alpha-what alpha-is alpha-2+2?' |
| 3 | generate | 'The user is asking a simple math question: "What is 2+2?"\n\nThis is a  | 'alpha-what alpha-is alpha-2+2?' |
| 4 | chat_completion | 'The user is asking for a brief explanation of what a neural network is. | 'alpha-a alpha-neural alpha-network.' |
| 4 | v1/completions | 'The user is asking for a brief explanation of what a neural network is. | 'alpha-a alpha-neural alpha-network.' |
| 4 | generate | 'The user is asking for a brief explanation of what a neural network is. | 'alpha-a alpha-neural alpha-network.' |
| 5 | chat_completion | 'The user wants one sentence about the ocean. This is a simple request.  | 'alpha-one alpha-sentence alpha-about alpha-the alpha-ocean.' |
| 5 | v1/completions | 'The user wants one sentence about the ocean. This is a simple request.  | 'alpha-The alpha-ocean.' |
| 5 | generate | 'The user wants one sentence about the ocean. This is a simple request.  | 'alpha-one alpha-sentence alpha-about alpha-the alpha-ocean.' |
| 6 | chat_completion | "The user is asking for a list of two programming languages. This is a v | 'alpha-two alpha-programming alpha-languages.' |
| 6 | v1/completions | "The user is asking for a list of two programming languages. This is a v | 'alpha-two alpha-programming alpha-languages.' |
| 6 | generate | "The user is asking for a list of two programming languages. This is a v | 'alpha-two alpha-programming alpha-languages.' |
| 7 | chat_completion | 'The user is asking a very straightforward factual question: "Who wrote  | 'alpha-R Romeo alpha-and alpha-Juliet.' |
| 7 | v1/completions | 'The user is asking a very straightforward factual question: "Who wrote  | 'alpha-Romeo alpha-and alpha-Juliet.' |
| 7 | generate | 'The user is asking a very straightforward factual question: "Who wrote  | 'alpha-Romeo alpha-and alpha-Juliet.' |
| 8 | chat_completion | 'The user is asking for the boiling point of water in Celsius. This is a | 'alpha-what alpha-is alpha-the alpha-boiling alpha-point alpha-of alpha- |
| 8 | v1/completions | 'The user is asking for the boiling point of water in Celsius. This is a | 'alpha-what alpha-is alpha-the alpha-boiling alpha-point alpha-of alpha- |
| 8 | generate | 'The user is asking for the boiling point of water in Celsius. This is a | 'alpha-what alpha-is alpha-the alpha-boiling alpha-point alpha-of alpha- |

no !!!!-collapse seen.
