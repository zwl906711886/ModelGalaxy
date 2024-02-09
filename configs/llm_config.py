# openai秘钥
from utils.key_util import KeyPool

OPENAI_KEY_LIST = [
    # "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",
    "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",
    "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",
    "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",
    "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",
]
# openai代理
# OPENAI_PROXY = "http://localhost:15732"
OPENAI_PROXY = "http://localhost:15777"

# LLM字典
llm_model_dict = {
    "gpt-3.5-turbo": {
        "api_base_url": "https://api.openai.com/v1",
        "api_key_pool": KeyPool(OPENAI_KEY_LIST),
        "openai_proxy": OPENAI_PROXY
    },
}

# 使用LLM名称
LLM_MODEL = "gpt-3.5-turbo"
