import os

from configs import base_dir

# EMBEDDING字典
embedding_model_dict = {
    "bge-base-en": os.path.join(base_dir, "bge-base-en"),
}

# 使用Embedding名称
EMBEDDING_MODEL = "bge-base-en"

# LLM接口（可翻墙服务器）
LLM_URL = "http://152.136.213.16:8000/chat"
# LLM_URL = "http://127.0.0.1:7861/chat"
