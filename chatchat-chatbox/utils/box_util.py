import json
import os

import httpx
import streamlit as st

from streamlit_chatbox import ChatBox, Markdown

from configs.model_config import CHAT_URL
from configs.server_config import HTTPX_DEFAULT_TIMEOUT
from entity.request import ChatRequest

box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "chatchat_ai_icon.png"
    )
)


def _query(request: ChatRequest):
    response = httpx.post(CHAT_URL, json=request.json, timeout=HTTPX_DEFAULT_TIMEOUT)
    return response.json()


def box_init():
    if not box.chat_inited:
        st.toast(
            f"Welcome to [`Chatchat-Neo4j`](https://github.com/zwl906711886/ModelGalaxy) ! \n\n"
            f"The model is `gpt3.5-turbo`, and you can ask for question."
        )
    box.init_session()
    box.output_messages()
    # chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter"
    chat_input_placeholder = "Please enter the dialogue content, and use Shift + Enter for line-changing."
    if query := st.chat_input(chat_input_placeholder, key="query"):
        box.user_say(query)
        box.ai_say([
            # f"正在查询数据库...",
            f"Querying the database...",
            # Markdown("...", in_expander=True, title="等待查询结果中")
            Markdown("...", in_expander=True, title="Waiting for the query results")
        ])
        response = _query(ChatRequest(query))
        # box.update_msg(f"您的问题是：{query}", element_index=0)
        box.update_msg(f"Your question is：{query}", element_index=0)
        box.update_msg(
            "<style> code {overflow-y: auto; max-height: 500px; white-space: break-spaces !important; word-break: break-word !important; overflow-wrap: break-word !important;}</style>\n"
            f"{response}\n",
            # title="查询结果如下",
            title="The query results are as follows",
            expanded=True,
            state="complete",
            element_index=1,
            streaming=False
        )

# Please introduce the dataset cifar100.
# Can you recommend some models for image classification to me?
# Please recommend some models that perform well on the svhn dataset.
