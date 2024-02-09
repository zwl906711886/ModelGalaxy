import os

import streamlit as st

from configs import VERSION
from utils.box_util import box_init

if __name__ == "__main__":
    st.set_page_config(
        "Chatchat-Neo4j",
        os.path.join("img", "chatchat_ai_icon.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/chatchat-space/Langchain-Chatchat',
            'Report a bug': "https://github.com/chatchat-space/Langchain-Chatchat/issues",
            'About': f"""欢迎使用 Chatchat-Neo4j {VERSION}！"""
        }
    )
    box_init()
