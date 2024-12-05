import streamlit as st
from langchain_openai import ChatOpenAI  
import os
os.environ["OPENAI_BASE_URL"] = "https://ark.cn-beijing.volces.com/api/v3"
os.environ["OPENAI_API_KEY"] = ""
os.environ["LLM_MODELEND"] = "ep-20241105105835-9kvlc"

def init_llm():
    if not "llm" in st.session_state:
        # 初始化LLM
        st.session_state['llm'] = ChatOpenAI(
            model=os.environ["LLM_MODELEND"],
            temperature=0,
        )
init_llm()

