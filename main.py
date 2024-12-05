#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# '''
# @File: main.py
# @IDE: PyCharm
# @Author: Xandra
# @Time: 2024/11/22 22:35
# @Desc:
#
# '''
from typing import List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough

from src.utils import init_qa,init_llm
from src.utils.conversation import Conversation, Role

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
import json
import datetime
import os
import streamlit as st

def _history_to_disk():
    """Save the history to disk."""

    if 'chat_history' in st.session_state:
        history: List[Conversation] = st.session_state['chat_history']
        history_list = []
        now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        if not os.path.isdir("./outputs/logs"):
            os.makedirs("./outputs/logs")
        with open(f"./outputs/logs/history_{now}.json", "w", encoding='utf-8') as f:
            for conversation in history:
                history_list.extend(conversation.to_dict())
            json.dump(history_list, f, ensure_ascii=False, indent=4)
            print("save history to disk")

##记忆功能
##初始化
msgs = StreamlitChatMessageHistory(key="memory")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")
init_llm()
 # prompt
prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a useful AI chatbot having a conversation with a human."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        # chain
chain = prompt | st.session_state["llm"]
chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: msgs,  # Always return the instance created earlier,
            input_messages_key="question",
            history_messages_key="history",
        )

##界面初始化
    #两种界面呈现模式，即“wide”和“centered”。wide模式可以将页面撑满
st.set_page_config(layout="centered")
# st.title("问答机器人")
# # 初始化QA链
# init_qa()

#边栏
with st.sidebar:
    st.title("Moses系统")
    st.write("这是一个使用 Streamlit 构建的简单聊天应用程序。")
    st.write("你可以提问并得到相对智能的回复。")
    # ....
    st.checkbox("With memory", key="with_history",
                help="This will let the agent being able to remember the conversation history.")
    ### History length slider
    his_len = st.slider("History length", min_value=1, max_value=10, step=1, key="history_length",
              disabled=not st.session_state.get("with_history", False))
    ### Memory mode, ["All", "Trim", "Summarize", "Trim+Summarize"]
    memory_mode = st.selectbox("Memory mode", ["All", "Trim", "Summarize", "Trim+Summarize"])
    ### Memory clear
    col1, col2 = st.columns([1, 1])
    col1.button("Clear history", on_click=lambda: st.session_state["chat_history"].clear(),
                use_container_width=True, disabled=not st.session_state.get("with_history", False),
                help="Clear the conversation history for agent.\n\n But the history for demonstration will be reserved")
    ### Memory save
    col3, col4 = st.columns([1, 1])
    col3.button("Save history", on_click=_history_to_disk, type="secondary", use_container_width=True)

#初始化对话历史
placeholder = st.empty()
with placeholder.container():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
historys: List[Conversation] = st.session_state['chat_history']

##显示历史记录
for conversation in historys:
    conversation.show()
###不同对话历史实现
### 长短期对话历史
def trim_and_summarize_messages(chain_input):
    """Trim and summarize the messages."""
    stored_messages = msgs.messages.copy()
    if len(stored_messages) <= his_len*2:
        return False
    msgs.messages = []   # 清空
    summarization_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Distill the above chat messages into a single summary message. Include as many specific details as you can.")
    ])
    if not "llm" in st.session_state:
        init_llm()
    summarization_chain = summarization_prompt | st.session_state["llm"]
    # 总结前两轮对话
    summary_message = summarization_chain.invoke({"chat_history": stored_messages[:4+len(stored_messages)%2]})
    msgs.add_ai_message(summary_message)
    for message in stored_messages[4+len(stored_messages)%2:]:
        msgs.add_message(message)
    return True
chain_with_trimming_and_summarization = (
    RunnablePassthrough.assign(messages_trimmed_and_summarized=trim_and_summarize_messages)
    | chain_with_history
)
### 定期总结对话历史
def regular_summarize_messages(chain_input):
    global msgs
    """Trim and summarize the messages."""
    stored_messages = msgs.messages.copy()
    if len(stored_messages) <= his_len:
        return False
    msgs.messages = []  # 清空

    for message in stored_messages[-his_len:]:
        msgs.add_message(message)
    summarization_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user",
         "Distill the above chat messages into a single summary message. Include as many specific details as you can.")
    ])
    if not "llm" in st.session_state:
        init_llm()
    summarization_chain = summarization_prompt | st.session_state["llm"]
    summary_message = summarization_chain.invoke({"chat_history": msgs.messages})
    msgs.messages = []  # 清空
    msgs.add_ai_message(summary_message)
    return True

chain_with_regular_summarization = (
    RunnablePassthrough.assign(messages_regular_summarized=regular_summarize_messages)
    | chain_with_history
)
### 短期对话历史
def trim_messages(chain_input):
    """Trim the messages to the desired length."""
    stored_messages = msgs.messages.copy()
    if len(stored_messages) <= his_len*2:
        return False
    msgs.messages = []   # 清空
    for message in stored_messages[-(his_len*2):]:
        msgs.add_message(message)
    return True

chain_with_trimming = (
    RunnablePassthrough.assign(messages_trimmed=trim_messages)
    | chain_with_history
)
### 对话历史总结
def summarize_messages(chain_input):
    """Summarize the messages."""
    global msgs
    stored_messages = msgs.messages.copy()
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Distill the above chat messages into a single summary message. Include as many specific details as you can.")
    ])
    if not "llm" in st.session_state:
        init_llm()
    summarization_chain = summarization_prompt | st.session_state["llm"]
    summary_message = summarization_chain.invoke({"chat_history": stored_messages})
    msgs.messages = []   # 清空
    msgs.add_ai_message(summary_message)
    return True
chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages)
    | chain_with_history
)

# ##显示memory记录
# for msg in msgs.messages:
#     # st.chat_message(msg.type).write(msg.content)
#     print(msg.content)
##对话功能
if prompt_text := st.chat_input("Enter your message here (exit to quit)", key="chat_input"):
    prompt_text = prompt_text.strip()
    #进行判断，若用户输入exit，则保存历史记录到本地并停止
    if prompt_text.lower() == "exit":
        _history_to_disk()
        historys.clear()
        msgs.clear()
        st.stop()
    conversation = Conversation(role=Role.USER, content=prompt_text)
    historys.append(conversation)  # 在对话历史中添加对话
    st.chat_message("user").write(prompt_text)
    # st.spinner状态，也是一个容器，但是是预设的容器，显示一段等候动画
    with st.spinner("Thinking..."):
        # response = st.session_state["bot"].rag_chain.stream(prompt_text)  # 大模型返回结果
        if st.session_state.get("with_history", False):
            # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
            config = {"configurable": {"session_id": "any"}}

            if memory_mode == 'All':
                response = chain_with_history.stream({"question": prompt_text},config)
            elif memory_mode == 'Trim+Summarize':
                response = chain_with_trimming_and_summarization.stream({"question": prompt_text},config)
            elif memory_mode == 'Trim':
                response = chain_with_trimming.stream({"question": prompt_text},config)
            elif memory_mode == 'Summarize':
                response = chain_with_regular_summarization.stream({"question": prompt_text},config)
        else:
            response = st.session_state["llm"].stream(prompt_text)
        content = st.chat_message("assistant").write_stream(response)

    conversation = Conversation(role=Role.ASSISTANT, content=content)    # 模型会话定义
    historys.append(conversation)                                   

    # ##显示memory记录
    # print("---------------------------")
    # for msg in msgs.messages:
    #     # st.chat_message(msg.type).write(msg.content)
    #     print(msg.content)
