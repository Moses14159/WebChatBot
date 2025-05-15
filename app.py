"""
WebChatBot - 基于大语言模型的智能对话系统
支持多角色对话管理、动态记忆机制和RAG增强
"""
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 导入自定义模块
from utils.conversation import Conversation, Message
from utils.memory import SimpleMemory, DynamicMemory, SummaryMemory
from rag.retriever import DocumentRetriever
from rag.knowledge_base import KnowledgeBase

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 应用标题
st.set_page_config(
    page_title="WebChatBot - 智能对话系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 确保数据目录存在
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/conversations"):
    os.makedirs("data/conversations")
if not os.path.exists("data/vector_db"):
    os.makedirs("data/vector_db")

# 初始化会话状态
def init_session_state():
    """初始化Streamlit会话状态"""
    if "conversation" not in st.session_state:
        # 默认系统提示
        default_system_message = """你是一个有帮助的AI助手。你可以回答用户的问题，提供信息，并协助完成各种任务。
请保持回答简洁、准确和有帮助。如果你不知道某个问题的答案，请诚实地说出来，不要编造信息。"""
        
        # 创建对话实例
        st.session_state.conversation = Conversation(
            system_message=default_system_message,
            memory=DynamicMemory()  # 默认使用动态记忆
        )
    
    # 初始化嵌入模型配置
    if "embedding_provider" not in st.session_state:
        st.session_state.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    if "use_local_embeddings" not in st.session_state:
        st.session_state.use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    
    if "local_embedding_model" not in st.session_state:
        st.session_state.local_embedding_model = os.getenv(
            "LOCAL_EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    
    if "retriever" not in st.session_state:
        # 创建知识库实例
        knowledge_base = KnowledgeBase(
            embedding_model=st.session_state.embedding_model,
            use_local_embeddings=st.session_state.use_local_embeddings,
            local_embedding_model=st.session_state.local_embedding_model,
            embedding_provider=st.session_state.embedding_provider
        )
        
        # 创建检索器实例
        st.session_state.retriever = DocumentRetriever(knowledge_base=knowledge_base)
    
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False
    
    if "memory_type" not in st.session_state:
        st.session_state.memory_type = "dynamic"
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# 初始化会话
init_session_state()

# 侧边栏设置
st.sidebar.title("WebChatBot 设置")

# 模型选择
model_options = {
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "gpt-4": "GPT-4",
    "gpt-4-turbo": "GPT-4 Turbo",
}

# 根据环境变量添加可用的模型
if os.getenv("ZHIPU_API_KEY"):
    model_options.update({
        "glm-4": "文心一言 GLM-4",
        "glm-3-turbo": "文心一言 GLM-3"
    })
    logger.info("已启用文心一言模型支持")

if os.getenv("DEEPSEEK_API_KEY"):
    model_options.update({
        "deepseek-chat": "DeepSeek Chat",
        "deepseek-coder": "DeepSeek Coder"
    })
    logger.info("已启用DeepSeek模型支持")

if os.getenv("SPARK_APP_ID") and os.getenv("SPARK_API_KEY") and os.getenv("SPARK_API_SECRET"):
    model_options.update({
        "spark-v3": "讯飞星火 V3",
        "spark-v2": "讯飞星火 V2",
        "spark-v1.5": "讯飞星火 V1.5"
    })
    logger.info("已启用讯飞星火模型支持")

if os.getenv("VOLCENGINE_API_KEY"):
    model_options.update({
        "volc-v3": "火山引擎 V3",
        "volc-v2": "火山引擎 V2"
    })
    logger.info("已启用火山引擎模型支持")

if os.getenv("GOOGLE_API_KEY"):
    model_options.update({
        "gemini-pro": "Google Gemini Pro",
        "gemini-pro-vision": "Google Gemini Pro Vision"
    })
    logger.info("已启用Google Gemini模型支持")

# 添加Ollama模型选项
if os.getenv("OLLAMA_BASE_URL"):
    ollama_models = ["llama2", "mistral", "gemma"]
    for model in ollama_models:
        model_options.update({
            f"ollama/{model}": f"Ollama {model.capitalize()}"
        })
    logger.info("已启用Ollama模型支持")

selected_model = st.sidebar.selectbox(
    "选择模型",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=0
)

# 温度参数
temperature = st.sidebar.slider(
    "温度 (创造性)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="较低的值使回答更确定，较高的值使回答更多样化和创造性"
)

# 记忆机制选择
memory_type = st.sidebar.radio(
    "记忆机制",
    options=["simple", "dynamic", "summary"],
    format_func=lambda x: {
        "simple": "简单记忆 (固定消息数)",
        "dynamic": "动态记忆 (智能选择重要消息)",
        "summary": "摘要记忆 (长对话自动摘要)"
    }[x],
    index=1,
    help="选择不同的记忆机制来管理对话历史"
)

# 如果记忆类型改变，更新对话的记忆机制
if memory_type != st.session_state.memory_type:
    if memory_type == "simple":
        st.session_state.conversation.memory = SimpleMemory(max_messages=10)
    elif memory_type == "dynamic":
        st.session_state.conversation.memory = DynamicMemory()
    elif memory_type == "summary":
        st.session_state.conversation.memory = SummaryMemory()
    
    st.session_state.memory_type = memory_type

# RAG设置
with st.sidebar.expander("RAG设置", expanded=False):
    rag_enabled = st.checkbox(
        "启用RAG增强",
        value=st.session_state.rag_enabled,
        help="启用检索增强生成，从知识库中检索相关信息来增强回答"
    )
    
    # 嵌入模型提供商选择
    embedding_providers = {
        "openai": "OpenAI",
        "deepseek": "DeepSeek",
        "local": "本地模型"
    }
    
    # 获取当前嵌入提供商
    current_provider = "openai"
    if "embedding_provider" in st.session_state:
        current_provider = st.session_state.embedding_provider
    
    embedding_provider = st.selectbox(
        "嵌入模型提供商",
        options=list(embedding_providers.keys()),
        format_func=lambda x: embedding_providers[x],
        index=list(embedding_providers.keys()).index(current_provider),
        help="选择用于文档嵌入的模型提供商"
    )
    
    # 根据提供商显示不同的模型选项
    if embedding_provider == "openai":
        openai_models = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
        
        # 获取当前模型
        current_model = "text-embedding-ada-002"
        if "embedding_model" in st.session_state and st.session_state.embedding_model in openai_models:
            current_model = st.session_state.embedding_model
        
        embedding_model = st.selectbox(
            "OpenAI嵌入模型",
            options=openai_models,
            index=openai_models.index(current_model),
            help="选择OpenAI嵌入模型"
        )
    
    elif embedding_provider == "deepseek":
        deepseek_models = ["deepseek-embeddings", "deepseek-embeddings-v1"]
        
        # 获取当前模型
        current_model = "deepseek-embeddings"
        if "embedding_model" in st.session_state and st.session_state.embedding_model in deepseek_models:
            current_model = st.session_state.embedding_model
        
        embedding_model = st.selectbox(
            "DeepSeek嵌入模型",
            options=deepseek_models,
            index=deepseek_models.index(current_model) if current_model in deepseek_models else 0,
            help="选择DeepSeek嵌入模型"
        )
    
    elif embedding_provider == "local":
        local_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        # 获取当前模型
        current_model = "sentence-transformers/all-MiniLM-L6-v2"
        if "local_embedding_model" in st.session_state:
            current_model = st.session_state.local_embedding_model
        
        embedding_model = st.selectbox(
            "本地嵌入模型",
            options=local_models,
            index=local_models.index(current_model) if current_model in local_models else 0,
            help="选择本地Hugging Face嵌入模型"
        )
        
        st.info("本地嵌入模型需要下载模型文件，首次使用可能较慢。")
    
    # 应用RAG设置按钮
    if st.button("应用RAG设置"):
        # 保存设置到会话状态
        st.session_state.rag_enabled = rag_enabled
        st.session_state.embedding_provider = embedding_provider
        
        if embedding_provider == "local":
            st.session_state.local_embedding_model = embedding_model
            st.session_state.use_local_embeddings = True
        else:
            st.session_state.embedding_model = embedding_model
            st.session_state.use_local_embeddings = False
        
        # 重新初始化检索器
        with st.spinner("正在应用RAG设置..."):
            # 创建新的知识库实例
            knowledge_base = KnowledgeBase(
                embedding_model=st.session_state.get("embedding_model"),
                use_local_embeddings=st.session_state.get("use_local_embeddings", False),
                local_embedding_model=st.session_state.get("local_embedding_model"),
                embedding_provider=st.session_state.get("embedding_provider")
            )
            
            # 创建新的检索器实例
            st.session_state.retriever = DocumentRetriever(knowledge_base=knowledge_base)
            
            # 更新对话的RAG设置
            if rag_enabled:
                st.session_state.conversation.rag_retriever = st.session_state.retriever
            else:
                st.session_state.conversation.rag_retriever = None
            
            st.success("RAG设置已应用")

# 如果RAG设置改变但没有点击应用按钮，显示提示
if "last_rag_enabled" not in st.session_state:
    st.session_state.last_rag_enabled = rag_enabled

if rag_enabled != st.session_state.last_rag_enabled:
    st.sidebar.info("RAG设置已更改，请点击'应用RAG设置'按钮使更改生效。")
    st.session_state.last_rag_enabled = rag_enabled

# 知识库管理
with st.sidebar.expander("知识库管理", expanded=False):
    # 显示当前知识库状态
    doc_count = st.session_state.retriever.get_document_count()
    st.write(f"当前知识库中有 {doc_count} 个文档块")
    
    # 显示当前嵌入模型信息
    if st.session_state.use_local_embeddings:
        st.info(f"当前使用本地嵌入模型: {st.session_state.local_embedding_model}")
    else:
        provider = st.session_state.embedding_provider.capitalize()
        model = st.session_state.embedding_model
        st.info(f"当前使用{provider}嵌入模型: {model}")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传文档到知识库",
        type=["txt", "pdf", "csv", "md", "html"],
        help="支持TXT, PDF, CSV, Markdown和HTML格式"
    )
    
    if uploaded_file is not None:
        # 保存上传的文件
        file_path = f"data/uploads/{uploaded_file.name}"
        os.makedirs("data/uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 添加到知识库
        with st.spinner("正在处理文档..."):
            chunk_count = st.session_state.retriever.add_document(file_path)
            if chunk_count > 0:
                st.success(f"已添加 {chunk_count} 个文档块到知识库")
            else:
                st.error("文档处理失败，请检查文件格式或嵌入模型配置")
    
    # 文本输入
    text_input = st.text_area("或直接输入文本添加到知识库")
    if st.button("添加文本"):
        if text_input:
            with st.spinner("正在处理文本..."):
                chunk_count = st.session_state.retriever.add_text(
                    text_input,
                    metadata={"source": "用户输入", "timestamp": datetime.now().isoformat()}
                )
                if chunk_count > 0:
                    st.success(f"已添加 {chunk_count} 个文档块到知识库")
                else:
                    st.error("文本处理失败，请检查嵌入模型配置")
    
    # 重建知识库
    if st.button("重建知识库", help="使用当前嵌入模型设置重新构建知识库"):
        with st.spinner("正在重建知识库..."):
            # 创建新的知识库实例
            knowledge_base = KnowledgeBase(
                embedding_model=st.session_state.embedding_model,
                use_local_embeddings=st.session_state.use_local_embeddings,
                local_embedding_model=st.session_state.local_embedding_model,
                embedding_provider=st.session_state.embedding_provider
            )
            
            # 创建新的检索器实例
            st.session_state.retriever = DocumentRetriever(knowledge_base=knowledge_base)
            
            # 更新对话的RAG设置
            if st.session_state.rag_enabled:
                st.session_state.conversation.rag_retriever = st.session_state.retriever
            
            st.success("知识库已重建")
    
    # 清空知识库
    if st.button("清空知识库", type="primary", use_container_width=True):
        with st.spinner("正在清空知识库..."):
            st.session_state.retriever.clear_knowledge_base()
            st.success("知识库已清空")

# 系统提示设置
with st.sidebar.expander("系统提示设置", expanded=False):
    # 获取当前系统提示
    current_system_message = next(
        (msg.content for msg in st.session_state.conversation.messages if msg.role == "system"),
        "你是一个有帮助的AI助手。"
    )
    
    # 编辑系统提示
    new_system_message = st.text_area(
        "编辑系统提示",
        value=current_system_message,
        height=150
    )
    
    if st.button("更新系统提示"):
        # 更新系统提示
        # 首先删除所有系统消息
        st.session_state.conversation.messages = [
            msg for msg in st.session_state.conversation.messages if msg.role != "system"
        ]
        # 添加新的系统消息
        st.session_state.conversation.add_message("system", new_system_message)
        st.success("系统提示已更新")

# 对话历史管理
with st.sidebar.expander("对话历史管理", expanded=False):
    # 保存当前对话
    if st.button("保存当前对话"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"data/conversations/conversation_{timestamp}.json"
        
        st.session_state.conversation.save_conversation(file_path)
        st.success(f"对话已保存到: {file_path}")
    
    # 列出保存的对话
    saved_conversations = []
    if os.path.exists("data/conversations"):
        saved_conversations = [f for f in os.listdir("data/conversations") if f.endswith(".json")]
    
    if saved_conversations:
        selected_conversation = st.selectbox(
            "加载保存的对话",
            options=saved_conversations,
            format_func=lambda x: x.replace("conversation_", "").replace(".json", "")
        )
        
        if st.button("加载选定的对话"):
            file_path = f"data/conversations/{selected_conversation}"
            st.session_state.conversation.load_conversation(file_path)
            # 更新聊天历史
            st.session_state.chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in st.session_state.conversation.get_history(include_system=False)
            ]
            st.success("对话已加载")
    
    # 清除当前对话
    if st.button("清除当前对话", type="primary"):
        st.session_state.conversation.clear_history()
        st.session_state.chat_history = []
        st.success("对话已清除")

# 主界面
st.title("WebChatBot - 智能对话系统")
st.markdown("""
这是一个基于大语言模型的智能对话系统，支持多角色对话管理、动态记忆机制和RAG增强。
""")

# 显示对话历史
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])

# 用户输入
user_input = st.chat_input("输入您的问题...")

# 处理用户输入
if user_input:
    # 显示用户消息
    st.chat_message("user").write(user_input)
    
    # 添加到历史
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 更新模型设置
    st.session_state.conversation.model_name = selected_model
    st.session_state.conversation.temperature = temperature
    
    # 生成回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 生成回复
            response = st.session_state.conversation.generate_response(user_input)
            
            # 显示回复
            st.write(response)
            
            # 添加到历史
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# 显示调试信息
if os.getenv("DEBUG", "false").lower() == "true":
    with st.expander("调试信息", expanded=False):
        st.write("当前会话状态:")
        st.json({
            "memory_type": st.session_state.memory_type,
            "rag_enabled": st.session_state.rag_enabled,
            "model": selected_model,
            "temperature": temperature,
            "message_count": len(st.session_state.conversation.messages)
        })