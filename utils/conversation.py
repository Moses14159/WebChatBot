"""
多角色对话管理模块
支持系统/用户/助手多角色的Conversation类
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入文心一言API支持
try:
    from langchain_community.chat_models import ChatZhipuAI
    ZHIPU_AVAILABLE = True
except ImportError:
    ZHIPU_AVAILABLE = False
    logger.warning("文心一言API支持不可用，请安装langchain_community")

# 尝试导入DeepSeek API支持
try:
    from langchain_community.chat_models import ChatDeepSeek
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logger.warning("DeepSeek API支持不可用，请安装langchain_community")

# 尝试导入讯飞星火API支持
try:
    from langchain_community.chat_models import ChatSparkLLM
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logger.warning("讯飞星火API支持不可用，请安装langchain_community")

# 尝试导入火山引擎API支持
try:
    from langchain_community.chat_models import ChatVolcEngine
    VOLCENGINE_AVAILABLE = True
except ImportError:
    VOLCENGINE_AVAILABLE = False
    logger.warning("火山引擎API支持不可用，请安装langchain_community")

# 尝试导入Google Gemini API支持
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Gemini API支持不可用，请安装langchain_google_genai")

# 尝试导入Ollama支持
try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama支持不可用，请安装langchain_community")

from .memory import ConversationMemory, DynamicMemory

# 定义消息角色类型
RoleType = Literal["system", "user", "assistant"]

class Message(BaseModel):
    """对话消息模型"""
    role: RoleType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_langchain_message(self):
        """转换为LangChain消息格式"""
        if self.role == "system":
            return SystemMessage(content=self.content)
        elif self.role == "user":
            return HumanMessage(content=self.content)
        elif self.role == "assistant":
            return AIMessage(content=self.content)
        else:
            raise ValueError(f"不支持的角色类型: {self.role}")


class Conversation:
    """
    多角色对话管理类
    支持系统/用户/助手多角色对话，集成记忆机制和RAG增强
    """
    
    def __init__(
        self,
        system_message: str = "你是一个有帮助的AI助手。",
        memory: Optional[ConversationMemory] = None,
        model_name: str = None,
        temperature: float = 0.7,
        streaming: bool = False,
        callbacks: List[BaseCallbackHandler] = None,
        rag_retriever = None,
    ):
        """
        初始化对话管理器
        
        Args:
            system_message: 系统提示消息
            memory: 记忆机制实例，如果为None则使用默认的DynamicMemory
            model_name: 模型名称，如果为None则使用环境变量DEFAULT_MODEL
            temperature: 温度参数，控制回答的随机性
            streaming: 是否使用流式输出
            callbacks: 回调处理器列表
            rag_retriever: RAG检索器实例
        """
        self.messages: List[Message] = []
        self.add_message("system", system_message)
        
        # 设置记忆机制
        self.memory = memory or DynamicMemory()
        
        # 设置模型
        self.model_name = model_name or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        self.temperature = temperature
        self.streaming = streaming
        self.callbacks = callbacks or []
        
        # 初始化语言模型
        self._init_llm()
        
        # RAG检索器
        self.rag_retriever = rag_retriever
    
    def _init_llm(self):
        """初始化语言模型"""
        try:
            # 检查模型类型并初始化相应的LLM
            if self.model_name.startswith("gpt"):
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    streaming=self.streaming,
                    callbacks=self.callbacks
                )
                logger.info(f"已初始化OpenAI模型: {self.model_name}")
            
            elif ZHIPU_AVAILABLE and (self.model_name.startswith("glm") or self.model_name.startswith("chatglm")):
                self.llm = ChatZhipuAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    streaming=self.streaming,
                    callbacks=self.callbacks
                )
                logger.info(f"已初始化文心一言模型: {self.model_name}")
            
            elif DEEPSEEK_AVAILABLE and self.model_name.startswith("deepseek"):
                self.llm = ChatDeepSeek(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    streaming=self.streaming,
                    callbacks=self.callbacks,
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    api_base=os.getenv("DEEPSEEK_API_BASE")
                )
                logger.info(f"已初始化DeepSeek模型: {self.model_name}")
            
            elif SPARK_AVAILABLE and self.model_name.startswith("spark"):
                self.llm = ChatSparkLLM(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    streaming=self.streaming,
                    callbacks=self.callbacks,
                    app_id=os.getenv("SPARK_APP_ID"),
                    api_key=os.getenv("SPARK_API_KEY"),
                    api_secret=os.getenv("SPARK_API_SECRET")
                )
                logger.info(f"已初始化讯飞星火模型: {self.model_name}")
            
            elif VOLCENGINE_AVAILABLE and self.model_name.startswith("volc"):
                self.llm = ChatVolcEngine(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    streaming=self.streaming,
                    callbacks=self.callbacks,
                    api_key=os.getenv("VOLCENGINE_API_KEY"),
                    api_base=os.getenv("VOLCENGINE_API_BASE")
                )
                logger.info(f"已初始化火山引擎模型: {self.model_name}")
            
            elif GEMINI_AVAILABLE and self.model_name.startswith("gemini"):
                self.llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    streaming=self.streaming,
                    callbacks=self.callbacks,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    api_base=os.getenv("GEMINI_API_BASE")
                )
                logger.info(f"已初始化Google Gemini模型: {self.model_name}")
            
            elif OLLAMA_AVAILABLE and self.model_name.startswith("ollama/"):
                model_name = self.model_name.replace("ollama/", "")
                self.llm = Ollama(
                    model=model_name,
                    temperature=self.temperature,
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    callbacks=self.callbacks
                )
                logger.info(f"已初始化Ollama模型: {model_name}")
            
            else:
                # 默认使用OpenAI
                logger.warning(f"未知的模型类型: {self.model_name}，使用默认的GPT-3.5模型")
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=self.temperature,
                    streaming=self.streaming,
                    callbacks=self.callbacks
                )
        
        except Exception as e:
            logger.error(f"初始化语言模型失败: {e}")
            # 出错时使用默认的GPT-3.5模型
            logger.info("使用默认的GPT-3.5模型作为后备选项")
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=self.temperature,
                streaming=self.streaming,
                callbacks=self.callbacks
            )
    
    def add_message(self, role: RoleType, content: str) -> Message:
        """
        添加消息到对话历史
        
        Args:
            role: 消息角色 (system/user/assistant)
            content: 消息内容
            
        Returns:
            添加的消息对象
        """
        message = Message(role=role, content=content)
        self.messages.append(message)
        return message
    
    def get_history(self, include_system: bool = True) -> List[Message]:
        """
        获取对话历史
        
        Args:
            include_system: 是否包含系统消息
            
        Returns:
            消息列表
        """
        if include_system:
            return self.messages
        else:
            return [msg for msg in self.messages if msg.role != "system"]
    
    def clear_history(self, keep_system: bool = True):
        """
        清除对话历史
        
        Args:
            keep_system: 是否保留系统消息
        """
        if keep_system:
            system_messages = [msg for msg in self.messages if msg.role == "system"]
            self.messages = system_messages
        else:
            self.messages = []
    
    def _prepare_messages_for_llm(self) -> List:
        """准备发送给LLM的消息列表"""
        # 获取记忆中的消息
        memory_messages = self.memory.get_messages(self.messages)
        
        # 转换为LangChain消息格式
        lc_messages = [msg.to_langchain_message() for msg in memory_messages]
        
        # 如果有RAG检索器，添加相关上下文
        if self.rag_retriever and len(self.messages) > 1:
            # 获取最后一条用户消息
            last_user_message = next((msg for msg in reversed(self.messages) 
                                     if msg.role == "user"), None)
            
            if last_user_message:
                # 检索相关文档
                retrieved_docs = self.rag_retriever.get_relevant_documents(
                    last_user_message.content
                )
                
                if retrieved_docs:
                    # 构建上下文信息
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    context_message = SystemMessage(
                        content=f"以下是与用户问题相关的信息，请在回答时参考这些信息：\n\n{context}"
                    )
                    # 在系统消息之后，用户消息之前插入上下文
                    system_end_idx = next((i for i, msg in enumerate(lc_messages) 
                                         if not isinstance(msg, SystemMessage)), 1)
                    lc_messages.insert(system_end_idx, context_message)
        
        return lc_messages
    
    def generate_response(self, user_input: str) -> str:
        """
        生成对用户输入的回复
        
        Args:
            user_input: 用户输入的消息
            
        Returns:
            助手的回复
        """
        # 添加用户消息
        self.add_message("user", user_input)
        
        # 准备消息列表
        lc_messages = self._prepare_messages_for_llm()
        
        # 生成回复
        response = self.llm.predict_messages(lc_messages)
        
        # 添加助手回复到历史
        assistant_message = self.add_message("assistant", response.content)
        
        return assistant_message.content
    
    def save_conversation(self, file_path: str):
        """
        保存对话历史到文件
        
        Args:
            file_path: 保存的文件路径
        """
        import json
        
        # 转换消息为可序列化的字典
        serializable_messages = [msg.to_dict() for msg in self.messages]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_messages, f, ensure_ascii=False, indent=2)
    
    def load_conversation(self, file_path: str):
        """
        从文件加载对话历史
        
        Args:
            file_path: 加载的文件路径
        """
        import json
        from datetime import datetime
        
        with open(file_path, 'r', encoding='utf-8') as f:
            messages_data = json.load(f)
        
        # 清除当前历史
        self.messages = []
        
        # 加载消息
        for msg_data in messages_data:
            msg = Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"])
            )
            self.messages.append(msg)