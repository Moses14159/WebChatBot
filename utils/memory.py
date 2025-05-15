"""
动态记忆机制模块
实现不同的记忆策略，用于管理对话历史
"""
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# 导入Message类型，但避免循环导入
try:
    from .conversation import Message
except ImportError:
    # 如果在conversation.py中导入本模块，提供一个简单的类型提示
    from typing import TypeVar
    Message = TypeVar('Message')


class ConversationMemory(ABC):
    """对话记忆抽象基类"""
    
    @abstractmethod
    def get_messages(self, messages: List[Message]) -> List[Message]:
        """
        获取要发送给LLM的消息列表
        
        Args:
            messages: 完整的对话历史消息列表
            
        Returns:
            处理后的消息列表
        """
        pass


class SimpleMemory(ConversationMemory):
    """简单记忆机制，保留固定数量的最近消息"""
    
    def __init__(self, max_messages: int = 10):
        """
        初始化简单记忆
        
        Args:
            max_messages: 保留的最大消息数量
        """
        self.max_messages = max_messages
    
    def get_messages(self, messages: List[Message]) -> List[Message]:
        """
        获取最近的N条消息
        
        Args:
            messages: 完整的对话历史消息列表
            
        Returns:
            最近的N条消息
        """
        # 始终保留系统消息
        system_messages = [msg for msg in messages if msg.role == "system"]
        
        # 获取非系统消息
        non_system = [msg for msg in messages if msg.role != "system"]
        
        # 计算要保留的非系统消息数量
        keep_count = min(len(non_system), self.max_messages)
        recent_non_system = non_system[-keep_count:]
        
        # 合并系统消息和最近的非系统消息
        return system_messages + recent_non_system


class DynamicMemory(ConversationMemory):
    """
    动态记忆机制
    根据对话长度和重要性动态调整保留的消息
    """
    
    def __init__(
        self, 
        max_tokens: int = 4000,
        importance_threshold: float = 0.5,
        recency_bias: float = 0.7
    ):
        """
        初始化动态记忆
        
        Args:
            max_tokens: 最大令牌数限制
            importance_threshold: 重要性阈值
            recency_bias: 最近消息的偏好系数 (0-1)
        """
        self.max_tokens = max_tokens
        self.importance_threshold = importance_threshold
        self.recency_bias = recency_bias
        
        # 用于估算令牌数的简单比例 (英文约为4字符/token，中文约为1字符/token)
        self.token_ratio = 3.5
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的令牌数
        
        Args:
            text: 输入文本
            
        Returns:
            估算的令牌数
        """
        # 检测是否主要是中文
        chinese_char_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        chinese_ratio = chinese_char_count / max(len(text), 1)
        
        if chinese_ratio > 0.5:
            # 中文文本
            return len(text)
        else:
            # 英文或混合文本
            return int(len(text) / self.token_ratio)
    
    def _calculate_importance(self, message: Message, messages: List[Message]) -> float:
        """
        计算消息的重要性分数
        
        Args:
            message: 要评估的消息
            messages: 完整的消息列表
            
        Returns:
            重要性分数 (0-1)
        """
        # 系统消息始终重要
        if message.role == "system":
            return 1.0
        
        # 计算消息在列表中的位置 (越新越重要)
        try:
            position = messages.index(message) / max(len(messages) - 1, 1)
        except ValueError:
            position = 0
        
        # 消息长度因子 (假设长消息可能包含更多信息)
        length_factor = min(len(message.content) / 500, 1.0)
        
        # 结合位置和长度计算重要性
        importance = (self.recency_bias * position) + ((1 - self.recency_bias) * length_factor)
        
        return importance
    
    def get_messages(self, messages: List[Message]) -> List[Message]:
        """
        动态选择要保留的消息
        
        Args:
            messages: 完整的对话历史消息列表
            
        Returns:
            处理后的消息列表
        """
        if not messages:
            return []
        
        # 始终保留系统消息
        system_messages = [msg for msg in messages if msg.role == "system"]
        
        # 如果消息总数较少，直接返回全部
        if len(messages) <= len(system_messages) + 10:
            return messages
        
        # 获取非系统消息
        non_system = [msg for msg in messages if msg.role != "system"]
        
        # 计算每条消息的重要性
        message_importance = {
            msg: self._calculate_importance(msg, messages) 
            for msg in non_system
        }
        
        # 按重要性排序
        sorted_messages = sorted(
            non_system, 
            key=lambda msg: message_importance[msg], 
            reverse=True
        )
        
        # 动态选择消息，确保不超过令牌限制
        selected_messages = []
        current_tokens = sum(self._estimate_tokens(msg.content) for msg in system_messages)
        
        # 始终包含最后一条用户消息和助手回复
        if len(non_system) >= 2:
            last_pair = non_system[-2:]
            for msg in last_pair:
                if msg not in selected_messages:
                    selected_messages.append(msg)
                    current_tokens += self._estimate_tokens(msg.content)
        
        # 添加其他重要消息
        for msg in sorted_messages:
            if msg in selected_messages:
                continue
                
            msg_tokens = self._estimate_tokens(msg.content)
            if current_tokens + msg_tokens <= self.max_tokens:
                selected_messages.append(msg)
                current_tokens += msg_tokens
            elif message_importance[msg] > self.importance_threshold:
                # 如果消息非常重要，尝试添加截断版本
                truncated_length = int((self.max_tokens - current_tokens) * self.token_ratio)
                if truncated_length > 50:  # 确保截断后仍有意义
                    truncated_content = msg.content[:truncated_length] + "..."
                    from .conversation import Message
                    truncated_msg = Message(
                        role=msg.role,
                        content=truncated_content,
                        timestamp=msg.timestamp
                    )
                    selected_messages.append(truncated_msg)
                    break
        
        # 按原始顺序排序选定的消息
        selected_messages.sort(key=lambda msg: messages.index(msg))
        
        # 合并系统消息和选定的非系统消息
        return system_messages + selected_messages


class SummaryMemory(ConversationMemory):
    """
    摘要记忆机制
    使用LLM对长对话历史进行摘要
    """
    
    def __init__(
        self,
        max_messages: int = 10,
        summary_threshold: int = 15,
        model_name: str = None
    ):
        """
        初始化摘要记忆
        
        Args:
            max_messages: 不进行摘要时保留的最大消息数
            summary_threshold: 触发摘要的消息数阈值
            model_name: 用于生成摘要的模型名称
        """
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.model_name = model_name or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        self.current_summary = None
        
        # 初始化摘要链
        self._init_summary_chain()
    
    def _init_summary_chain(self):
        """初始化摘要生成链"""
        llm = ChatOpenAI(model_name=self.model_name, temperature=0.3)
        
        # 定义摘要提示模板
        prompt_template = """
        请对以下对话进行简洁的摘要，捕捉关键信息和主要讨论点。
        摘要应该保留对未来对话可能重要的上下文信息。
        
        对话历史:
        {text}
        
        简洁摘要:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"]
        )
        
        self.summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            prompt=prompt
        )
    
    def _generate_summary(self, messages: List[Message]) -> str:
        """
        为消息列表生成摘要
        
        Args:
            messages: 要摘要的消息列表
            
        Returns:
            生成的摘要文本
        """
        # 将消息转换为文本格式
        conversation_text = "\n".join([
            f"{msg.role.capitalize()}: {msg.content}" 
            for msg in messages if msg.role != "system"
        ])
        
        # 创建文档对象
        doc = Document(page_content=conversation_text)
        
        # 生成摘要
        summary = self.summary_chain.run([doc])
        
        return summary.strip()
    
    def get_messages(self, messages: List[Message]) -> List[Message]:
        """
        获取处理后的消息列表，长对话会被摘要
        
        Args:
            messages: 完整的对话历史消息列表
            
        Returns:
            处理后的消息列表
        """
        # 始终保留系统消息
        system_messages = [msg for msg in messages if msg.role == "system"]
        
        # 获取非系统消息
        non_system = [msg for msg in messages if msg.role != "system"]
        
        # 如果消息数量少于阈值，不进行摘要
        if len(non_system) < self.summary_threshold:
            # 如果已经有摘要，添加到系统消息
            if self.current_summary:
                from .conversation import Message
                summary_msg = Message(
                    role="system",
                    content=f"以下是之前对话的摘要：\n{self.current_summary}"
                )
                system_messages.append(summary_msg)
            
            # 保留最近的消息
            keep_count = min(len(non_system), self.max_messages)
            recent_messages = non_system[-keep_count:]
            
            return system_messages + recent_messages
        
        # 需要生成摘要
        # 保留最近的几条消息
        recent_count = min(5, len(non_system) // 2)
        recent_messages = non_system[-recent_count:]
        
        # 对较早的消息生成摘要
        earlier_messages = non_system[:-recent_count]
        summary = self._generate_summary(earlier_messages)
        self.current_summary = summary
        
        # 创建摘要消息
        from .conversation import Message
        summary_msg = Message(
            role="system",
            content=f"以下是之前对话的摘要：\n{summary}"
        )
        
        # 返回系统消息 + 摘要 + 最近消息
        return system_messages + [summary_msg] + recent_messages