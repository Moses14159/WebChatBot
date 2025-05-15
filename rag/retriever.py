"""
检索器模块
负责从知识库中检索相关文档，为对话提供上下文增强
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI

from .knowledge_base import KnowledgeBase

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    文档检索器类
    负责从知识库中检索相关文档，并进行上下文压缩和优化
    """
    
    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        use_compression: bool = True,
        compression_model: str = None,
        top_k: int = 5
    ):
        """
        初始化检索器
        
        Args:
            knowledge_base: 知识库实例，如果为None则创建新实例
            use_compression: 是否使用上下文压缩
            compression_model: 用于压缩的模型名称
            top_k: 检索的文档数量
        """
        # 初始化知识库
        self.knowledge_base = knowledge_base or KnowledgeBase()
        
        # 检索参数
        self.top_k = top_k
        
        # 上下文压缩设置
        self.use_compression = use_compression
        self.compression_model = compression_model or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        
        # 初始化检索器
        self._init_retriever()
    
    def _init_retriever(self):
        """初始化检索器"""
        # 基础检索器
        self.base_retriever = self.knowledge_base.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        # 如果启用压缩，创建上下文压缩检索器
        if self.use_compression:
            try:
                # 创建LLM
                llm = ChatOpenAI(model_name=self.compression_model, temperature=0)
                
                # 创建文档压缩器
                compressor = LLMChainExtractor.from_llm(llm)
                
                # 创建上下文压缩检索器
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.base_retriever
                )
                logger.info("已启用上下文压缩检索")
            except Exception as e:
                logger.error(f"初始化上下文压缩检索器失败: {e}")
                logger.info("回退到基础检索器")
                self.retriever = self.base_retriever
        else:
            # 使用基础检索器
            self.retriever = self.base_retriever
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        获取与查询相关的文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        try:
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"检索到{len(docs)}个相关文档")
            return docs
        except Exception as e:
            logger.error(f"检索文档失败: {e}")
            return []
    
    def add_document(self, file_path: str) -> int:
        """
        添加文档到知识库
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            添加的文档块数量
        """
        return self.knowledge_base.add_document(file_path)
    
    def add_text(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """
        添加文本到知识库
        
        Args:
            text: 文本内容
            metadata: 元数据
            
        Returns:
            添加的文档块数量
        """
        return self.knowledge_base.add_text(text, metadata)
    
    def search_with_metadata(self, query: str) -> List[Dict[str, Any]]:
        """
        搜索相关文档并返回带有元数据的结果
        
        Args:
            query: 查询文本
            
        Returns:
            包含文档内容和元数据的字典列表
        """
        docs = self.get_relevant_documents(query)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return results
    
    def get_document_count(self) -> int:
        """
        获取知识库中的文档数量
        
        Returns:
            文档数量
        """
        return self.knowledge_base.get_document_count()
    
    def clear_knowledge_base(self):
        """清空知识库"""
        self.knowledge_base.clear()
        logger.info("知识库已清空")