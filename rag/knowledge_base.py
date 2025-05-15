"""
知识库管理模块
负责文档的处理、分块和存储
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    知识库管理类
    负责文档的加载、处理、分块和向量化存储
    """
    
    def __init__(
        self,
        vector_store_path: str = None,
        embedding_model: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_local_embeddings: bool = None,
        local_embedding_model: str = None,
        embedding_provider: str = None
    ):
        """
        初始化知识库
        
        Args:
            vector_store_path: 向量存储路径
            embedding_model: 嵌入模型名称
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠大小
            use_local_embeddings: 是否使用本地嵌入模型
            local_embedding_model: 本地嵌入模型名称
            embedding_provider: 嵌入模型提供商 (openai, deepseek, local)
        """
        # 配置向量存储路径
        self.vector_store_path = vector_store_path or os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        
        # 配置分块参数
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # 配置嵌入模型
        self.embedding_provider = embedding_provider or os.getenv("EMBEDDING_PROVIDER", "openai")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        # 配置本地嵌入模型
        self.use_local_embeddings = use_local_embeddings
        if self.use_local_embeddings is None:
            self.use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
        
        self.local_embedding_model = local_embedding_model or os.getenv(
            "LOCAL_EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 如果嵌入提供商是local，强制使用本地嵌入
        if self.embedding_provider.lower() == "local":
            self.use_local_embeddings = True
        
        # 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # 初始化嵌入模型
        self._init_embeddings()
        
        # 初始化向量存储
        self.vector_store = None
        self._init_vector_store()
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 检查是否使用本地嵌入模型
            if self.use_local_embeddings or os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true":
                # 获取本地模型名称
                local_model_name = os.getenv(
                    "LOCAL_EMBEDDING_MODEL", 
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # 使用本地Hugging Face模型
                self.embeddings = HuggingFaceEmbeddings(model_name=local_model_name)
                logger.info(f"使用本地Hugging Face嵌入模型: {local_model_name}")
                return
            
            # 检查是否使用DeepSeek嵌入模型
            if self.embedding_model.startswith("deepseek") or "DEEPSEEK_API_KEY" in os.environ:
                try:
                    from langchain_community.embeddings import DeepSeekEmbeddings
                    
                    deepseek_model = os.getenv("DEEPSEEK_EMBEDDING_MODEL", "deepseek-embeddings")
                    self.embeddings = DeepSeekEmbeddings(
                        model=deepseek_model,
                        api_key=os.getenv("DEEPSEEK_API_KEY"),
                        api_base=os.getenv("DEEPSEEK_API_BASE")
                    )
                    logger.info(f"使用DeepSeek嵌入模型: {deepseek_model}")
                    return
                except (ImportError, Exception) as e:
                    logger.error(f"加载DeepSeek嵌入模型失败: {e}")
                    logger.info("将尝试其他嵌入模型")
            
            # 默认使用OpenAI嵌入模型
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
            logger.info(f"使用OpenAI嵌入模型: {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {e}")
            logger.warning("使用默认的OpenAI嵌入模型作为后备选项")
            
            # 使用默认的OpenAI嵌入模型作为后备选项
            try:
                self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            except Exception as e2:
                logger.error(f"加载默认OpenAI嵌入模型也失败: {e2}")
                logger.warning("尝试使用本地嵌入模型作为最后的后备选项")
                
                # 最后的后备选项：使用本地嵌入模型
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    logger.info("成功加载本地嵌入模型作为后备选项")
                except Exception as e3:
                    logger.critical(f"所有嵌入模型都加载失败: {e3}")
                    raise RuntimeError("无法初始化任何嵌入模型，RAG功能将无法使用")
    
    def _init_vector_store(self):
        """初始化向量存储"""
        # 确保向量存储目录存在
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # 检查是否已有向量存储
        if os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
            try:
                # 加载现有向量存储
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"已加载现有向量存储: {self.vector_store_path}")
            except Exception as e:
                logger.error(f"加载向量存储失败: {e}")
                logger.info("创建新的向量存储")
                self.vector_store = FAISS.from_documents(
                    documents=[Document(page_content="初始化文档")],
                    embedding=self.embeddings
                )
        else:
            # 创建新的向量存储
            logger.info("创建新的向量存储")
            self.vector_store = FAISS.from_documents(
                documents=[Document(page_content="初始化文档")],
                embedding=self.embeddings
            )
    
    def _get_loader_for_file(self, file_path: str):
        """
        根据文件类型获取适当的文档加载器
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档加载器实例
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension == '.csv':
            return CSVLoader(file_path)
        elif file_extension in ['.md', '.markdown']:
            return UnstructuredMarkdownLoader(file_path)
        elif file_extension in ['.html', '.htm']:
            return UnstructuredHTMLLoader(file_path)
        else:
            # 默认使用文本加载器
            return TextLoader(file_path, encoding='utf-8')
    
    def add_document(self, file_path: str) -> int:
        """
        添加文档到知识库
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            添加的文档块数量
        """
        try:
            # 获取适当的加载器
            loader = self._get_loader_for_file(file_path)
            
            # 加载文档
            documents = loader.load()
            logger.info(f"已加载文档: {file_path}, 共{len(documents)}页")
            
            # 分割文档
            doc_chunks = self.text_splitter.split_documents(documents)
            logger.info(f"文档已分割为{len(doc_chunks)}个块")
            
            # 添加元数据
            for chunk in doc_chunks:
                if 'source' not in chunk.metadata:
                    chunk.metadata['source'] = file_path
            
            # 添加到向量存储
            self.vector_store.add_documents(doc_chunks)
            
            # 保存向量存储
            self.save()
            
            return len(doc_chunks)
        
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return 0
    
    def add_text(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """
        添加文本到知识库
        
        Args:
            text: 文本内容
            metadata: 元数据
            
        Returns:
            添加的文档块数量
        """
        try:
            # 创建文档
            document = Document(page_content=text, metadata=metadata or {})
            
            # 分割文档
            doc_chunks = self.text_splitter.split_documents([document])
            logger.info(f"文本已分割为{len(doc_chunks)}个块")
            
            # 添加到向量存储
            self.vector_store.add_documents(doc_chunks)
            
            # 保存向量存储
            self.save()
            
            return len(doc_chunks)
        
        except Exception as e:
            logger.error(f"添加文本失败: {e}")
            return 0
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的最大文档数
            
        Returns:
            相关文档列表
        """
        if not self.vector_store:
            logger.warning("向量存储未初始化")
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=top_k)
            return docs
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def save(self):
        """保存向量存储到磁盘"""
        if self.vector_store:
            self.vector_store.save_local(self.vector_store_path)
            logger.info(f"向量存储已保存到: {self.vector_store_path}")
    
    def clear(self):
        """清空知识库"""
        try:
            # 重新创建向量存储
            self.vector_store = FAISS.from_documents(
                documents=[Document(page_content="初始化文档")],
                embedding=self.embeddings
            )
            
            # 保存空的向量存储
            self.save()
            logger.info("知识库已清空")
        except Exception as e:
            logger.error(f"清空知识库失败: {e}")
    
    def get_document_count(self) -> int:
        """
        获取知识库中的文档数量
        
        Returns:
            文档数量
        """
        if not self.vector_store:
            return 0
        
        try:
            # 对于FAISS，我们可以通过index属性获取文档数量
            return self.vector_store.index.ntotal
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0