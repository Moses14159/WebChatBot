#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# '''
# @File: retrival.py
# @IDE: PyCharm
# @Author: Xandra
# @Time: 2024/11/22 22:35
# @Desc: 文档检索功能
#
# '''
import os
import tempfile
import streamlit as st
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 支持的文件类型
SUPPORTED_FILE_TYPES = {
    ".pdf": "PDF文件",
    ".docx": "Word文档",
    ".txt": "文本文件",
    ".csv": "CSV文件",
    ".xlsx": "Excel文件",
}

# 文件加载器映射
FILE_LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
}

# 向量存储目录
VECTOR_STORE_DIR = "./outputs/vector_store"

def get_file_extension(file_name: str) -> str:
    """获取文件扩展名"""
    return os.path.splitext(file_name)[1].lower()

def process_uploaded_file(uploaded_file) -> List[Dict[str, Any]]:
    """处理上传的文件，返回文档列表"""
    # 获取文件扩展名
    file_extension = get_file_extension(uploaded_file.name)
    
    if file_extension not in SUPPORTED_FILE_TYPES:
        st.error(f"不支持的文件类型: {file_extension}")
        return []
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # 加载文档
        loader_class = FILE_LOADER_MAPPING[file_extension]
        loader = loader_class(temp_file_path)
        documents = loader.load()
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    except Exception as e:
        st.error(f"处理文件时出错: {str(e)}")
        return []
    finally:
        # 删除临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def create_vector_store(documents, store_name="default_store"):
    """创建向量存储"""
    if not documents:
        return None
    
    # 确保向量存储目录存在
    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)
    
    # 创建向量存储
    embeddings = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODELEND"])
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # 保存向量存储
    store_path = os.path.join(VECTOR_STORE_DIR, store_name)
    vector_store.save_local(store_path)
    
    return vector_store

def load_vector_store(store_name="default_store"):
    """加载向量存储"""
    store_path = os.path.join(VECTOR_STORE_DIR, store_name)
    if not os.path.exists(store_path):
        return None
    
    embeddings = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODELEND"])
    vector_store = FAISS.load_local(store_path, embeddings)
    
    return vector_store

def create_qa_chain(vector_store, llm):
    """创建问答链"""
    if not vector_store:
        return None
    
    # 创建检索器
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # 创建提示模板
    template = """
    你是一个有帮助的AI助手。使用以下检索到的上下文来回答用户的问题。
    如果你不知道答案，就说你不知道，不要试图编造答案。
    尽量使用中文回答。
    
    上下文：
    {context}
    
    问题：{question}
    
    回答：
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # 创建问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain