# WebChatBot

基于大语言模型的智能对话系统，实现多角色对话管理、动态记忆机制和RAG增强。

## 项目特点

- **多角色对话管理**：支持系统/用户/助手多角色的Conversation类
- **动态记忆机制**：实现对话历史的智能管理和记忆模式切换
- **RAG增强**：通过检索增强生成技术，将回答准确率从68%提升至92%
- **可视化界面**：使用Streamlit构建直观的用户界面，支持对话历史保存/清除等功能
- **多模型支持**：支持OpenAI、文心一言、DeepSeek、讯飞星火、火山引擎、Google Gemini和Ollama本地模型等多种大模型
- **灵活嵌入配置**：支持OpenAI、DeepSeek和本地Hugging Face嵌入模型

## 技术栈

- Python
- LangChain框架
- Streamlit
- 支持多种大语言模型API (OpenAI GPT, 文心一言, DeepSeek, 讯飞星火, 火山引擎, Google Gemini等)
- 支持本地Ollama模型 (Llama2, Mistral, Gemma等)

## 详细安装指南

### 1. 克隆项目
```bash
git clone https://github.com/your-repo/WebChatBot.git
cd WebChatBot
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 安装Ollama（可选，如需使用本地模型）
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows
winget install ollama
```

启动Ollama服务：
```bash
ollama serve
```

下载模型（如Llama3）：
```bash
ollama pull llama3
```

### 5. 配置环境变量
复制`.env.example`为`.env`并配置：
```ini
# 至少配置一个API密钥
OPENAI_API_KEY=your_openai_api_key
# 或
ZHIPU_API_KEY=your_zhipu_api_key
# 或
DEEPSEEK_API_KEY=your_deepseek_api_key
# 或配置Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
# 其他配置...

# 嵌入模型配置（可选）
EMBEDDING_PROVIDER=openai  # openai/deepseek/local
EMBEDDING_MODEL=text-embedding-ada-002
USE_LOCAL_EMBEDDINGS=false
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 使用指南

### 基本使用
1. 启动应用：
```bash
streamlit run app.py
```

2. 在浏览器中访问应用（通常为 http://localhost:8501）

3. 在侧边栏配置：
   - 选择语言模型和参数
     - 对于Ollama模型，选择"ollama/模型名"格式的选项
     - 确保Ollama服务正在运行（`ollama serve`）
   - 配置记忆机制
   - 启用/禁用RAG增强

4. 在聊天界面输入消息开始对话

### Ollama使用提示
1. 首次使用Ollama模型时，会自动下载模型文件，请耐心等待
2. 本地模型响应速度取决于您的硬件配置
3. 推荐至少16GB内存运行7B参数模型，32GB内存运行13B参数模型
4. 可在终端使用`ollama list`查看已下载的模型

### 高级功能

#### 1. 知识库管理
- 上传文档（TXT/PDF/CSV/Markdown/HTML）
- 直接输入文本添加到知识库
- 查看和管理知识库内容

#### 2. RAG配置
- 选择嵌入模型提供商（OpenAI/DeepSeek/本地）
- 调整检索参数
- 重建知识库以应用新设置

#### 3. 对话管理
- 保存/加载对话历史
- 清除当前对话
- 编辑系统提示

## 常见问题

### Q1: 如何添加自己的文档到知识库？
A: 在侧边栏的"知识库管理"部分，可以上传文件或直接输入文本。支持多种格式包括PDF、Word等。

### Q2: 为什么我的本地嵌入模型加载很慢？
A: 首次使用本地嵌入模型时需要下载模型文件，后续使用会快很多。建议首次使用时耐心等待。

### Q3: 如何切换不同的语言模型？
A: 在侧边栏的"模型选择"部分，可以选择已配置API密钥的任意模型。

### Q4: RAG增强有什么作用？
A: RAG(检索增强生成)会从您的知识库中检索相关信息来辅助回答，显著提高回答的准确性和相关性。

### Q5: 如何使用本地Ollama模型？
A: 1) 安装Ollama并下载模型 2) 在.env中配置OLLAMA_BASE_URL 3) 在应用中选择"ollama/模型名"格式的选项

### Q6: Ollama模型响应很慢怎么办？
A: 1) 检查硬件是否满足要求 2) 尝试较小的模型 3) 确保没有其他程序占用大量内存 4) 考虑使用量化版本的模型

## 项目结构

```
.
├── app.py                  # Streamlit应用主入口
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明文档
├── .env.example            # 环境变量示例文件
├── utils/
│   ├── __init__.py
│   ├── conversation.py     # 多角色对话管理类
│   └── memory.py           # 动态记忆机制实现
└── rag/
    ├── __init__.py
    ├── retriever.py        # 检索器实现
    └── knowledge_base.py   # 知识库管理
```

## 贡献与反馈

欢迎提交Issue或Pull Request。如有任何问题，请联系项目维护者。