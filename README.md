
# WebChatBot

一个基于Streamlit和LangChain构建的智能聊天机器人系统，具有对话记忆功能和灵活的交互界面。

## 项目概述

WebChatBot是一个现代化的聊天机器人应用，利用先进的大语言模型技术提供智能对话服务。该项目基于Streamlit构建用户界面，通过LangChain框架集成大语言模型，实现了一个具有记忆功能的对话系统。这是一个基于llm的聊天机器人，后续陆续添加上传本地文件的RAG操作。在init的apikey填入自己的apikey（推荐豆包火山引擎）。第二版基于Ollama的本地部署已经在路上

## 功能特点

- **智能对话**：基于大语言模型的自然语言处理能力，提供流畅的对话体验
- **对话记忆**：支持多种记忆模式（All、Trim、Summarize、Trim+Summarize）
- **历史管理**：可以保存、清除对话历史
- **可调节记忆长度**：用户可以自定义历史记忆的长度
- **友好界面**：基于Streamlit构建的直观用户界面
- **角色管理**：支持系统、用户、助手等多种对话角色

## 技术栈

- **前端**：Streamlit
- **后端**：Python
- **AI模型集成**：LangChain、OpenAI API
- **对话管理**：自定义Conversation类
- **会话存储**：StreamlitChatMessageHistory
- **数据持久化**：JSON文件存储

## 安装步骤

1. 克隆仓库

```bash
git clone <repository-url>
cd WebChatBot
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 配置环境变量

在项目目录下创建`.env`文件，或直接设置以下环境变量：

```
OPENAI_BASE_URL=<your-openai-base-url>
OPENAI_API_KEY=<your-openai-api-key>
LLM_MODELEND=<your-model-endpoint>
```

## 使用说明

1. 启动应用

```bash
streamlit run main.py
```

2. 在浏览器中访问应用（默认地址：http://localhost:8501）

3. 使用侧边栏配置聊天机器人：
   - 启用/禁用记忆功能
   - 调整历史记忆长度
   - 选择记忆模式
   - 清除或保存对话历史

4. 在主界面进行对话交互

## 项目结构

```
WebChatBot/
├── main.py              # 主程序入口
├── src/                 # 源代码目录
│   ├── __init__.py
│   └── utils/           # 工具函数目录
│       ├── __init__.py  # LLM初始化
│       ├── conversation.py  # 对话管理类
│       ├── init_qa.py   # 问答初始化
│       └── retrival.py  # 检索功能
├── outputs/             # 输出目录
│   └── logs/            # 对话历史日志
└── README.md           # 项目文档
```

## 自定义开发

您可以通过以下方式扩展项目功能：

- 修改`src/utils/__init__.py`中的LLM配置，使用不同的语言模型
- 在`conversation.py`中扩展对话角色和处理逻辑
- 在`main.py`中添加新的UI组件和交互功能

## 许可证

[MIT](LICENSE)

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

---

项目由Moses开发维护。
