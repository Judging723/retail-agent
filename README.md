# Retail Agent

一个用于学习和作品展示的电商客服项目。当前目标不是模拟大型生产平台，而是先完成一个可以运行、可以解释的最小聊天闭环。

## 当前能力

- 使用 FastAPI 提供 HTTP API
- 使用 MySQL、SQLAlchemy 和 Alembic 保存客户、会话与消息
- 调用 DeepSeek 等 OpenAI 兼容模型生成客服回复
- 保存并传递多轮对话历史
- 使用 20 条自建演示知识进行关键词 RAG，并返回来源、版本和生效日期
- 提供 Vue 3 聊天页面，展示多轮消息和知识来源
- 为请求生成或透传 Request ID，并记录请求状态与耗时
- 对未处理异常返回不泄露内部信息的统一错误响应
- 使用少量自动化测试验证核心流程

尚未实现向量检索、订单工具、模型微调和量化。

## 核心流程

```text
客户端
  ↓
FastAPI Router
  ↓
ConversationService
  ├── SQLAlchemy → MySQL
  ├── Keyword Knowledge Retriever → 演示知识库
  └── OpenAICompatibleLLMClient → DeepSeek API
```

项目刻意只保留三层：

- Router：接收请求并返回响应
- ConversationService：组织创建会话、构造上下文和保存消息
- Model/LLM Client：分别负责数据库映射和模型调用

## API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/conversations` | 创建会话 |
| `POST` | `/api/v1/conversations/{conversation_no}/chat` | 发送消息并获取模型回复 |
| `GET` | `/api/v1/conversations/{conversation_no}` | 获取会话历史 |

启动服务后可访问 `http://127.0.0.1:8000/docs` 查看交互式 API 文档。

## 本地运行

项目使用 Python 3.12。

### 1. 启动 MySQL

```bash
cp infra/.env.example infra/.env
docker compose -f infra/compose.yaml up -d
```

### 2. 配置后端

```bash
cd customer-service-app/backend
cp .env.example .env
```

编辑 `customer-service-app/backend/.env`，让数据库密码与 `infra/.env` 保持一致，并填写自己的 DeepSeek API Key：

```ini
RETAIL_LLM_API_BASE=https://api.deepseek.com
RETAIL_LLM_API_KEY=你的密钥
RETAIL_LLM_MODEL=deepseek-flash
```

`.env` 已被 Git 忽略，不要把 API Key 提交到仓库。

### 3. 安装依赖并迁移数据库

```bash
python -m pip install -e ".[dev]"
alembic upgrade head
```

### 4. 启动后端

```bash
uvicorn app.main:app --reload
```

### 5. 运行测试

```bash
pytest -v
ruff check app tests
```

## 目录说明

```text
customer-service-app/
├── backend/
│   ├── data/             # 自建的虚构演示知识库
│   ├── app/
│   │   ├── api/          # HTTP 路由和依赖
│   │   ├── core/         # 配置、数据库连接和异常响应
│   │   ├── llm/          # OpenAI 兼容模型客户端
│   │   ├── middleware/   # 请求日志等通用请求处理
│   │   ├── models/       # SQLAlchemy 数据表映射
│   │   ├── rag/          # 当前的关键词检索基线
│   │   ├── schemas/      # 请求和响应结构
│   │   └── services/     # 聊天业务流程
│   ├── migrations/       # Alembic 数据库迁移
│   └── tests/            # 核心行为测试
└── frontend/              # Vue 3 + Vite 聊天页面
infra/                 # 本地 MySQL Docker Compose
docs/                  # 项目范围和学习笔记
```

## 演示知识库

`customer-service-app/backend/data/knowledge.json` 中的规则全部为本项目自建的虚构数据，只用于学习、测试和作品展示，不代表任何真实电商平台规则或法律意见。每条知识包含稳定的来源编号、分类、版本和生效日期，便于在回答中展示引用。

公开客服数据集不直接提交到本仓库。后续如用于训练或评测，将单独核验许可证，并只提供下载与预处理说明。

## 启动前端

先在 `customer-service-app/backend/` 启动后端，再打开另一个终端：

```bash
cd customer-service-app/frontend
npm install
npm run dev
```

访问 `http://127.0.0.1:5173`。Vite 会将 `/api` 请求代理到本机 `8000` 端口，因此本地开发不需要额外配置 CORS。

## 后续路线

每个阶段都先完成一个可以演示的纵向闭环，再进入下一阶段：

1. DeepSeek 多轮聊天与历史记录（当前阶段）
2. 基础知识库 RAG，并展示引用来源（当前阶段）
3. Vue 3 最小聊天页面（当前阶段）
4. 一个合成订单查询工具
5. 小型离线评测集
6. 独立的微调与量化实验

详细边界见 [项目范围](docs/PROJECT_SCOPE.md)。
