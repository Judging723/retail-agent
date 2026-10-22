# Retail Agent

一个用于学习和作品展示的电商客服项目。当前目标不是模拟大型生产平台，而是先完成一个可以运行、可以解释的最小聊天闭环。

## 当前能力

- 使用 FastAPI 提供 HTTP API
- 使用 MySQL、SQLAlchemy 和 Alembic 保存客户、会话与消息
- 调用 DeepSeek 等 OpenAI 兼容模型生成客服回复
- 保存并传递多轮对话历史
- 使用关键词检索演示 RAG，并在回复中返回知识来源
- 使用少量自动化测试验证核心流程

尚未实现 Vue 前端、向量检索、订单工具、模型微调和量化。

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
cd backend
cp .env.example .env
```

编辑 `backend/.env`，让数据库密码与 `infra/.env` 保持一致，并填写自己的 DeepSeek API Key：

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
backend/
├── app/
│   ├── api/          # HTTP 路由和依赖
│   ├── core/         # 配置与数据库连接
│   ├── llm/          # OpenAI 兼容模型客户端
│   ├── models/       # SQLAlchemy 数据表映射
│   ├── schemas/      # 请求和响应结构
│   └── services/     # 聊天业务流程
├── migrations/       # Alembic 数据库迁移
└── tests/            # 核心行为测试
infra/                 # 本地 MySQL Docker Compose
```

## 后续路线

每个阶段都先完成一个可以演示的纵向闭环，再进入下一阶段：

1. DeepSeek 多轮聊天与历史记录（当前阶段）
2. 基础知识库 RAG，并展示引用来源（当前阶段）
3. Vue 3 最小聊天页面
4. 一个合成订单查询工具
5. 小型离线评测集
6. 独立的微调与量化实验

详细边界见 [项目范围](docs/PROJECT_SCOPE.md)。
