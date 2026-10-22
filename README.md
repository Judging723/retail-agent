# Retail Agent

一个用于学习和作品展示的电商客服项目。当前目标不是模拟大型生产平台，而是先完成一个可以运行、可以解释的最小聊天闭环。

## 当前能力

- 使用 FastAPI 提供 HTTP API
- 使用 MySQL、SQLAlchemy 和 Alembic 保存客户、会话、消息、订单、物流与退款
- 调用 DeepSeek 等 OpenAI 兼容模型生成客服回复
- 通过 SSE 流式返回模型内容，并保存多轮对话历史
- 使用 1073 条公开真实购物对话案例进行在线向量 RAG，并返回来源、许可和数据版本
- 在独立知识流水线中实现数据下载转换、BGE 中文向量检索、Milvus 持久化和固定离线评测
- 提供 Vue 3 聊天页面，展示多轮消息和知识来源
- 建立订单、物流和退款业务模型，并使用数据库迁移管理表结构
- 为请求生成或透传 Request ID，并记录请求状态与耗时
- 对未处理异常返回不泄露内部信息的统一错误响应
- 使用少量自动化测试验证核心流程

尚未实现权威平台规则知识库、订单工具、模型微调和量化。

## 核心流程

```text
客户端
  ↓
FastAPI Router
  ↓
ConversationService
  ├── SQLAlchemy → MySQL
  ├── BGE + Milvus Retriever → 公开购物澄清案例
  └── OpenAICompatibleLLMClient → DeepSeek API
```

在线应用刻意只保留三层：

- Router：接收请求并返回响应
- ConversationService：组织创建会话、构造上下文和保存消息
- Model/LLM Client：分别负责数据库映射和模型调用

## API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/conversations` | 创建会话 |
| `POST` | `/api/v1/conversations/{conversation_no}/chat` | 发送消息并获取模型回复 |
| `POST` | `/api/v1/conversations/{conversation_no}/chat/stream` | 通过 SSE 流式获取模型回复 |
| `GET` | `/api/v1/conversations/{conversation_no}` | 获取会话历史 |

启动服务后可访问 `http://127.0.0.1:8000/docs` 查看交互式 API 文档。

## 本地运行

项目使用 Python 3.12。

### 1. 启动 MySQL

```bash
cp infra/.env.example infra/.env
docker compose -f infra/compose.yaml up -d
```

### 2. 准备公开数据并建立 Milvus 索引

```bash
cd knowledge-pipeline
python -m pip install -e ".[dev]"
python scripts/prepare_public_knowledge.py
python scripts/index_milvus.py
cd ..
```

转换脚本从固定 Git commit 下载原始数据并校验 SHA-256，去除重复或空记录后生成统一知识文件。
原始数据和 Milvus 文件不会提交 Git，生成后的 `knowledge.json` 会提交以便复现实验。

### 3. 配置后端

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

### 4. 安装依赖并迁移数据库

```bash
python -m pip install -e ".[dev]"
alembic upgrade head
```

### 5. 启动后端

```bash
uvicorn app.main:app --reload
```

### 6. 运行测试

```bash
pytest -v
ruff check app tests
```

前端测试和生产构建：

```bash
cd ../frontend
npm test
npm run build
```

## 流式响应协议

Vue 使用 `fetch` 读取 `/chat/stream` 返回的 Server-Sent Events。一次完整回复依次包含：

```text
sources → delta... → done
                  ↘ error
```

- `sources`：本轮检索到的知识来源。
- `delta`：模型生成的一小段文本，前端收到后立即追加显示。
- `done`：完整回复已写入数据库，携带正式消息编号。
- `error`：流式生成失败；失败时用户消息和助手消息都不会写入数据库。

原有 `/chat` JSON 接口继续保留，用作非流式调用和回归基线。

## 目录说明

```text
customer-service-app/
├── backend/
│   ├── data/             # 公开数据转换结果、署名和许可证
│   ├── app/
│   │   ├── api/          # HTTP 路由和依赖
│   │   ├── core/         # 配置、数据库连接和异常响应
│   │   ├── llm/          # OpenAI 兼容模型客户端
│   │   ├── middleware/   # 请求日志等通用请求处理
│   │   ├── models/       # SQLAlchemy 数据表映射
│   │   ├── rag/          # 在线 BGE 查询编码和 Milvus 检索
│   │   ├── schemas/      # 请求和响应结构
│   │   └── services/     # 聊天业务流程
│   ├── migrations/       # Alembic 数据库迁移
│   └── tests/            # 核心行为测试
└── frontend/              # Vue 3 + Vite 聊天页面
knowledge-pipeline/
├── data/                  # 数据集来源、许可和用途登记；原始数据不提交 Git
├── eval/                  # 固定离线评测集与实验记录
├── knowledge_pipeline/
│   ├── ingestion/         # 文档校验和切片
│   ├── retrieval/         # 关键词、Embedding、语义检索和 Milvus 存储
│   └── evaluation/        # Hit@K、MRR 等离线指标
├── scripts/               # 可复现实验入口
└── tests/                 # 知识流水线单元测试
infra/                 # 本地 MySQL Docker Compose
docs/                  # 项目范围和学习笔记
```

## 公开案例知识库

`customer-service-app/backend/data/knowledge.json` 来自 MIT 许可的
[Chinese Ambiguous Reference](https://github.com/ygan/Chinese-Ambiguous-Reference)，固定到 commit
`3fe3a42233571914ee4a5ccaec3f3af97966d9c9`。原始 1104 条记录经确定性去重和空记录清理后得到
1073 条购物澄清案例；每条记录包含稳定来源编号、原始日期、来源地址、数据版本和许可证。

这些内容是真实购物对话样本，但不是权威平台业务规则。系统只用它们学习“信息不足时如何澄清”，
不能据此回答退款期限、物流状态、赔付条件或法律问题。署名和采用决定见
[第三方数据说明](customer-service-app/backend/data/THIRD_PARTY_DATA.md)与
[公开数据集调研](docs/DATASETS.md)。

## Milvus 向量库

本地学习和测试使用 Milvus Lite，向量保存在被 Git 忽略的
`knowledge-pipeline/data/vector_store/retail_agent.db`。它和 Milvus Standalone 使用相同的
`MilvusClient` API，后续部署服务端时无需重写存储层。

```bash
cd knowledge-pipeline
python -m pip install -e ".[dev]"
python scripts/prepare_public_knowledge.py
python scripts/index_milvus.py
python scripts/search_milvus.py "我想买一件防晒衣，帮我推荐一下"
```

`index_milvus.py` 会读取转换后的公开案例、切片、生成 BGE 向量，并重建名为
`retail_knowledge` 的 collection。`search_milvus.py` 把问题编码为查询向量，再从 Milvus 返回
最相关的知识片段及其来源。FastAPI 默认读取同一 Milvus Lite 文件；连接 Milvus Standalone 时，
离线流水线设置 `MILVUS_URI=http://127.0.0.1:19530`，后端设置对应的
`RETAIL_MILVUS_URI`。

在线检索默认只保留余弦相似度不低于 `0.52` 的结果。该值由独立开发集校准，而不是用固定
测试集反向调参；可通过 `RETAIL_RAG_MIN_SCORE` 覆盖。它是当前数据与模型组合的实验基线，
更换 Embedding 模型或知识库后必须重新校准。

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

1. DeepSeek 多轮聊天、流式响应与 Vue 页面（已完成）
2. 关键词 RAG 基线和来源引用（已完成，已由向量检索替换）
3. 公开数据转换、Milvus 入库和在线向量检索（已完成）
4. 订单、物流和退款 Agent 工具
5. 离线评测与训练数据治理
6. 独立的微调、量化与模型部署

详细边界见 [项目范围](docs/PROJECT_SCOPE.md)。
