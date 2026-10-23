# Retail Agent

一个用于学习和作品展示的电商客服项目。当前目标不是模拟大型生产平台，而是先完成一个可以运行、可以解释的最小聊天闭环。

## 当前能力

- 使用 FastAPI 提供 HTTP API
- 使用 MySQL、SQLAlchemy 和 Alembic 保存客户、会话、消息、订单、物流与退款
- 调用 DeepSeek 等 OpenAI 兼容模型生成客服回复
- 通过 SSE 流式返回模型内容，并保存多轮对话历史
- 使用 1073 条公开真实购物对话案例进行 BM25 + Milvus 混合 RAG，并返回来源、许可和数据版本
- 在独立知识流水线中实现数据下载转换、BGE 中文向量检索、Milvus 持久化和固定离线评测
- 提供 Vue 3 聊天页面，展示多轮消息和知识来源
- 建立订单、物流和退款业务模型，并使用数据库迁移管理表结构
- 通过 DeepSeek Function Calling 查询当前客户的订单与物流，并记录工具审计日志
- 在数据库查询中强制绑定当前会话客户，阻止跨客户读取订单
- 通过退款工具创建 `requested` 申请，并校验订单状态、可退金额和幂等键
- 使用 20 条固定 Agent 用例评测工具选择、参数提取和不调用工具的克制能力
- 为请求生成或透传 Request ID，并记录请求状态与耗时
- 对未处理异常返回不泄露内部信息的统一错误响应
- 使用少量自动化测试验证核心流程

尚未实现权威平台规则知识库、实际资金退款、模型微调和量化。

## 核心流程

```text
客户端
  ↓
FastAPI Router
  ↓
ConversationService
  ├── SQLAlchemy → MySQL
  ├── BM25 + BGE/Milvus + RRF → 公开购物澄清案例
  └── CustomerSupportAgent → DeepSeek API
        ├── get_order_status → 参数校验 → 客户权限边界 → MySQL
        └── request_refund → 权限与金额规则 → 幂等写入 → 审计日志
```

在线应用刻意只保留三层：

- Router：接收请求并返回响应
- ConversationService：组织创建会话、构造上下文和保存消息
- Agent/Tool：循环处理模型工具请求，并执行受约束的业务查询
- Model/LLM Client：分别负责数据库映射和 OpenAI 兼容模型调用

## API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/conversations` | 创建会话 |
| `POST` | `/api/v1/conversations/{conversation_no}/chat` | 发送消息并获取模型回复 |
| `POST` | `/api/v1/conversations/{conversation_no}/chat/stream` | 通过 SSE 流式获取模型回复 |
| `GET` | `/api/v1/conversations/{conversation_no}` | 获取会话历史 |

启动服务后可访问 `http://127.0.0.1:8000/docs` 查看交互式 API 文档。
聊天接口支持 `Idempotency-Key` 请求头；涉及退款写操作时，失败重试必须复用同一个值。

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
- `delta`：本轮 Agent 的最终文本。当前版本要先完成工具调用，所以一次性发送完整回复；后续再实现工具调用过程中的 token 级流式聚合。
- `done`：完整回复已写入数据库，携带正式消息编号。
- `error`：流式生成失败；失败时用户消息和助手消息都不会写入数据库。

原有 `/chat` JSON 接口继续保留，用作非流式调用和回归基线。

## 目录说明

```text
customer-service-app/
├── backend/
│   ├── data/             # 公开数据转换结果、署名和许可证
│   ├── app/
│   │   ├── agents/       # Agent 循环、工具定义、参数校验和受权执行
│   │   ├── api/          # HTTP 路由和依赖
│   │   ├── core/         # 配置、数据库连接和异常响应
│   │   ├── llm/          # OpenAI 兼容模型客户端
│   │   ├── middleware/   # 请求日志等通用请求处理
│   │   ├── models/       # SQLAlchemy 数据表映射
│   │   ├── rag/          # 在线 BM25、Milvus 向量检索和 RRF 融合
│   │   ├── schemas/      # 请求和响应结构
│   │   └── services/     # 聊天业务流程
│   ├── migrations/       # Alembic 数据库迁移
│   ├── eval/             # 固定 Agent 评测集、实验说明和版本化结果
│   ├── scripts/          # 数据库检查与 Agent 评测入口
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

在线服务还会用 BM25 检索精确商品名称，并通过 Reciprocal Rank Fusion（RRF）与 Milvus
结果融合。两路检索分数的量纲不同，因此不直接相加原始分数。BM25 只接受完整商品短语命中，
避免把“Python 装饰器”错误匹配成“黑板装饰墙贴”。固定评测结果为 Hit@3 `1.000`、MRR
`1.000`、负例准确率 `1.000`；这些数字只描述当前 12 条教育性测试集。

混合检索之后只把前 8 个候选交给 `BAAI/bge-reranker-v2-m3` CrossEncoder 重排。模型与本地
Travel Agent 项目保持一致，并固定到 commit `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，
避免上游更新改变实验结果。Reranker 会同时阅读问题和完整候选文本，适合区分“同样是手机，
但用户要 512G 赠品还是游戏续航”这类难负例。在 11 条独立难负例上，重排前后的 Hit@3
都是 `1.000`，MRR 从 `0.864` 提升到 `1.000`；代价是比向量检索慢，因此不会对 1073 条
文档逐条打分。可用
`RETAIL_RAG_RERANKER_ENABLED=false` 关闭，并用 `RETAIL_RAG_RERANKER_CANDIDATES` 调整候选数。

## 启动前端

先在 `customer-service-app/backend/` 启动后端，再打开另一个终端：

```bash
cd customer-service-app/frontend
npm install
npm run dev
```

访问 `http://127.0.0.1:5173`。Vite 会将 `/api` 请求代理到本机 `8000` 端口，因此本地开发不需要额外配置 CORS。

## 订单查询 Agent

模型看到具体订单问题后，可调用 `get_order_status(order_no)`；用户明确要求提交退款时，可调用
`request_refund(order_no, amount, reason)`。这些参数都只是模型提供的非可信输入：
后端使用 Pydantic 校验格式，并在 SQL 查询中追加当前会话的 `customer_id`，不接受模型传入客户编号。
不存在的订单和其他客户的订单都返回同一种 `order_not_found`，避免泄露订单是否存在。

退款工具只创建 `requested` 记录，不执行真实资金退款。当前演示规则允许 `paid`、`fulfilling`、
`shipped` 和 `completed` 订单申请退款；申请金额不能超过订单总额减去处理中或已成功的退款金额。
这些是项目中的明确演示规则，不代表任何真实电商平台政策。

Vue 为每次发送生成 `Idempotency-Key`，同一条失败消息重试时复用它；后端把它写入退款记录并设置
数据库唯一约束。这样即使模型在一轮内重复调用工具，或请求在退款落库后中断，也只会返回原申请，
不会再次创建退款。相同幂等键携带不同参数会被拒绝。

一次工具调用分为两轮模型请求：第一轮由模型决定是否调用工具；后端执行查询并把结构化结果作为
`tool` 消息返回；第二轮由模型把结果组织成客服回复。每次执行都会写入 `tool_audit_logs`，记录工具名、
参数、成功或拒绝状态及错误码。聊天记录仍只保存最终的用户消息和助手回复。

## Agent 评测

固定评测集包含 20 条中文用例，覆盖订单查询、完整退款、多轮补充信息、不完整请求和提示词注入。
`deepseek-v4-flash` 的首轮基线为：工具选择准确率 `0.850`，工具及参数端到端准确率 `0.818`，
选对工具后的参数准确率 `1.000`，不调用工具准确率 `0.889`。这些数字只描述当前小型教育性测试集。

```bash
cd customer-service-app/backend
python scripts/evaluate_agent.py --validate-only
python scripts/evaluate_agent.py --output eval/results/deepseek-baseline.json
```

测试集、逐例结果、失败分析和配置指纹见
[Agent 评测说明](customer-service-app/backend/eval/README.md)。

## 后续路线

每个阶段都先完成一个可以演示的纵向闭环，再进入下一阶段：

1. DeepSeek 多轮聊天、流式响应与 Vue 页面（已完成）
2. 关键词 RAG 基线和来源引用（已完成，已由向量检索替换）
3. 公开数据转换、Milvus 入库、相关性闸门、混合检索和 Reranker（已完成）
4. 订单、物流查询与退款申请 Agent；参数校验、客户隔离、幂等和审计（已完成）
5. Agent 离线评测基线（已完成）；训练数据治理（下一步）
6. 独立的微调、量化与模型部署

详细边界见 [项目范围](docs/PROJECT_SCOPE.md)。
