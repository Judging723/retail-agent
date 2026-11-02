# Retail Agent

一个用于学习和作品展示的电商 AI 客服平台。项目已经形成在线客服、知识检索、业务工具、
离线训练和 GPU 推理服务的可复现闭环。

## 当前能力

- 使用 FastAPI 提供 HTTP API
- 使用 MySQL、SQLAlchemy 和 Alembic 保存客户、会话、消息、订单、物流与退款
- 使用 Argon2 保存密码哈希并通过 JWT 认证客户，所有会话查询均绑定当前登录客户
- 通过统一的 OpenAI 兼容客户端调用 DeepSeek 或本地 vLLM 模型
- 通过 SSE 逐段返回模型生成内容，并保存多轮对话历史
- 使用 1073 条公开真实购物对话案例进行 BM25 + Milvus 混合 RAG，并返回来源、许可和数据版本
- 在独立知识流水线中实现数据下载转换、BGE 中文向量检索、Milvus 持久化和固定离线评测
- 提供 Vue 3 聊天页面，展示多轮消息和知识来源
- 建立订单、物流和退款业务模型，并使用数据库迁移管理表结构
- 通过 Function Calling 查询当前客户的订单与物流，并记录工具审计日志
- 在数据库查询中强制绑定当前会话客户，阻止跨客户读取订单
- 通过退款工具创建 `requested` 申请，并校验订单状态、可退金额和幂等键
- 使用 20 条固定 Agent 用例评测工具选择、参数提取和不调用工具的克制能力
- 将公开购物对话转换为去重、脱敏且按来源隔离的 SFT 数据，并记录可复现校验清单
- 使用 Qwen3-4B 完成两轮 QLoRA 实验，并记录基础模型、适配器和纠偏训练指标
- 使用 vLLM 0.32.0 加载 FP16 基础模型与 LoRA，验证 OpenAI 兼容推理接口
- 对知识库不存在的平台政策使用确定性闸门，避免模型编造期限、金额或承诺
- 为请求生成或透传 Request ID，并记录请求状态与耗时
- 对未处理异常返回不泄露内部信息的统一错误响应
- 启动时预热 RAG，并在检索故障时降级为空上下文；通过 `/ready` 区分存活、就绪和降级状态
- 提供 Docker Compose `app` profile，串联知识索引、数据库迁移、FastAPI、Nginx 与 Vue
- 使用自动化测试、前端生产构建和 GitHub Actions 验证核心流程


## 核心流程

```text
客户端
  ↓
JWT 认证 → FastAPI Router
  ↓
ConversationService
  ├── SQLAlchemy → MySQL
  ├── BM25 + BGE/Milvus + RRF → 公开购物澄清案例
  ├── PolicyGuardrail → 未知平台政策的确定性答复
  └── CustomerSupportAgent → OpenAI 兼容模型（DeepSeek / vLLM）
        ├── get_order_status → 参数校验 → 客户权限边界 → MySQL
        └── request_refund → 权限与金额规则 → 幂等写入 → 审计日志
```

在线应用刻意只保留四层：

- Router：接收请求并返回响应
- ConversationService：组织创建会话、构造上下文和保存消息
- Agent/Tool：循环处理模型工具请求，并执行受约束的业务查询
- Model/LLM Client：分别负责数据库映射和 OpenAI 兼容模型调用

## API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 健康检查 |
| `GET` | `/api/v1/ready` | 数据库与 RAG 就绪/降级状态 |
| `POST` | `/api/v1/auth/register` | 注册客户并获取访问令牌 |
| `POST` | `/api/v1/auth/login` | 登录并获取访问令牌 |
| `GET` | `/api/v1/auth/me` | 获取当前登录客户 |
| `POST` | `/api/v1/conversations` | 为当前登录客户创建会话 |
| `POST` | `/api/v1/conversations/{conversation_no}/chat` | 发送消息并获取模型回复 |
| `POST` | `/api/v1/conversations/{conversation_no}/chat/stream` | 通过 SSE 流式获取模型回复 |
| `GET` | `/api/v1/conversations/{conversation_no}` | 获取会话历史 |

启动服务后可访问 `http://127.0.0.1:8000/docs` 查看交互式 API 文档。
除健康检查、注册和登录外，在线接口均需发送 `Authorization: Bearer <token>`。
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

编辑 `customer-service-app/backend/.env`，让数据库密码与 `infra/.env` 保持一致。可选择云端
DeepSeek 或本地 vLLM，两者共用同一个客户端。

先生成本地 JWT 密钥并填入 `.env`：

```bash
openssl rand -hex 32
```

```ini
RETAIL_JWT_SECRET=粘贴上一步生成的随机字符串
```

DeepSeek 配置示例：

```ini
RETAIL_LLM_API_BASE=https://api.deepseek.com
RETAIL_LLM_API_KEY=你的密钥
RETAIL_LLM_MODEL=deepseek-flash
RETAIL_LLM_TIMEOUT_SECONDS=60
RETAIL_LLM_MAX_RETRIES=1
```

本地 vLLM 配置示例：

```ini
RETAIL_LLM_API_BASE=http://127.0.0.1:8001/v1
RETAIL_LLM_API_KEY=local-vllm
RETAIL_LLM_MODEL=retail-agent
```

`.env` 已被 Git 忽略，不要把 API Key 提交到仓库。
从旧版本迁移的客户没有密码哈希，不能直接登录；本地演示时请注册一个新的客户编号。

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

也可以从仓库根目录运行统一检查：

```bash
make check
```

前端测试和生产构建：

```bash
cd ../frontend
npm test
npm run build
```

## Docker Compose 完整栈

只执行 `docker compose up` 时仍只启动 MySQL，方便日常开发；`app` profile 会依次建立 Milvus Lite
索引、执行 Alembic 迁移、启动 FastAPI，并用 Nginx 托管 Vue 和反向代理 SSE：

```bash
cp infra/.env.example infra/.env
# 编辑 infra/.env，至少替换 RETAIL_JWT_SECRET，并配置 RETAIL_LLM_API_KEY
docker compose --env-file infra/.env -f infra/compose.yaml --profile app up --build
```

首次启动需要下载固定版本的 BGE Embedding 与 Reranker，耗时取决于网络；模型缓存、向量库和 MySQL
数据分别保存在具名 volume 中，后续启动会复用。启动完成后访问 `http://127.0.0.1:5173`，后端文档为
`http://127.0.0.1:8000/docs`。停止服务：

```bash
docker compose --env-file infra/.env -f infra/compose.yaml --profile app down
```

如果 vLLM 与 Docker 运行在同一台主机，可将接口地址设置为
`http://host.docker.internal:8001/v1`；当前两台设备不在可互通网络时，使用 DeepSeek API，GPU 服务仍
按 `model-serving/README.md` 独立运行。

## 流式响应协议

Vue 使用 `fetch` 读取 `/chat/stream` 返回的 Server-Sent Events。一次完整回复依次包含：

```text
sources → delta... → done
                  ↘ error
```

- `sources`：本轮检索到的知识来源。
- `delta`：模型生成的文本增量，前端收到后立即追加显示。OpenAI 兼容客户端会在同一次流式
  请求中重组工具调用参数，Agent 执行工具后继续接收最终回答，不会为了流式展示重复请求模型。
  确定性的政策兜底不调用模型，因此只发送一个完整 `delta`。
- `done`：完整回复已写入数据库，携带正式消息编号。
- `error`：流式生成失败；失败时用户消息和助手消息都不会写入数据库。

原有 `/chat` JSON 接口继续保留，用作非流式调用和回归基线。

## 在线可观测性

流式接口使用独立的 `app.chat_stream` 结构化日志，记录：

- `ttft_ms`：从接口开始处理到第一个文本增量的时间，包含鉴权、数据库读取和 RAG；
- `total_ms`：完整流结束所需时间；
- `delta_count` 和 `character_count`：本次输出的增量数与字符数；
- `status`：`completed`、`model_error`、`incomplete` 或 `interrupted`。

没有产生文本时 `ttft_ms=-1`。普通请求日志只能反映 HTTP 响应建立时间，分析流式体验时应以
`app.chat_stream` 为准。工具执行使用 `app.agent` 日志记录工具名、成功或拒绝状态、错误码和耗时，
但不会记录订单号、退款原因等业务参数。

离线 Agent 评测还会把失败划分为 `missing_tool`、`unexpected_tool`、`wrong_tool`、
`multiple_tools`、`invalid_arguments` 和 `wrong_arguments`，避免只看一个总准确率而不知道应该
修改提示词、参数 Schema 还是业务工具。

应用启动时默认预热 Embedding、Milvus 和 Reranker，避免首位用户承担模型初始化时间。RAG 初始化
或在线检索失败时，运行时会关闭异常客户端、标记为 `degraded` 并返回空知识上下文；聊天、政策闸门
和订单工具仍可继续工作。`/health` 只表示进程存活，`/ready` 会实际执行数据库查询并报告 RAG 为
`ready`、`lazy` 或 `degraded`。测试环境不预热，可用 `RETAIL_RAG_WARMUP_ENABLED=false` 显式关闭。

## 目录说明

```text
customer-service-app/
├── backend/
│   ├── data/             # 公开数据转换结果、署名和许可证
│   ├── app/              # 仅包含在线请求所需的运行时代码
│   │   ├── agents/       # Agent 循环、工具定义、参数校验和受权执行
│   │   ├── api/          # HTTP 路由和依赖
│   │   ├── core/         # 配置、数据库连接、JWT 和密码安全
│   │   ├── llm/          # OpenAI 兼容模型客户端
│   │   ├── middleware/   # 请求日志等通用请求处理
│   │   ├── models/       # SQLAlchemy 数据表映射
│   │   ├── rag/          # 在线 BM25、Milvus 向量检索和 RRF 融合
│   │   ├── schemas/      # 请求和响应结构
│   │   ├── policies.py   # 不依赖模型的业务边界规则
│   │   ├── prompts.py    # 在线与评测共享的系统提示词
│   │   └── services/     # 认证和聊天业务流程编排
│   ├── benchmarks/       # Agent 离线评测代码、固定用例和版本化结果
│   ├── migrations/       # Alembic 数据库迁移
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
model-pipeline/
├── model_pipeline/        # SFT 数据治理、脱敏和 LLaMA-Factory 导出
├── scripts/               # SFT 数据准备与导出入口
├── data/reports/          # 输入版本、处理统计、输出哈希和实验指标
└── tests/                 # 数据质量、可复现性和防泄漏测试
model-serving/              # vLLM + LoRA 启动脚本、配置和冒烟测试
infra/                 # 本地 MySQL Docker Compose
.github/workflows/     # 后端、离线流水线和前端 CI
```

## 公开案例知识库

`customer-service-app/backend/data/knowledge.json` 来自 MIT 许可的
[Chinese Ambiguous Reference](https://github.com/ygan/Chinese-Ambiguous-Reference)，固定到 commit
`3fe3a42233571914ee4a5ccaec3f3af97966d9c9`。原始 1104 条记录经确定性去重和空记录清理后得到
1073 条购物澄清案例；每条记录包含稳定来源编号、原始日期、来源地址、数据版本和许可证。

这些内容是真实购物对话样本，但不是权威平台业务规则。系统只用它们学习“信息不足时如何澄清”，
不能据此回答退款期限、物流状态、赔付条件或法律问题。署名和采用决定见
[第三方数据说明](customer-service-app/backend/data/THIRD_PARTY_DATA.md)与
[数据集登记与采用规则](knowledge-pipeline/data/README.md)及
[机器可读登记表](knowledge-pipeline/data/dataset_registry.json)。

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

混合检索之后默认只把前 4 个候选交给 `BAAI/bge-reranker-v2-m3` CrossEncoder 重排。模型与本地
Travel Agent 项目保持一致，并固定到 commit `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，
避免上游更新改变实验结果。Reranker 会同时阅读问题和完整候选文本，适合区分“同样是手机，
但用户要 512G 赠品还是游戏续航”这类难负例。在 11 条独立难负例上，重排前后的 Hit@3
都是 `1.000`，MRR 从 `0.864` 提升到 `1.000`；代价是比向量检索慢，因此不会对 1073 条
文档逐条打分。在 Apple M4 本地复现实验中，候选数从 8 减至 4 后，11 条查询的重排检索总耗时
从 `7.024s` 降至 `3.746s`（减少 `46.7%`），固定评测指标不变。该小样本结果用于选择教学项目的
默认值，不代表生产环境延迟承诺。可用
`RETAIL_RAG_RERANKER_ENABLED=false` 关闭，并用 `RETAIL_RAG_RERANKER_CANDIDATES` 调整候选数。

每次在线检索会输出一条 `rag_search_finished` 日志，分别记录 Embedding、Milvus、BM25、RRF、
Reranker 和总耗时，以及各阶段候选数量。日志只记录查询字符数，不记录原始问题，避免把订单号等
用户内容写入日志。一次热查询实测中，Milvus 为 `2.74ms`，Reranker 为 `479.03ms`，因此本轮优先
缩减重排候选数，而没有替换并非瓶颈的 Milvus。

## 启动前端

先在 `customer-service-app/backend/` 启动后端，再打开另一个终端：

```bash
cd customer-service-app/frontend
npm install
npm run dev
```

访问 `http://127.0.0.1:5173`。Vite 会将 `/api` 请求代理到本机 `8000` 端口，因此本地开发不需要额外配置 CORS。

页面先注册或登录，再用访问令牌创建会话。客户编号不再由创建会话接口接收，后端只信任 JWT 中的
身份并在数据库查询中追加 `customer_id`，因此知道别人的会话编号也无法读取其聊天记录。

## 关键技术选型

- **JWT + Argon2**：当前只有客户登录和会话隔离，不需要引入完整 OAuth 服务；PyJWT 边界清晰，
  Argon2 适合密码哈希。若未来接入企业 SSO、多端注销或第三方授权，再替换为 OIDC/Keycloak。
- **FastAPI + SQLAlchemy Async**：与项目的 Python/模型生态一致，接口类型和异步 I/O 支持较好；
  如果团队以 Java 为主且业务规则远多于 AI 流程，Spring Boot 会更合适。
- **MySQL**：订单、退款和审计数据需要事务及约束，关系数据库比 MongoDB 更匹配；若从零部署并
  希望同时使用 `pgvector` 简化组件数量，可改为 PostgreSQL，但当前已有 Milvus，不值得只为统一而迁移。
- **Milvus Lite → Milvus Standalone**：本地文件模式便于学习，服务端模式适合数据增长和并发，二者
  共用客户端接口；知识量长期只有几千片段时，继续使用 Lite 或改用 pgvector 都比维护集群更实际。
- **Vue 3 + Vite**：页面较轻、上手成本低、构建快；若需要 SSR、内容站或大型 React 团队协作，
  再考虑 Nuxt/Next.js。当前没有这些需求。
- **自定义 Function Calling Agent**：只有订单查询和退款两个工具，显式循环更容易解释与测试；
  只有出现多分支状态图、人工审批和长任务恢复时，才值得引入 LangGraph。

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
python scripts/evaluate_agent.py --output benchmarks/results/deepseek-baseline.json
```

测试集、逐例结果、失败分析和配置指纹见
[Agent 评测说明](customer-service-app/backend/benchmarks/README.md)。

## 政策防幻觉边界

当前公开知识库只有购物澄清案例，不包含权威退款、退货、赔付或保修规则。仅靠提示词或微调，
模型仍可能生成看似合理但没有依据的具体数字。因此 `ConversationService` 在调用 RAG 和 LLM 之前
执行一个小型确定性政策闸门：命中知识库未覆盖的政策问题时，不调用模型、不返回无关引用，统一
提示用户查看订单页规则或联系人工客服。固定测试同时验证普通商品咨询、具体订单查询和退款操作
不会被误拦截。

这不是完整政策系统；当项目以后接入版本化的权威规则库时，应逐条用可引用的规则替代这些兜底。

## QLoRA 与模型部署实验

在一台配备两张 RTX 2080 Ti（每张 11 GB）的 Ubuntu 主机上，项目以单卡 4-bit NF4 QLoRA
微调 `Qwen/Qwen3-4B-Instruct-2507`。第一轮训练使用 1910 条训练样本、253 条开发样本和
270 条测试样本；第二轮加入 40 条人工业务边界对话，并通过重复采样导出 160 行纠偏数据。

| 实验 | Dev loss | Test loss | 说明 |
| --- | ---: | ---: | --- |
| 未微调基础模型 | — | 6.7815 | 固定测试集基线 |
| 澄清 QLoRA | 1.8574 | 1.8193 | 最佳 checkpoint 为 step 200 |
| 一轮业务边界纠偏 | 1.8745 | 1.8353 | 低学习率、1 epoch |

12 条人工业务边界回归中，严格通过 6 条、部分通过 4 条、失败 2 条。这不是准确率声明，而是明确
揭示模型边界的行为检查。完整指标和失败案例见
[QLoRA 实验报告](model-pipeline/data/reports/qlora_experiments.json)。

vLLM 0.32.0 已在 GPU 主机上成功加载 FP16 基础模型和纠偏 LoRA，`/v1/models` 与
`/v1/chat/completions` 均已验证。启动脚本、冒烟测试和网络限制见
[模型服务说明](model-serving/README.md)。

