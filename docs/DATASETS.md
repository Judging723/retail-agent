# 公开数据集调研与使用边界

调研日期：2026-10-22。

## 先区分两类数据

公开对话、意图和情感数据集适合训练模型或建立评测集，但不等于知识库。知识库中的
退款期限、物流规则、赔付条件必须来自可追溯且有权使用的业务文档。当前项目继续使用
Chinese Ambiguous Reference 构建“购物澄清案例库”，用于检索相似案例和设计追问；它不用于
回答退款、物流、赔付等平台规则问题。

## 推荐顺序

| 数据集 | 适合任务 | 许可/访问 | 当前决定 |
| --- | --- | --- | --- |
| [Chinese Ambiguous Reference](https://github.com/ygan/Chinese-Ambiguous-Reference) | 购物咨询歧义检测、澄清策略 | MIT；可直接访问 | 第一批实验数据 |
| [CrossWOZ](https://github.com/thu-coai/CrossWOZ) | 意图、槽位、对话状态 | Apache-2.0；可直接访问 | 用于学习 NLU 格式和迁移基线，不冒充电商数据 |
| [MASSIVE](https://github.com/alexa/massive) | 中文意图分类、槽位抽取 | CC BY 4.0；需要署名 | 可作为通用 NLU 基线 |
| [Amazon ESCI](https://github.com/amazon-science/esci-data) | 查询与商品相关性、Reranker | Apache-2.0；可直接访问 | 可选的相关性学习数据，中文客服相关性较弱 |
| [JDDC 2.1](https://github.com/hrlinlp/jddc2.1) | 客服生成、改写、总结、多模态 | 需要机构邮箱申请 | 获批并复核条款后再使用 |
| [E-commerce Dialogue Corpus](https://github.com/cooelf/DeepUtteranceAggregation) | 多轮回复选择 | 仓库未提供清晰许可证 | 暂不下载、训练或再分发 |
| [CSDS](https://github.com/xiaolinAndy/CSDS) | 客服对话总结 | 仓库未提供清晰许可证 | 暂不下载、训练或再分发 |
| [FewCLUE EPRSTMT](https://github.com/CLUEbenchmark/FewCLUE) | 电商评论情感分类 | 数据许可仍需确认 | 确认许可后再作为小样本基线 |
| [SMP ECDT 镜像](https://github.com/hml-ubt/SMP2017-2019-ECDT-data) | 意图和领域分类 | 第三方镜像，许可不清晰 | 优先寻找官方来源，当前不使用 |

机器可读的决策记录位于
`knowledge-pipeline/data/dataset_registry.json`。下载得到的原始数据放在本地
`knowledge-pipeline/data/raw/`，处理产物放在 `knowledge-pipeline/data/processed/`，两者均不提交
Git。这样可以保留可复现实验说明，同时避免误提交大文件、用户对话和授权不明的数据。

## 第一批实施数据

Chinese Ambiguous Reference 已先完成一条可运行闭环：

1. 固定源代码 commit 和原始文件 SHA-256，保留 MIT 许可证。
2. 将 1104 条原始记录确定性转换为 1073 条去重、非空案例。
3. 使用 BGE 生成向量并写入 Milvus Lite。
4. FastAPI 检索相似案例，提示 LLM 借鉴澄清方式，并返回可点击来源。
5. 用固定检索集验证 Hit@K 和 MRR。

下一阶段再从该数据生成意图/歧义分类训练集，建立“直接回答或先澄清”的规则基线和
轻量模型基线。

CrossWOZ 和 MASSIVE 留到 NLU 阶段；ESCI 留到 Reranker 阶段；需要申请或许可不清晰的
数据集不进入当前开发路径。
