# 公开数据集调研与使用边界

调研日期：2026-10-23。

## 先区分两类数据

公开对话、意图和情感数据集适合训练模型或建立评测集，但不等于知识库。知识库中的
退款期限、物流规则、赔付条件必须来自可追溯且有权使用的业务文档。当前项目继续使用
Chinese Ambiguous Reference 构建“购物澄清案例库”，用于检索相似案例和设计追问；它不用于
回答退款、物流、赔付等平台规则问题。

## 推荐顺序

| 数据集 | 适合任务 | 许可/访问 | 当前决定 |
| --- | --- | --- | --- |
| [Chinese Ambiguous Reference](https://github.com/ygan/Chinese-Ambiguous-Reference) | 购物咨询歧义检测、澄清策略 | MIT；可直接访问 | 第一批实验数据 |
| [Kapibala Sales Dialogues](https://huggingface.co/datasets/kapibala-ai/kapibala-sales-dialogues) | 中文顾客消息情感分类 | CC BY 4.0；需署名 | 合成语料、模型弱标签，用于情感训练闭环基线 |
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

当前已从该数据生成“直接回答或先澄清”的弱监督分类训练集，并建立规则基线和
轻量模型基线。

### 标注审计说明

数据仓库 README 给出了数字编码的文字解释，但固定版本 JSON 中部分编码的实际话语分布与该表
存在明显差异。例如代码 `2` 大量出现在客服澄清追问中，代码 `3` 大量出现在列举选项中。
SIGDIAL 论文确认核心策略包括 CQ、LIST、LCQ、IG 和 IAR，但没有给出 JSON 数字编码映射。

因此当前 NLU 数据不把数值编码包装成“直接人工标注的用户意图”，而是明确声明为弱监督任务：
根据下一组客服回复的原始策略编码，推导当前用户表达是否触发澄清。所有训练记录都保留原始编码，
冲突样本会被剔除，实验报告中也记录映射假设。原始数据不会被改写。

- [固定版本数据说明](https://github.com/ygan/Chinese-Ambiguous-Reference/tree/3fe3a42233571914ee4a5ccaec3f3af97966d9c9)
- [SIGDIAL 2024 论文](https://aclanthology.org/2024.sigdial-1.25/)

CrossWOZ 和 MASSIVE 留作后续细粒度意图/槽位实验；ESCI 留作扩展 Reranker 实验；需要申请或许可不清晰的
数据集不进入当前开发路径。

## 情感分类实施数据

Kapibala Sales Dialogues 固定到 commit
`c94eb4fbc32d6f73f15bdb3dfc84600afd809844`。当前流水线只选择中文顾客消息，并以发布的
`p_negative`、`p_neutral`、`p_positive` 最大概率生成三分类标签；完全相同的消息会去重，同一
`setting_id` 的不同实验分支只进入同一个数据划分。

这个选择解决了“来源和许可证可核验”的问题，但不等于获得了真实人工金标：对话是合成的，情感概率
也由 `lxyuan/distilbert-base-multilingual-cased-sentiments-student` 生成。因此实验结果只能表述为
“在合成对话上复现教师伪标签”，不能宣传为真实客服情绪识别准确率。后续接入经过授权和人工复核的
业务样本时，应建立独立测试集重新评测。
