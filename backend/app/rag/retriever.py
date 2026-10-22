from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeDocument:
    source_id: str
    title: str
    content: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedDocument:
    source_id: str
    title: str
    content: str
    score: int


DEFAULT_KNOWLEDGE_DOCUMENTS = (
    KnowledgeDocument(
        source_id="return-policy",
        title="七天无理由退货规则",
        content=(
            "符合完好标准的普通商品，可在签收次日起七天内申请无理由退货。"
            "定制、生鲜和已拆封后影响安全或品质的商品不适用。"
        ),
        keywords=("退货", "退换", "七天", "7天", "签收", "无理由"),
    ),
    KnowledgeDocument(
        source_id="shipping-query",
        title="物流查询说明",
        content=(
            "查询物流时需要提供订单号，或下单账号等可定位订单的信息。"
            "客服确认订单后才能返回承运商和物流轨迹。"
        ),
        keywords=("物流", "快递", "配送", "发货", "订单号", "到哪"),
    ),
    KnowledgeDocument(
        source_id="refund-timing",
        title="退款到账时间",
        content=(
            "退款审核通过后会原路退回。平台处理通常需要一至三个工作日，"
            "实际到账时间还取决于银行或支付机构。"
        ),
        keywords=("退款", "到账", "退钱", "工作日", "原路退回"),
    ),
)


class KnowledgeRetriever:
    def __init__(
        self,
        documents: Sequence[KnowledgeDocument] = DEFAULT_KNOWLEDGE_DOCUMENTS,
    ) -> None:
        self._documents = tuple(documents)

    def search(self, query: str, *, limit: int = 2) -> list[RetrievedDocument]:
        normalized_query = query.casefold()
        results: list[RetrievedDocument] = []

        for document in self._documents:
            matched_keywords = [
                keyword for keyword in document.keywords if keyword.casefold() in normalized_query
            ]
            if not matched_keywords:
                continue

            results.append(
                RetrievedDocument(
                    source_id=document.source_id,
                    title=document.title,
                    content=document.content,
                    score=sum(len(keyword) for keyword in matched_keywords),
                )
            )

        results.sort(key=lambda document: (-document.score, document.source_id))
        return results[:limit]
