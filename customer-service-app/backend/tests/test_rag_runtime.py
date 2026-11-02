from app.rag.runtime import KnowledgeRuntime


class FakeRetriever:
    def __init__(self, *, fail_search: bool = False) -> None:
        self.fail_search = fail_search
        self.queries: list[tuple[str, int]] = []
        self.closed = False

    def search(self, query: str, *, limit: int = 2) -> list:
        self.queries.append((query, limit))
        if self.fail_search:
            raise RuntimeError("search failed")
        return []

    def close(self) -> None:
        self.closed = True


def test_runtime_initializes_once_and_warms_up() -> None:
    retriever = FakeRetriever()
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return retriever

    runtime = KnowledgeRuntime(factory)

    assert runtime.status == "not_initialized"
    assert runtime.warmup() is True
    assert runtime.search("第二次查询") == []
    assert factory_calls == 1
    assert retriever.queries == [("商品咨询", 1), ("第二次查询", 2)]
    assert runtime.status == "ready"

    runtime.close()
    assert retriever.closed is True
    assert runtime.status == "closed"


def test_runtime_caches_initialization_failure_as_degraded() -> None:
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        raise RuntimeError("cannot load index")

    runtime = KnowledgeRuntime(factory)

    assert runtime.search("第一次查询") == []
    assert runtime.search("第二次查询") == []
    assert factory_calls == 1
    assert runtime.status == "degraded"


def test_runtime_degrades_when_live_search_fails() -> None:
    retriever = FakeRetriever(fail_search=True)
    runtime = KnowledgeRuntime(lambda: retriever)

    assert runtime.search("触发检索异常") == []
    assert runtime.status == "degraded"
    assert retriever.closed is True
