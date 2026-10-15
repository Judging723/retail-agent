# app/middleware/request_logging.py
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import Request

logger = logging.getLogger("app.request")


async def request_logging_middleware(request: Request, call_next):
    # 1. 读取 X-Request-ID 请求头，没有则生成 uuid
    request_id = request.headers.get("X-Request-ID")

    if not request_id:
        request_id = str(uuid4())

    request.state.request_id = request_id

    # 2. 记录开始时间
    start_time = perf_counter()

    try:
        # 3. 调用后续链路
        response = await call_next(request)

        # 4. 计算耗时，转成毫秒
        duration_ms = (perf_counter() - start_time) * 1000

        # 5. 把 request_id 写入响应头
        response.headers["X-Request-ID"] = request_id

        # 6. 记录正常完成日志，使用参数化，禁止f-string
        logger.info(
            "request_completed request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    except Exception:
        # 捕获异常，自动打印堆栈，不吞异常，最后继续抛出
        duration_ms = (perf_counter() - start_time) * 1000
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise
