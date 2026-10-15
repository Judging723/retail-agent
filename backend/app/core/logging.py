import logging


def configure_logging(log_level: str) -> None:
    """
    使用Python标准logging库配置全局日志
    :param log_level: 日志等级字符串，如 "DEBUG", "INFO", "WARNING", "ERROR"
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
