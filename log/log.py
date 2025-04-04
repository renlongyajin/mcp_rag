import logging
import sys
from logging.handlers import RotatingFileHandler

# 强制重置Uvicorn默认配置
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False
})

# 基础配置（必须在其他导入前）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            filename='log/log.txt',
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
    ],
    force=True
)

# 获取Uvicorn相关logger并重新配置
uvicorn_loggers = [
    logging.getLogger("uvicorn"),
    logging.getLogger("uvicorn.error"),
    logging.getLogger("uvicorn.access"),
    logging.getLogger("uvicorn.asgi")
]

for logger in uvicorn_loggers:
    logger.handlers = []  # 清除原有handler
    logger.propagate = True  # 启用向上传播
    logger.setLevel(logging.INFO)  # 强制设置级别

# FastAPI日志处理
fastapi_logger = logging.getLogger("fastapi")
fastapi_logger.propagate = True
fastapi_logger.handlers = []

# 验证配置
def check_uvicorn_loggers():
    for name in ["uvicorn", "uvicorn.access"]:
        logger = logging.getLogger(name)
        print(f"{name} handlers: {logger.handlers}")
        print(f"{name} propagate: {logger.propagate}")

# 在应用启动前调用
check_uvicorn_loggers()