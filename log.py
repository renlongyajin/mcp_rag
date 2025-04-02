import logging
import sys

# 配置全局日志（只需执行一次）
logging.basicConfig(
    level=logging.INFO,  # 设置默认级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 输出到控制台
    ]
)

# 后续获取logger的代码保持不变
logger = logging.getLogger(__name__)
