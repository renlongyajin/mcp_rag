from dotenv import load_dotenv

from knowledge_manage.KnowledgeManager import KnowledgeManager

load_dotenv()


class GlobalConfig:
    _instance = None  # 类变量，用于保存单例实例
    _initialized = False  # 标记是否已初始化

    def __new__(cls, *args, **kwargs):
        # 如果实例不存在，则创建新实例
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 确保初始化逻辑只执行一次
        if not self._initialized:
            self.knowledge_manager = KnowledgeManager()
            self.__class__._initialized = True  # 设置为已初始化


global_config = GlobalConfig()  # 单例实例
