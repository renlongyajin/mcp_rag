from dotenv import load_dotenv

from knowledge_manage.KnowledgeManager import KnowledgeManager
from dotenv import load_dotenv
import httpx
import ssl

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
            # 初始化同步客户端
            self.sync_proxy_client = self._create_proxy_client()
            # 初始化异步客户端
            self.async_proxy_client = self._create_async_proxy_client()
            self.__class__._initialized = True  # 设置为已初始化

    def _create_proxy_client(self) -> httpx.Client:
        """创建同步代理客户端"""
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2  # 显式设置最低版本
        ssl_ctx.maximum_version = ssl.TLSVersion.TLSv1_2  # 锁定TLS 1.2
        ssl_ctx.set_ciphers('ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256')
        ssl_context = ssl.create_default_context()
        ssl_context.options |= ssl.OP_NO_TLSv1_3
        ssl_context.set_ciphers("ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256")

        return httpx.Client(
            transport=httpx.HTTPTransport(
                proxy="socks5://192.168.1.211:7891",
                verify=ssl_context,
                retries=3
            ),
            timeout=30.0,
            trust_env=False
        )

    def _create_async_proxy_client(self) -> httpx.AsyncClient:
        """创建异步代理客户端"""
        ssl_context = ssl.create_default_context()
        ssl_context.options |= ssl.OP_NO_TLSv1_3
        
        return httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(
                proxy="socks5://192.168.1.211:7891",
                verify=ssl_context,
                retries=3
            ),
            timeout=30.0,
            trust_env=False
        )
global_config = GlobalConfig()  # 单例实例
