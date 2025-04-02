import os
import pickle
import time
from hashlib import md5
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import faiss
import numpy as np
import openai


class FileHashManager:
    """文件哈希状态管理器"""

    def __init__(self, hash_db_path: str = "file_hashes2.pkl"):
        if os.getenv("FILE_MANAGER_DIR"):
            file_manager_dir = os.getenv("FILE_MANAGER_DIR")
        else:
            file_manager_dir = "."
        self.hash_db_path = os.path.join(file_manager_dir, hash_db_path)
        self._pending_changes = False
        self.hash_db_path = Path(self.hash_db_path)
        self.file_hashes = self._load_hashes()

    def _load_hashes(self) -> Dict[str, str]:
        """加载哈希数据库"""
        if self.hash_db_path.exists():
            with open(self.hash_db_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_hashes(self):
        """持久化哈希状态"""
        with open(self.hash_db_path, 'wb') as f:
            pickle.dump(self.file_hashes, f)

    def check_modified(self, file_path: str) -> bool:
        """检查文件是否修改"""
        file_path = str(file_path)
        if not os.path.exists(file_path):
            return False

        with open(file_path, 'rb') as f:
            current_hash = md5(f.read()).hexdigest()

        return current_hash != self.file_hashes.get(file_path, "")

    def update_hash(self, file_path: str):
        """更新文件哈希记录"""
        with open(file_path, 'rb') as f:
            content = f.read()
            self.file_hashes[str(file_path)] = md5(content).hexdigest()

    def remove_hash(self, file_path: str):
        """删除哈希记录"""
        if str(file_path) in self.file_hashes:
            del self.file_hashes[str(file_path)]
            self._pending_changes = True

    def sync_with_filesystem(self, root_dir: str):
        """同步数据库与实际文件系统"""
        root = Path(root_dir)
        existing_files = {str(p) for p in root.rglob('*') if p.is_file()}

        # 删除已经不存在的文件记录
        for file_path in list(self.file_hashes.keys()):
            if file_path not in existing_files:
                self.remove_hash(file_path)


class FileLevelIndexer:
    def __init__(self,
                 model=None,
                 index_path: str = "file_index.faiss",
                 hash_db_path: str = "file_hashes2.pkl"):
        """
        初始化参数
        :param model: 嵌入模型（需实现encode方法）
        :param index_path: 索引存储路径
        :param hash_db_path: 哈希数据库路径
        """
        self.model = model
        self.index_path = Path(index_path)
        self.hash_manager = FileHashManager(hash_db_path)
        self.file_map: Dict[int, Tuple[str, str]] = {}  # {idx: (file_path, content)}
        self.index = None

    def scan_files(self, root_dir: str = "./knowledge_base") -> 'FileLevelIndexer':
        """改进版扫描：保留未修改文件的记录"""
        root = Path(root_dir).expanduser()

        for md_file in root.rglob("*.md"):
            file_path = str(md_file)
            is_modified = self.hash_manager.check_modified(file_path)

            # 获取文件内容逻辑
            if is_modified:
                with open(md_file, 'r', encoding='utf-8') as f:
                    self.hash_manager.update_hash(file_path)

        return self

    def build_index(self) -> 'FileLevelIndexer':
        """构建/更新语义索引"""
        # 生成嵌入向量
        contents = [v[1] for v in self.file_map.values()]
        embeddings = self.model.encode(contents) if self.model else np.random.rand(len(contents), 384)

        # 初始化或更新FAISS索引
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.reset()
        self.index.add(embeddings.astype('float32'))

        # 持久化索引和哈希状态
        self._save_resources()
        return self

    def _save_resources(self):
        """保存索引和哈希状态"""
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
        self.hash_manager.save_hashes()

    def load_index(self) -> 'FileLevelIndexer':
        """加载已有索引"""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        return self

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """语义搜索"""
        query_embed = self.model.encode([query]) if self.model else np.random.rand(1, 384)
        distances, indices = self.index.search(query_embed.astype('float32'), top_k)

        return [self.file_map[idx] for idx in indices[0] if idx in self.file_map]


class ProxyOpenAIIndexer(FileLevelIndexer):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model_name: str = "text-embedding-3-small", proxy_url: Optional[str] = None):
        super().__init__()
        self.api_key = api_key
        self.api_model = model_name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=proxy_url if proxy_url else base_url
        )

    def encode(self, queries: List[str]) -> np.ndarray:
        """实现标准encode接口"""
        embeddings = self._batch_embedding(queries)
        return np.array(embeddings).astype('float32')

    def _batch_embedding(self, texts: List[str]) -> List[List[float]]:
        """带代理支持的嵌入生成"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.api_model,
                    timeout=30
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                print(e)
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return [[]] * len(texts)

    def build_index(self):
        texts = [v[1] for v in self.file_map.values()]
        embeddings = self._batch_embedding(texts)

        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.reset()
        self.index.add(np.array(embeddings).astype('float32'))
        return self


class PersistentIndexer(ProxyOpenAIIndexer):
    def __init__(self, index_file="saved_index2.faiss", **kwargs):
        if os.getenv("FAISS_INDEX_DIR"):
            faiss_index_dir = os.getenv("FAISS_INDEX_DIR")
        else:
            faiss_index_dir = ".."
        os.makedirs(faiss_index_dir, exist_ok=True)
        index_file = os.path.join(faiss_index_dir, index_file)
        super().__init__(**kwargs)
        self.index_file = Path(index_file)
        self.metadata_file = Path(f"{index_file}.meta")

    def build_index(self):
        """构建或加载已有索引"""
        if self.index_file.exists() and self.metadata_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, 'rb') as f:
                self.file_map = pickle.load(f)
        else:
            super().build_index()
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.file_map, f)
        return self


class OptimizedIndexer2(PersistentIndexer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash_manager = FileHashManager()

    def scan_files(self, root_dir='./knowledge_base'):
        new_file_map = {}
        root = Path(os.path.expanduser(root_dir))

        # 先同步文件系统状态
        self.hash_manager.sync_with_filesystem(root_dir)

        for md_file in root.rglob('*.md'):
            file_path = str(md_file)
            # 检查文件是否修改
            if not self.hash_manager.check_modified(file_path):
                continue

            # 处理新文件/修改过的文件
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()[:7000]
                self.hash_manager.update_hash(file_path)

                new_file_map[len(new_file_map)] = (file_path, content)

        self.file_map = new_file_map
        self.hash_manager.save_hashes()
        return self

    def build_index(self):
        try:
            super().build_index()
            self._save_index()
        finally:
            # 确保索引构建完成后保存哈希状态
            self.hash_manager.save_hashes()

    def _save_index(self):
        """保存索引和元数据"""
        faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.file_map, f)
