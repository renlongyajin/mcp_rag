import os
import pickle
import time
from pathlib import Path
from typing import List, Optional
import faiss
import numpy as np
import openai

from FileHashManager import FileHashManager


class KnowledgeIndexer:
    def __init__(self):
        self.model = None  # 将在子类中初始化
        self.index = None
        self.file_map = {}  # {idx: (file_path, chunk_content)}
        self.chunk_size = 1500
        self.overlap = 300
        self.file_hashes = {}  # 用于增量更新

    def split_into_chunks(self, text):  # 改为双下划线或改为公共方法
        """按指定大小和重叠分割文本"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.overlap)
        return chunks

    def scan_files(self, root_dir='./knowledge_base'):
        """基础扫描方法"""
        self.file_map.clear()
        root = Path(os.path.expanduser(root_dir))
        for md_file in root.rglob('*.md'):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            for chunk in self.split_into_chunks(content):
                self.file_map[len(self.file_map)] = (str(md_file), chunk)
        return self

    def build_index(self):
        """基础索引构建方法"""
        texts = [v[1] for v in self.file_map.values()]
        embeddings = self.model.encode(texts) if self.model else np.zeros((len(texts), 384))

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.reset()
        self.index.add(np.array(embeddings).astype('float32'))
        return self


class ProxyOpenAIIndexer(KnowledgeIndexer):
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
    def __init__(self, index_file="saved_index.faiss", **kwargs):
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


class OptimizedIndexer(PersistentIndexer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash_manager = FileHashManager()

    def scan_files(self, root_dir='./knowledge_base'):
        """优化后的文件扫描"""
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
                content = f.read()
                self.hash_manager.update_hash(file_path, content)

                for chunk in self.split_into_chunks(content):
                    new_file_map[len(new_file_map)] = (file_path, chunk)

            # 定期自动保存哈希数据库（例如每处理100个文件）
            if len(new_file_map) % 100 == 0:
                self.hash_manager.save_hashes()

        self.file_map = new_file_map
        return self

    def build_index(self):
        try:
            super().build_index()
        finally:
            # 确保索引构建完成后保存哈希状态
            self.hash_manager.save_hashes()

    def _save_index(self):
        """保存索引和元数据"""
        faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, 'wb') as f:
            pickle.dump({
                'file_map': self.file_map,
                'file_hashes': self.file_hashes
            }, f)


class APIQueryIndexer(ProxyOpenAIIndexer):
    def encode(self, queries: List[str]) -> np.ndarray:
        """重写编码方法"""
        embeds = self._batch_embedding(queries)
        return np.array(embeds).astype('float32')


