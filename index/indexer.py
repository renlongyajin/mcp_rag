import os
import pickle
import time
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import openai

from FileHashManager import FileHashManager
from config.config import global_config
from log import logger
from utils import batch_embed


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
                content = f.read().encode("utf8")
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
        # 过滤非字符串和空文本
        valid_texts = [str(t).strip() for t in texts if t and str(t).strip()]
        if not valid_texts:
            return []

        # 初始化结果容器（维护与valid_texts相同的顺序）
        all_embeddings = []
        batches = batch_embed(valid_texts, model_name=self.api_model)
        batch_index = 0
        for batch in batches:
            batch_index += 1
            start_time = time.time()
            try:
                # for text in batch:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.api_model,
                    timeout=30
                )
                # 将当前批次的embedding添加到总列表
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(
                    f"[{batch_index}/{len(batches)}] 成功处理批次 {len(batch)} 条，耗时 {(time.time() - start_time):.2f}s")
            except Exception as e:
                logger.error(f"[{batch_index}/{len(batches)}] 批次处理失败：{str(e)}")
                all_embeddings.extend([[] for _ in batch])  # 保持长度一致
            # 最终验证（确保每个输入都有对应输出）
        if len(all_embeddings) != len(valid_texts):
            logger.error(f"结果数量不匹配！输入 {len(valid_texts)} 条，输出 {len(all_embeddings)} 条")
            return []

        return all_embeddings

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
        super().__init__(**kwargs)
        if os.getenv("FAISS_INDEX_DIR"):
            faiss_index_dir = os.getenv("FAISS_INDEX_DIR")
        else:
            faiss_index_dir = ".."
        faiss_index_dir = os.path.join(faiss_index_dir, global_config.knowledge_base.knowledge_name)
        os.makedirs(faiss_index_dir, exist_ok=True)
        index_file = os.path.join(faiss_index_dir, index_file)
        # self.index_file = os.fsencode(str(index_file)).decode("gbk")
        self.index_file = Path(index_file)
        self.metadata_file = Path(f"{index_file}.meta")

    def build_index(self):
        """构建或加载已有索引"""
        if os.path.exists(self.index_file) and self.metadata_file.exists():
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
                if not content.strip():  # 跳过无效文档
                    continue
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
