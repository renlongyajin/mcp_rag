import os
import pickle
import time
from hashlib import md5
from pathlib import Path

from config.config import global_config


class FileHashManager:
    """高效的文件哈希状态管理器"""

    def __init__(self, hash_db_path: str = "./file_hashes.pkl"):
        if os.getenv("FILE_MANAGER_DIR"):
            file_manager_dir = os.getenv("FILE_MANAGER_DIR")
        else:
            file_manager_dir = "."
        file_manager_dir = os.path.join(file_manager_dir, global_config.knowledge_base.knowledge_name)
        os.makedirs(file_manager_dir, exist_ok=True)
        hash_db_path = os.path.join(file_manager_dir, hash_db_path)
        self.hash_db_path = Path(hash_db_path)
        self.file_hashes = self._load_hashes()
        self._pending_changes = False

    def _load_hashes(self) -> dict:
        """加载或初始化哈希数据库"""
        if self.hash_db_path.exists():
            with open(self.hash_db_path, 'rb') as f:
                return pickle.load(f)
        return {}  # 返回空字典

    def save_hashes(self):
        """保存变更到磁盘（需显式调用）"""
        if self._pending_changes:
            with open(self.hash_db_path, 'wb') as f:
                pickle.dump(self.file_hashes, f)
            self._pending_changes = False

    def update_hash(self, file_path: str, content: str = None):
        """更新文件哈希记录"""
        file_path = str(file_path)
        if content is None:
            with open(file_path, 'rb', encoding='utf8') as f:
                content = f.read()
        self.file_hashes[file_path] = {
            'hash': md5(content.encode("utf8")).hexdigest(),
            'mtime': time.time(),
            'size': len(content)
        }
        self._pending_changes = True

    def check_modified(self, file_path: str) -> bool:
        """检查文件是否被修改"""
        file_path = str(file_path)
        if file_path not in self.file_hashes:
            return True

        try:
            stat = os.stat(file_path)
            saved = self.file_hashes[file_path]

            # 三重验证：修改时间 → 文件大小 → 内容哈希
            if stat.st_mtime > saved['mtime'] + 1:  # 1秒缓冲防止精度问题
                return True
            if stat.st_size != saved['size']:
                return True

            # 前两个条件不满足时才计算哈希
            with open(file_path, 'rb') as f:
                return md5(f.read()).hexdigest() != saved['hash']

        except (FileNotFoundError, OSError):
            return True  # 文件不存在视为已修改（需要后续清理）

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
