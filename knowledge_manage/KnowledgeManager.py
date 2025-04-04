import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class KnowledgeBase:
    knowledge_name: str
    knowledge_dir: str
    description: str
    chunk_size: int
    chunk_overlap: int


class KnowledgeManager:
    def __init__(self):
        knowledge_json_name = r"knowledge_base.json"
        self.cur_path = os.path.abspath(os.path.dirname(__file__))
        self.knowledge_json_path = os.path.join(self.cur_path, knowledge_json_name)
        self.knowledge_base: Optional[KnowledgeBase] = None
        self.knowledge_base_config = {}
        self.all_knowledge_base_config = {}
        self.all_knowledge_base: Optional[list[KnowledgeBase]] = []
        self.load_knowledge_base()
        knowledge_name = os.getenv("KNOWLEDGE_BASE_NAME")
        if knowledge_name == "" or knowledge_name is None:
            raise Exception("请在.env中写上一个默认的知识库再启动！")
        self.init_knowledge_base(knowledge_name)

    def _dict2knowledge(self, knowledge_name: str):
        knowledge_base_config = self.all_knowledge_base_config[knowledge_name]
        knowledge_base_config["knowledge_name"] = knowledge_name
        knowledge_base = KnowledgeBase(**knowledge_base_config)
        return knowledge_base

    def load_knowledge_base(self):
        self.all_knowledge_base = []
        with open(self.knowledge_json_path, "r+", encoding='utf8') as f:
            self.all_knowledge_base_config = json.load(f)
            for knowledge_name in self.all_knowledge_base_config:
                knowledge_base = self._dict2knowledge(knowledge_name)
                self.all_knowledge_base.append(knowledge_base)

    def write_knowledge_base(self):
        with open(self.knowledge_json_path, "w+", encoding='utf8') as f:
            json.dump(self.all_knowledge_base_config, f, ensure_ascii=False, indent=4)

    def init_knowledge_base(self, knowledge_name: str):
        if knowledge_name == "" or knowledge_name is None:
            raise Exception(f"请输入要启动的知识库的名称！")
        if knowledge_name in self.all_knowledge_base_config:
            self.knowledge_base = self._dict2knowledge(knowledge_name)
            return self.knowledge_base
        else:
            raise Exception(f"不存在名称为{knowledge_name}的知识库！请前往knowledge_base.json注册。")

    def create_knowledge_base(self, knowledge_name: str, knowledge_dir: str, description: str, chunk_size=1000,
                              chunk_overlap=200):
        if not os.path.exists(knowledge_dir):
            raise Exception(f"本地路径{knowledge_dir}不存在！")
        self.all_knowledge_base_config[knowledge_name] = {"knowledge_dir": knowledge_dir, "description": description,
                                                          "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                                                          "knowledge_name": knowledge_name}
        self.write_knowledge_base()

    def get_info(self, knowledge_base_name: str):
        if knowledge_base_name in self.all_knowledge_base_config:
            return self.all_knowledge_base_config[knowledge_base_name]
        return {}
