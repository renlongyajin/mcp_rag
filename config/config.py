import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class KnowledgeBase:
    knowledge_name: str
    knowledge_dir: str
    description: str


class GlobalConfig:
    def __init__(self):
        self.knowledge_base: Optional[KnowledgeBase] = None
        self.knowledge_base_config = {}
        self.all_knowledge_base_config = {}
        curPath = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(curPath, "knowledge_base.json"), "r+", encoding='utf8') as f:
            self.all_knowledge_base_config = json.load(f)

    def init_knowledge_base(self, knowledge_name: str):
        if knowledge_name == "" or knowledge_name is None:
            raise Exception(f"请输入要启动的知识库的名称！")
        if knowledge_name in self.all_knowledge_base_config:
            self.knowledge_base_config = self.all_knowledge_base_config[knowledge_name]
            self.knowledge_base_config["knowledge_name"] = knowledge_name
            self.knowledge_base = KnowledgeBase(**self.knowledge_base_config)
            return self.knowledge_base
        else:
            raise Exception(f"不存在名称为{knowledge_name}的知识库！请前往knowledge_base.json注册。")


global_config = GlobalConfig()
