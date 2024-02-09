import os
from typing import *

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS

from configs.faiss_config import data_dir
from configs.model_config import EMBEDDING_MODEL
from utils.embedding_util import load_embedding


class FaissManager:
    def __init__(
            self,
            embedding: Embeddings,
    ):
        self.cache = {}
        self.embedding = embedding
        self.load_cache()

    def value_list2vector_base(
            self,
            prop_dir: str,
            value_list: List[str]
    ):
        vb = FAISS.from_texts(value_list, self.embedding)
        vb.save_local(prop_dir)

    def load_vector_base(
            self,
            prop_dir: str,
    ) -> FAISS:
        vb = FAISS.load_local(prop_dir, self.embedding)
        return vb

    def load_cache(
            self,
    ):
        self.cache = {}
        if not os.path.exists(data_dir):
            return
        database_list = os.listdir(data_dir)
        for database in database_list:
            self.cache[database] = {}
            database_dir = os.path.join(data_dir, database)
            label_list = os.listdir(database_dir)
            label_cnt = 0
            for label in label_list:
                label_cnt += 1
                print(f"Loading label {label} {label_cnt}/{len(label_list)}...")
                self.cache[database][label] = {}
                label_dir = os.path.join(database_dir, label)
                prop_list = os.listdir(label_dir)
                prop_cnt = 0
                for prop in prop_list:
                    prop_cnt += 1
                    print(f"Loading prop {prop} {prop_cnt}/{len(prop_list)}...")
                    if os.listdir(prop_dir := os.path.join(label_dir, prop)):
                        self.cache[database][label][prop] = self.load_vector_base(prop_dir)
        print(self.cache)

    def search_vector_base(
            self,
            database: str,
            label: str,
            prop: str,
    ) -> Optional[FAISS]:
        database_dict = self.cache.get(database)
        if isinstance(database_dict, dict):
            label_dict = database_dict.get(label)
            if isinstance(label_dict, dict):
                vb = label_dict.get(prop)
                if isinstance(vb, FAISS):
                    return vb
        return None

    def search_repair_list(
            self,
            database,
            schema_dict: Dict[str, Any],
    ) -> List[List[Union[str, List[str]]]]:
        # 储存str:index
        repair_dict = {}
        # 储存[str,[str]]
        repair_list = []
        for label in schema_dict.keys():
            prop_value_dict = schema_dict.get(label)
            if isinstance(prop_value_dict, dict):
                for prop in prop_value_dict.keys():
                    value_list = prop_value_dict.get(prop)
                    if isinstance(value_list, list):
                        vb = self.search_vector_base(database, label, prop)
                        if not vb:
                            continue
                        for value in value_list:
                            result_list = vb.similarity_search(value, 3)
                            if not repair_dict.keys().__contains__(value):
                                repair_dict[value] = len(repair_list)
                                repair_list.append([value, []])
                            for result in result_list:
                                repair_list[repair_dict[value]][1].append(result.page_content)
        return repair_list
