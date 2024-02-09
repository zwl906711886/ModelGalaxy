import json
from typing import *

import httpx
from langchain import SQLDatabase
from langchain.graphs import Neo4jGraph

from configs.database_config import DRAFT_MAPPING, DATABASE_MAPPING
from configs.model_config import LLM_URL
from entity.request import LLMRequest
from utils.faiss_util import FaissManager


class LLMManager:
    def __init__(
            self,
            graph: Neo4jGraph,
            db: SQLDatabase,
    ):
        self.graph = graph
        self.schema = {
            "1": graph.schema,
            "2": "表：evaluation_tables\n属性：task, dataset, metrics, model_name, id\n"
        }
        self.db = db
        self.node_properties = []
        self.relationship_properties = []
        self.relationships = []
        try:
            self.refresh_schema()
        except Exception as e:
            print(e)

    @staticmethod
    def _query(request: LLMRequest):
        response = httpx.post(LLM_URL, json=request.json, timeout=100)
        return str(response.json())

    def query_database(
            self,
            check_result: str,
            statement: str
    ):
        if check_result == "1":
            return self.graph.query(statement)
        elif check_result == "2":
            return self.db.run(statement)
        else:
            return ""

    def chat(
            self,
            query: str,
            faiss_manager: FaissManager,
    ) -> str:
        result_list = []
        answer = "抱歉，我无法理解您的问题"
        check_result = "0"
        try:
            # 检查异常信息并且生成draft
            check_result = self._query(LLMRequest("CHECK", {"question": query}))[0]
            # 输入分支不为异常语句
            if check_result == "1" or check_result == "2":
                statement = self._query(
                    LLMRequest(DRAFT_MAPPING[check_result], {"question": query, "schema": self.schema[check_result]}))
                print(f"statement: {statement}")
                # 获取匹配值
                try:
                    value_dict = json.loads(self._query(LLMRequest("MATCH", {"question": statement})))
                    print(f"value_dict: {value_dict}")
                except Exception as e:
                    value_dict = {}
                # 有匹配值则查询数据库
                if value_dict:
                    repair_list = faiss_manager.search_repair_list(DATABASE_MAPPING[check_result], value_dict)
                    print(f"repair_list: {repair_list}")
                    statement_candidates = []
                    self.generate_statement_candidates(statement, repair_list, statement_candidates)
                    print(f"candidates: {statement_candidates}")
                    # 遍历所有结果，从中选取几个较好的供用户选择
                    for statement in statement_candidates:
                        result = []
                        try:
                            result = self.query_database(check_result, statement)
                        except Exception as e:
                            pass
                        if result:
                            result_list.append([statement, result])
                        if len(result_list) >= 3:
                            break
                else:
                    result_list = [[statement, self.query_database(check_result, statement)]]
        except Exception as e:
            result_list = []
        try:
            # 深层查询数据库
            result_list.append(self.deep_query(check_result, result_list))
            answer = self._query(
                LLMRequest("ANSWER", {"question": query, "result": json.dumps(result_list, ensure_ascii=False)}))
        except Exception as e:
            print(e)
            pass
        return answer

    def deep_query(
            self,
            check_result: str,
            result_list: List[List[Union[List[Dict], str]]]
    ) -> List[List[Union[List, str]]]:
        name_list = []
        deep_result = []
        for result in result_list:
            if isinstance(result[1], str):
                obj_list = eval(result[1])
            else:
                obj_list = result[1]
            if not isinstance(obj_list, list):
                continue
            for obj in obj_list:
                if check_result == "1":
                    for key in obj.keys():
                        name_list.append(obj[key].get('name'))
                elif check_result == "2":
                    if len(obj) == 4:
                        name_list.append(obj[3])
        name_list = list(set(name_list))
        # 分别查询model和dataset表
        if check_result == "1":
            deep_result.extend(self.analyse_result('model', name_list))
            deep_result.extend(self.analyse_result('dataset', name_list))
        elif check_result == "2":
            deep_result.extend(self.analyse_result('model', name_list))
        print(f'deep_result: f{deep_result}')
        return deep_result

    def analyse_result(
            self,
            table: str,
            name_list: List[str],
    ) -> List:
        result_list = []
        name_str = '(\'' + '\',\''.join(name_list) + '\')'
        if len(name_list) > 0 and (table == 'model' or table == 'dataset'):
            result = self.query_database("2",
                                         f"SELECT id, name, description FROM {table}_manage_{table} where name IN {name_str}")
            table = table.removesuffix('set')
            if result != '':
                result = eval(result)
                for obj in result:
                    result_list.append(
                        {
                            'url': f'http://192.168.3.242:8088/modelhub/{table}Detail/{obj[0]}',
                            'name': f'{obj[1]}',
                            'description': f'{obj[2]}',
                        }
                    )
        return result_list

    def generate_statement_candidates(
            self,
            cypher: str,
            repair_list: List[List[Union[str, List[str]]]],
            result_list: List[str],
            index_list: List[int] = None,
            step: int = 0
    ) -> None:
        if step == len(repair_list):
            temp = cypher
            for i in range(len(repair_list)):
                temp = temp.replace(
                    f"\"{repair_list[i][0]}\"",
                    f"\"{repair_list[i][1][index_list[i]]}\""
                ).replace(
                    f"\'{repair_list[i][0]}\'",
                    f"\'{repair_list[i][1][index_list[i]]}\'"
                )
            result_list.append(temp)
            return
        if index_list is None:
            index_list = [0 for _ in range(len(repair_list))]
        for i in range(len(repair_list[step][1])):
            index_list[step] = i
            self.generate_statement_candidates(cypher, repair_list, result_list, index_list, step + 1)

    def refresh_schema(self):
        from langchain.graphs.neo4j_graph import (
            node_properties_query,
            rel_properties_query,
            rel_query
        )
        node_properties = [el['output'] for el in self.graph.query(node_properties_query)]
        relationship_properties = [el['output'] for el in self.graph.query(rel_properties_query)]
        relationships = [el['output'] for el in self.graph.query(rel_query)]

        self.node_properties = []
        self.relationship_properties = []
        self.relationships = []
        for n in node_properties:
            if n['labels'].find('page') == -1:
                self.node_properties.append({
                    'properties': [el['property'] for el in n['properties']],
                    'labels': n['labels']
                })
        for r in relationship_properties:
            if r['type'] != 'links':
                self.relationship_properties.append({
                    'properties': [el['property'] for el in r['properties']],
                    'type': r['type']
                })
        self.relationships = [el for el in relationships if el.find('page') == -1]
        self.schema["1"] = f"""
        节点属性如下：
        {self.node_properties}
        关系属性如下：
        {self.relationship_properties}
        节点之间关系如下：
        {self.relationships}
        """
        print(self.schema)
