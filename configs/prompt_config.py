from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from utils.prompt_util import Prompt

check_prompt = Prompt([
    SystemMessagePromptTemplate.from_template(
        "你是一个帮我判断问题是否是查询语句的助手。"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        模型库的主要组成部分为数据集、任务、论文、工具、方法等。
        我会给出一个用户输入的语句，你需要判断这个语句是正常的查询模型库语句还是其他的异常的交互式语句。
        例如“你好”，“再见”，“fiafabvbeuaffw”这种主要目的不是查询模型库的异常语句，你只需要返回“0”。
        如果是这种“我想要获取数据集cifar100的详细信息。”用户没有期望比较模型的运行表现，只是想要获取一些信息的查询语句，你只需要返回“1”。
        如果是这种“我想要获取在cifar100数据集上运行表现比CNN好的模型。”，用户很明显想要比较模型的运行表现，你只需要返回“2”。
        你只需要返回结果数字，不需要返回其他解释性语句，理性地做出判断。
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        你好
        """
    ),
    AIMessagePromptTemplate.from_template(
        "0"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        再见
        """
    ),
    AIMessagePromptTemplate.from_template(
        "0"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        你是谁
        """
    ),
    AIMessagePromptTemplate.from_template(
        "0"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        你在做什么
        """
    ),
    AIMessagePromptTemplate.from_template(
        "0"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        hdawogheqfoaf
        """
    ),
    AIMessagePromptTemplate.from_template(
        "0"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        19
        """
    ),
    AIMessagePromptTemplate.from_template(
        "0"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        帮我看看这个文档
        """
    ),
    AIMessagePromptTemplate.from_template(
        "0"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        我想要获取数据集cifar100的详细信息。
        """
    ),
    AIMessagePromptTemplate.from_template(
        "1"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        请给我表现比typiclust模型好的模型。
        """
    ),
    AIMessagePromptTemplate.from_template(
        "1"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        有没有和bert方法相似的方法的论文。
        """
    ),
    AIMessagePromptTemplate.from_template(
        "1"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        有没有比bert好的模型。
        """
    ),
    AIMessagePromptTemplate.from_template(
        "2"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        我想要获取在cifar100数据集上运行表现比CNN好的模型。
        """
    ),
    AIMessagePromptTemplate.from_template(
        "2"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        用户输入的语句是：
        {question}
        """
    )
])
"""
检查提示，输入inputs参数：question
"""

cypher_draft_generator_prompt = Prompt([
    SystemMessagePromptTemplate.from_template(
        "你是一个帮我将自然语言转成查询图数据库的cypher语句的助手。"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        只用在我新给出的schema中出现的节点标签属性和关系属性。
        不要用没给出的类型和属性构造cypher语句。
        在cypher语句要匹配的标签两边都添加上'`'符号，目的是避免报错。
        需要注意生成cypher语句的返回结构一定和schema一致，如果不清楚要返回的属性，可以直接将整个对象返回。
        如果用户要使用递归多跳查询，需要在结果前加上DISTINCT标识符避免结果重复，递归的上限设置为5跳，即如果过用户没有限制跳数，则一定要在*..之后加上5，
        即类似[:`perform better`*..5]限制上限。
        需要在查询语句结尾加上LIMIT 10的限制，从而设置查询条数上限。
        一定要注意生成的cypher语句中如果关系是带有方向的，那么一定要注意这个方向要符合schema中的关系方向，如果不能理解关系的方向性，那么生成的语句关系就不要带方向。
        如果用户直接输入了cypher语句，则直接返回用户输入的cypher语句。
        如果生成的匹配字符串为中文的话，生成的cypher语句中一定要翻译成英文，最终生成的查询语句中一定不要含有中文。
        记住，一定要将匹配中的中文字符串翻译成英文。
        不需要携带其他解释性语句。
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['name', 'paper_title'], 'labels': 'model'}}, {{'properties': ['full_name', 'paper', 'introduced_date', 'name'], 'labels': 'dataset'}}]
        关系属性如下：
        []
        节点之间关系如下：
        ['(:model)-[:perform better]->(:model)', '(:model)-[:trained in]->(:paper)', '(:dataset)-[:datasets from]->(:paper)']
        我的问题是：
        我想要获取数据集cifar100的详细信息。
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(n:`dataset`{{name:"cifar100"}}) RETURN n LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['name', 'paper_title'], 'labels': 'model'}}, {{'properties': ['full_name', 'paper', 'introduced_date', 'name'], 'labels': 'dataset'}}]
        关系属性如下：
        []
        节点之间关系如下：
        ['(:model)-[:perform better]->(:model)', '(:model)-[:trained in]->(:paper)', '(:dataset)-[:datasets from]->(:paper)']
        我的问题是：
        Please introduce the dataset cifar100.
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(n:`dataset`{{name:"cifar100"}}) RETURN n LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['name', 'paper_title'], 'labels': 'model'}}, {{'properties': ['full_name', 'paper', 'introduced_date', 'name'], 'labels': 'dataset'}}]
        关系属性如下：
        []
        节点之间关系如下：
        ['(:model)-[:perform better]->(:model)', '(:model)-[:trained in]->(:paper)', '(:dataset)-[:datasets from]->(:paper)']
        我的问题是：
        Please introduce the cifar100 dataset.
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(n:`dataset`{{name:"cifar100"}}) RETURN n LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['name', 'paper_title'], 'labels': 'model'}}, {{'properties': ['full_name', 'paper', 'introduced_date', 'name'], 'labels': 'dataset'}}]
        关系属性如下：
        []
        节点之间关系如下：
        ['(:model)-[:perform better]->(:model)', '(:model)-[:trained in]->(:paper)', '(:dataset)-[:datasets from]->(:paper)']
        我的问题是：
        我想要获取cifar10数据集的相关信息。
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(n:`dataset`{{name:"cifar10"}}) RETURN n LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['name', 'paper_title'], 'labels': 'model'}}, {{'properties': ['full_name', 'paper', 'introduced_date', 'name'], 'labels': 'dataset'}}]
        关系属性如下：
        []
        节点之间关系如下：
        ['(:model)-[:perform better]->(:model)', '(:model)-[:trained in]->(:paper)', '(:dataset)-[:datasets from]->(:paper)']
        我的问题是：
        我想要获取typiclust模型的相关信息。
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(n:`model`{{name:"typiclust"}}) RETURN n LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['name', 'paper_title'], 'labels': 'model'}}]
        关系属性如下：
        []
        节点之间关系如下：
        ['(:model)-[:perform better]->(:model)', '(:model)-[:trained in]->(:paper)']
        我的问题是：
        我想要获取表现比typiclust模型好的模型。
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(:`model`{{name:"typiclust"}})<-[:`perform better`*..10]-(n:`model`) RETURN DISTINCT n LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['name', 'paper_title'], 'labels': 'model'}}, {{'properties': ['name'], 'labels': 'task'}}, {{'properties': ['name', 'full_name', 'introduced_date'], 'labels': 'dataset'}}]
        关系属性如下：
        []
        节点之间关系如下：
        []
        我的问题是：
        我想要获取进行过图像分类任务的模型的信息。
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(t:`task`{{name:"Image Classification"}})-[:`task on`]-(d:`dataset`)-[:`datasets from`]-(p:`paper`)-[:`trained in`]-(m:`model`) RETURN DISTINCT m LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['name', 'paper_title'], 'labels': 'model'}}, {{'properties': ['name'], 'labels': 'task'}}, {{'properties': ['name', 'full_name', 'introduced_date'], 'labels': 'dataset'}}]
        关系属性如下：
        []
        节点之间关系如下：
        []
        我的问题是：
        我想要获取文本生成任务的信息。
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(:`task`{{name:"Text Generation"}}) RETURN n"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['name'], 'labels': 'method'}}]
        关系属性如下：
        []
        节点之间关系如下：
        ['(:method)-[:introduced in ]->(:paper)', '(:method)-[:similar to]->(:method)']
        我的问题是：
        我想获取和bert方法相似的方法。
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(:`method`{{name:"BERT"}})-[:`similar to`*..5]-(n:`method`) RETURN DISTINCT n LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['name'], 'labels': 'method'}}]
        关系属性如下：
        []
        节点之间关系如下：
        ['(:method)-[:introduced in ]->(:paper)', '(:method)-[:similar to]->(:method)']
        我的问题是：
        我想获取和介绍了和bert方法相似的方法的论文。
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(:`method`{{name:"BERT"}})-[:`similar to`*..5]-(:`method`)-[:`introduced in `]->(n:`paper`) RETURN DISTINCT n LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}, {{'properties': ['full_name', 'paper', 'introduced_date', 'name'], 'labels': 'dataset'}}]
        关系属性如下：
        []
        节点之间关系如下：
        ['(:dataset)-[:datasets from]->(:paper)']
        我的问题是：
        我想要知道数据集RLV来自于哪篇论文。
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(:`dataset`{{name:"RLV"}})-[:`datasets from`]->(n:`paper`) RETURN n LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        节点属性如下：
        [{{'properties': ['abstract', 'title', 'date', 'url_abs'], 'labels': 'paper'}}]
        关系属性如下：
        []
        节点之间关系如下：
        []
        我的问题是：
        MATCH(:`dataset`{{name:"RLV"}})-[:`datasets from`]->(n:`paper`) RETURN n
        """
    ),
    AIMessagePromptTemplate.from_template(
        """MATCH(:`dataset`{{name:"RLV"}})-[:`datasets from`]->(n:`paper`) RETURN n LIMIT 10"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的图数据库的schema是：
        {schema}
        我的问题是：
        {question}
        """
    )
])
"""
生成草稿提示，输入inputs参数：schema, question
"""

sql_draft_generator_prompt = Prompt([
    SystemMessagePromptTemplate.from_template(
        "你是一个按照我提供的逻辑来生成sql语句的助手。"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        记住，你只需要根据我的逻辑来生成语句，不需要分析语句本身的逻辑。
        我会给你一个mysql数据库的表结构和一个目标语句。
        首先，你需要从目标语句中获取任务名称和数据集名称。
        然后，你只需要以下面这个sql语句为基础来生成语句：“SELECT task, dataset, metrics, model_name FROM evaluation_tables WHERE task = ? AND dataset = ? LIMIT 10”
        其中第一个问号匹配用户查询的任务名称，第二个问号匹配用户查询的数据集名称。
        需要限制返回的元素数量为10，即在生成语句末尾添加“LIMIT 10”。
        不需要使用“ORDER BY”进行排序。
        记住不要添加任何其他的解释性语句，只需要生成按照我的逻辑得出的sql语句即可。
        如果生成的匹配字符串为中文的话，生成的sql语句中一定要翻译成英文，最终生成的查询语句中一定不要含有中文。
        记住，一定要将匹配中的中文字符串翻译成英文。
        记住不需要考虑目标语句本身的逻辑含义。
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的mysql数据库的表结构是：
        表：evaluation_tables
        属性：task, dataset, metrics, model_name
        目标语句是：
        我想要知道在cifar100数据集上哪个模型表现最好？
        """
    ),
    AIMessagePromptTemplate.from_template(
        "SELECT task, dataset, metrics, model_name FROM evaluation_tables LIMIT 10"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的mysql数据库的表结构是：
        表：evaluation_tables
        属性：task, dataset, metrics, model_name
        目标语句是：
        请给我推荐一些用于图像分类的模型
        """
    ),
    AIMessagePromptTemplate.from_template(
        "SELECT task, dataset, metrics, model_name FROM evaluation_tables WHERE dataset = 'cifar100' LIMIT 10"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的mysql数据库的表结构是：
        表：evaluation_tables
        属性：task, dataset, metrics, model_name
        目标语句是：
        我想要知道针对于Active Learning任务，哪个模型在cifar100数据集上表现最好？
        """
    ),
    AIMessagePromptTemplate.from_template(
        "SELECT task, dataset, metrics, model_name FROM evaluation_tables WHERE task = 'Active Learning' AND dataset = 'cifar100' LIMIT 10"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的mysql数据库的表结构是：
        表：evaluation_tables
        属性：task, dataset, metrics, model_name
        目标语句是：
        给我推荐一些表现比较好的模型
        """
    ),
    AIMessagePromptTemplate.from_template(
        "SELECT task, dataset, metrics, model_name FROM evaluation_tables LIMIT 10"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的mysql数据库的表结构是：
        表：evaluation_tables
        属性：task, dataset, metrics, model_name
        目标语句是：
        我想要知道针对于图像生成任务，哪个模型在cifar100数据集上表现最好？
        """
    ),
    AIMessagePromptTemplate.from_template(
        "SELECT task, dataset, metrics, model_name FROM evaluation_tables WHERE task = 'Image Generation' AND dataset = 'cifar100' LIMIT 10"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的mysql数据库的表结构是：
        表：evaluation_tables
        属性：task, dataset, metrics, model_name
        目标语句是：
        我想知道哪个模型表现比较好
        """
    ),
    AIMessagePromptTemplate.from_template(
        "SELECT task, dataset, metrics, model_name FROM evaluation_tables WHERE task = 'Image Classification' LIMIT 10"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的mysql数据库的表结构是：
        表：evaluation_tables
        属性：task, dataset, metrics, model_name
        目标语句是：
        请给我推荐一些用于文本生成的模型
        """
    ),
    AIMessagePromptTemplate.from_template(
        "SELECT task, dataset, metrics, model_name FROM evaluation_tables WHERE task = 'Text Generation' LIMIT 10"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的mysql数据库的表结构是：
        {schema}
        目标语句是：
        {question}
        """
    )
])

match_binder_prompt = Prompt([
    SystemMessagePromptTemplate.from_template(
        "你是一个帮我将cypher或者sql语句中的特殊值提取成json字典的助手。"
    ),
    HumanMessagePromptTemplate.from_template(
        """
        如果给出的cypher语句或者sql需要在match或者where部分把某个标签的某个属性限制为某个字符串，那么就把这个标签、这个属性和这个字符串拿出来。
        把这个语句中所有的(标签，属性，字符串)三元组都拿出来之后，构建成一个符合{{标签:{{属性:[字符串]}}}}的结构的json字典返回给我。
        请只返回json字典，不要使用代码块封装，不要添加任何其他的抱歉或者备注语句，请忽略我提供的cypher语句中的'`'字符。
        只提取字符串值，如果是浮点型或者整数型则不需要提取。
        记住，如果是sql语句，最外层为mreops，此时第一层标签为数据库名称，目前的数据库名为mreops。
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的问题是：
        MATCH(:`dataset`{{name:"RLV"}})-[:`datasets from`]->(n:`paper`) RETURN n
        """
    ),
    AIMessagePromptTemplate.from_template(
        """{{"dataset":{{"name":["RLV"]}}}}"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的问题是：
        SELECT task, dataset, metrics, model_name FROM evaluation_tables WHERE dataset = 'cifar100'
        """
    ),
    AIMessagePromptTemplate.from_template(
        """{{"mreops":{{"dataset":["cifar100"]}}}}"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的问题是：
        MATCH(:`method`{{name:"BERT"}})-[:`similar to`*..5]-(n:`method`{{name:"VGG",age:15,date:"2002-06-26"}}) RETURN n
        """
    ),
    AIMessagePromptTemplate.from_template(
        """{{"method":{{"name":["BERT","VGG"],"date":["2002-06-26"]}}}}"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的问题是：
        SELECT task, dataset, metrics, model_name FROM evaluation_tables WHERE task = 'Active Learning' AND dataset = 'cifar100'
        """
    ),
    AIMessagePromptTemplate.from_template(
        """{{"mreops":{{"task":["Active Learning"],"dataset":["cifar100"]}}}}"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的问题是：
        SELECT task, dataset, metrics, model_name FROM evaluation_tables LIMIT 10
        """
    ),
    AIMessagePromptTemplate.from_template(
        """{{}}"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的问题是：
        MATCH(:`method`)-[:`similar to`*..5]-(n:`method`) WHERE n.name = "VGG" AND n.code = 20 AND n.date = "2002-06-26" RETURN n
        """
    ),
    AIMessagePromptTemplate.from_template(
        """{{"method":{{"name":["VGG"],"date":["2002-06-26"]}}}}"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的问题是：
        MATCH(:`method`)-[]-(n:`method`) RETURN n
        """
    ),
    AIMessagePromptTemplate.from_template(
        """{{}}"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
        我的问题是：
        {question}
        """
    )
])
"""
获取匹配特殊值提示，输入inputs参数：question
"""

answer_generator_prompt = Prompt([
    SystemMessagePromptTemplate.from_template(
        "You are an assistant to help me find answers from the knowledge base."
    ),
    HumanMessagePromptTemplate.from_template(
        """
        <command>Answer questions in natural language concisely and professionally based on known information. If you find a link related to the query question in the known information, you must attach the link information to the answer.</command>
        <information>{{ {result} }}</information>
        <question>{{ {question} }}</question>
        <limit>The language of the answer must be based on the question. If the question is English, it must be answered in English. If the question is Chinese, it must be answered in Chinese.</limit>
        """
        # <指令>根据已知信息，简洁和专业的来使用自然语言回答问题。如果在已知信息中找到了和查询问题相关的链接，一定要在回答中附加链接信息。</指令>
        # <已知信息>{{ {result} }}</已知信息>
        # <问题>{{ {question} }}</问题>
        # <限制>回答的语言一定要以问题为准，如果问题是英文的，一定要用英文回答，如果问题是中文的，一定要用中文回答。</限制>
        #
    ),
])
"""
生成回答结果特殊值提示，输入inputs参数：question, result
"""

# 获取节点标签列表，用作构造向量数据库存储
NODE_LABELS_TEMPLATE = """# Task:
生成一个json字典字符串，这个字典包含了我的图数据库中的所有的节点标签和节点的属性。
这个json字典组成结构和Example中的一样。
# Instructions:
只用在下面schema中出现的节点标签属性和关系属性。
不要用没给出的类型和属性进行构造。
# Schema:
{schema}
# Example:
样例问题: [{{'properties': [{{'property': 'name', 'type': 'STRING'}}], 'labels': 'test1'}},
{{'properties': [{{'property': 'code', 'type': 'STRING'}}, 
{{'property': 'msg', 'type': 'STRING'}}], 'labels': 'test2'}}]
样例回答: {{{{"test1": ["name"], "test2": ["code", "msg"]}}}}
# Note: 
不要添加任何解释性语句在回答中。
不要使用任何代码块对结果进行封装。
结果只需要给出一个结果字符串即可。
"""

# 获取所有属性
MATCH_DISTINCT_VALUE_CYPHER = "MATCH(n:`{label}`) RETURN n.{prop}"

prompt_code_dict = {
    "CHECK": check_prompt,
    "CYPHER_DRAFT": cypher_draft_generator_prompt,
    "SQL_DRAFT": sql_draft_generator_prompt,
    "MATCH": match_binder_prompt,
    "ANSWER": answer_generator_prompt,
}
