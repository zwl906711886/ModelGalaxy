from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from configs.llm_config import llm_model_dict, LLM_MODEL
from configs.prompt_config import prompt_code_dict
from entity.request import LLMRequest


def get_llm():
    openai_api_key = llm_model_dict[LLM_MODEL]["api_key_pool"].get()
    print(f"使用的秘钥是：{openai_api_key}")
    llm = ChatOpenAI(
        streaming=False,
        verbose=False,
        openai_api_key=openai_api_key,
        openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
        model_name=LLM_MODEL,
        temperature=0
    )
    return llm


def chat(msg: LLMRequest):
    retry = 3
    result = "无法查询到结果"
    print("\n\n")
    while retry > 0:
        try:
            print(f"传递参数为：{msg}")
            if not prompt_code_dict.keys().__contains__(msg.code):
                return ""
            if msg.inputs is None:
                msg.inputs = {}
            chain = LLMChain(
                llm=get_llm(),
                prompt=prompt_code_dict[msg.code].prompt
            )
            result = chain.run(msg.inputs)
            break
        except Exception as e:
            print(e)
            retry -= 1
    # cypher方向修复
    if msg.code == "CYPHER_DRAFT":
        result = result.replace("<-", "-").replace("->", "-")
    print(f"返回结果为：{result}")
    print("\n\n")
    return result
