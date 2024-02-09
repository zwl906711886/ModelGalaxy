from langchain.prompts import ChatPromptTemplate


class Prompt:
    def __init__(self, messages):
        self.messages = messages
        self.prompt = ChatPromptTemplate.from_messages(messages)
