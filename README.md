# ModelGalaxy
The code of paper: ModelGalaxy: A Versatile Model Retrieval Platform

## LLM Empowered Conversational Retrieval

## requirements

```
faiss-gpu==1.7.2 			# if use cpu: faiss-cpu==1.7.2
fastapi==0.105.0
langchain==0.0.287
httpx==0.26.0
neo4j==5.15.0
numpy==1.26.1
pymysql==1.1.0
torch==1.11.0+cu115 		# if use gpu
torchaudio==0.11.0+cu115	# if use gpu
torchvision==0.12.0+cu115	# if use gpu
transformers==4.33.2
uvicorn==0.25.0
```

### database

Neo4j and MySQL contain a lot of model library data crawled from the Internet, which is not convenient to disclose.

### chatchat-chatbox

A frontend using Streamlit to display the model library ai question and answer retrieval related content.

### chatchat_manage

A backend using FastApi is mainly responsible for running the process according to the LLM call chain.

### chatchat-llm

A backend using FastApi, mainly using the account pool proxy to call gpt.
