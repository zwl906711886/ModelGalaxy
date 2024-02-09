import uvicorn

from configs.api_config import create_app
from configs.server_config import BACKEND_SERVER

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=BACKEND_SERVER["host"],
        port=BACKEND_SERVER["port"]
    )
