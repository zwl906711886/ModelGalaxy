import subprocess

from configs.server_config import FRONTEND_SERVER

if __name__ == "__main__":
    host = FRONTEND_SERVER["host"]
    port = FRONTEND_SERVER["port"]
    p = subprocess.Popen([
        "streamlit", "run", "./server_util.py",
        "--server.address", host,
        "--server.port", str(port),
        "--theme.base", "light",
        "--theme.primaryColor", "#165dff",
        "--theme.secondaryBackgroundColor", "#f5f5f5",
        "--theme.textColor", "#000000",
    ])
    p.wait()
