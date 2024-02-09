import logging

# 日志格式
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)
# 是否显示详细日志
log_verbose = False

# 请求默认超时时间（秒）。如果加载模型或对话较慢，出现超时错误，可以适当加大该值。
HTTPX_DEFAULT_TIMEOUT = 300.0

# 服务器默认绑定host
DEFAULT_BIND_HOST = "127.0.0.1"

# 前端配置
FRONTEND_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 8502,
}
