import logging

# 日志格式
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)
# 是否显示详细日志
log_verbose = False

# 是否开启跨域
OPEN_CROSS_DOMAIN = False

# 服务器默认绑定host
DEFAULT_BIND_HOST = "127.0.0.1"

# 后端配置
BACKEND_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 7861,
}
