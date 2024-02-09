# Neo4j数据库配置
DEFAULT_NEO4J_CONFIG = {
    "url": "bolt://192.168.3.242:7687",
    "username": "neo4j",
    "password": "123456",
    "database": "neo4j"
}

# MySQL数据库配置
DEFAULT_MYSQL_CONFIG = {
    "host": "192.168.3.242",
    "port": "3306",
    "name": "mreops",
    "user": "root",
    "password": "1234"
}

# 查询语句编号映射
DRAFT_MAPPING = {
    "1": "CYPHER_DRAFT",
    "2": "SQL_DRAFT"
}

# 数据库映射
DATABASE_MAPPING = {
    "1": "neo4j",
    "2": "mysql"
}
