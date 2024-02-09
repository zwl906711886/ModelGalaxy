import os
from typing import List, Union

from configs import base_dir

# 向量库存储路径
if not os.path.exists(data_dir := os.path.join(base_dir, "data")):
    os.mkdir(data_dir)
