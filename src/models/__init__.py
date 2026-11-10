"""模型包：导入即注册全部结构，供 build_model 按名字取用。"""

from src.models import baselines, uc_escm  # noqa: F401  (副作用注册)
