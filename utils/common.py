

import os

"""  ----------------------------------------------- utils functions ----------------------------------------------  """

def resolve_path(path_str: str):
    """兼容相对/绝对路径，返回绝对路径。"""
    if path_str is None: return None
    return path_str if os.path.isabs(path_str) else os.path.abspath(os.path.join(os.getcwd(), path_str))


def clean_quotes(value:str):
    """去除字段两边的引号"""
    if not value: return ''
    while len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value