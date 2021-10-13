from functools import reduce


def get(dic, keys):
    """
    :param dic: Potentially multi-level dictionary
    :param keys: Potentially `.`-separated keys
    """
    return reduce(lambda acc, elm: acc[elm], keys, dic)

