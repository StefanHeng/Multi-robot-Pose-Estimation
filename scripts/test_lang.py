from icecream import ic


if __name__ == '__main__':
    # pass

    # from functools import reduce
    # d = dict(
    #     a=dict(
    #         b=1,
    #         c=2
    #     ),
    #     d=1,
    #     e=dict(
    #         f=dict(
    #             g=3,
    #             h=4
    #         ),
    #         i=5
    #     )
    # )
    # ks = ['a', 'b']
    #
    # def get(dic, keys):
    #     return reduce(lambda acc, elm: acc[elm], keys, dic)
    #
    # ic(get(d, ks))
    #
    # # def keys(dic):
    # #     def _keys(d_, k, prefix=''):
    # #         v = d_[k]
    # #         if type(v) is dict:
    # #             return [_keys(v_, k_, f'{prefix}.{k}') for k_, v_ in v.items()]
    # #         else:
    # #             return [f'{prefix}.{k}' for k in ]
    #
    #
    # def keys(dic, prefix=''):
    #     def _full(k_):
    #         return k_ if prefix == '' else f'{prefix}.{k_}'
    #     for k, v in dic.items():
    #         if isinstance(v, dict):
    #             for k__ in keys(v, prefix=_full(k)):
    #                 yield k__
    #         else:
    #             yield _full(k)
    #
    # # ic(list(keys(d)))
    # for key in keys(d):
    #     ic(key)

    import subprocess

    string = "echo Hello world"
    result = subprocess.getoutput(string)
    print("result::: ", result)

