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

    # import subprocess
    #
    # string = "echo Hello world"
    # result = subprocess.getoutput(string)
    # print("result::: ", result)

    # # Sort one array based on another array
    # import numpy as np
    # arr1 = np.array([4, 2, 6, 1, 12, 9])
    # arr2 = np.array([2, 3, 1, 5, 8, 23])
    # idxs = arr1.argsort()  # Sort in ascending order
    # ic(idxs)
    # ic(arr1[idxs])
    # ic(arr2[idxs])

    import numpy as np
    # arr = np.array([2, 3, 2, 3, 3, 2])
    #
    # def idx(a, v):
    #     return np.where(a == v)[0][0]
    # ic(idx(arr, 2))
    # ic(idx(arr, 3))
    diff = np.array([0.1, 0.2, 0.3])
    ic(np.dot(diff, diff.T))
    ic(np.sum(np.square(diff)))

