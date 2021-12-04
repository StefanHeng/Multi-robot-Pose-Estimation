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

    # import numpy as np
    # # arr = np.array([2, 3, 2, 3, 3, 2])
    # #
    # # def idx(a, v):
    # #     return np.where(a == v)[0][0]
    # # ic(idx(arr, 2))
    # # ic(idx(arr, 3))
    # diff = np.array([0.1, 0.2, 0.3])
    # ic(np.dot(diff, diff.T))
    # ic(np.sum(np.square(diff)))

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot([1, 2], [3, 4])
    # plt.show()

    # # Taken from https://matplotlib.org/stable/gallery/widgets/buttons.html
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.widgets import Button
    #
    # freqs = np.arange(2, 20, 3)
    #
    # fig, ax = plt.subplots()
    # plt.subplots_adjust(bottom=0.2)
    # t = np.arange(0.0, 1.0, 0.001)
    # s = np.sin(2 * np.pi * freqs[0] * t)
    # l, = plt.plot(t, s, lw=2)
    #
    # class Index:
    #     def __init__(self):
    #         self.ind = 0
    #
    #     def next(self, event):
    #         self.ind += 1
    #         i = self.ind % len(freqs)
    #         ydata = np.sin(2 * np.pi * freqs[i] * t)
    #         l.set_ydata(ydata)
    #         plt.draw()
    #
    #     def prev(self, event):
    #         self.ind -= 1
    #         i = self.ind % len(freqs)
    #         ydata = np.sin(2 * np.pi * freqs[i] * t)
    #         l.set_ydata(ydata)
    #         plt.draw()
    #
    #
    # callback = Index()
    # axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    # axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    # bnext = Button(axnext, 'Next')
    # bnext.on_clicked(callback.next)
    # bprev = Button(axprev, 'Previous')
    # bprev.on_clicked(callback.prev)
    #
    # plt.show()

    # import numpy as np
    #
    # def rot_mat(theta):
    #     c, s = np.cos(theta), np.sin(theta)
    #     return np.array([
    #         [c, -s],
    #         [s, c]
    #     ])
    # ic(rot_mat(0))

    # import pint
    # reg = pint.UnitRegistry()
    # sz_m = 1.74 * reg.meter
    # ic(sz_m, type(sz_m))  # 1.74 meter
    # sz_in = sz_m.to(reg.inch)
    # ic(sz_in, type(sz_in))
    # ic(vars(sz_in), sz_in.magnitude)

    # Modified from https://stackoverflow.com/questions/8247973/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    #
    # # my random data
    # scale = 10
    # np.random.seed(101)
    # x = np.random.random(10) * scale
    # y = np.random.random(10) * scale
    # ic(x, y)
    #
    # # spacing of arrows
    # arr_space = .1  # good value for scale of 1
    # arr_space *= scale
    #
    # # r is the distance spanned between pairs of points
    # r = [0]
    # for i in range(1, len(x)):
    #     dx = x[i] - x[i - 1]
    #     dy = y[i] - y[i - 1]
    #     r.append(np.sqrt(dx * dx + dy * dy))
    # r = np.array(r)
    # ic(r)
    #
    # # rtot is a cumulative sum of r, it's used to save time
    # rtot = []
    # for i in range(len(r)):
    #     rtot.append(r[0:i].sum())
    # rtot.append(r.sum())
    # ic(rtot)
    #
    # arrowData = []  # will hold tuples of x,y,theta for each arrow
    # arrowPos = 0  # current point on walk along data
    # rcount = 1
    # while arrowPos < r.sum():
    #     x1, x2 = x[rcount - 1], x[rcount]
    #     y1, y2 = y[rcount - 1], y[rcount]
    #     da = arrowPos - rtot[rcount]
    #     theta = np.arctan2((x2 - x1), (y2 - y1))
    #     ax = np.sin(theta) * da + x1
    #     ay = np.cos(theta) * da + y1
    #     arrowData.append((ax, ay, theta))
    #     arrowPos += arr_space
    #     while arrowPos > rtot[rcount + 1]:
    #         rcount += 1
    #         if arrowPos > rtot[-1]:
    #             break
    #
    # # could be done in above block if you want
    # for ax, ay, theta in arrowData:
    #     # use aspace as a guide for size and length of things
    #     # scaling factors were chosen by experimenting a bit
    #     axes.arrow(ax, ay,
    #                np.sin(theta) * arr_space / 10, np.cos(theta) * arr_space / 10,
    #                head_width=arr_space / 8)
    #
    # axes.plot(x, y)
    # axes.set_xlim(x.min() * .9, x.max() * 1.1)
    # axes.set_ylim(y.min() * .9, y.max() * 1.1)
    #
    # plt.show()

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    #
    # def distance(data):
    #     return np.sum((data[1:] - data[:-1]) ** 2, axis=1) ** .5
    #
    #
    # def draw_path(path):
    #     HEAD_WIDTH = 2
    #     HEAD_LEN = 3
    #
    #     fig = plt.figure()
    #     axes = fig.add_subplot(111)
    #
    #     x = path[:, 0]
    #     y = path[:, 1]
    #     axes.plot(x, y)
    #
    #     theta = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])
    #     dist = distance(path) - HEAD_LEN
    #
    #     x = x[:-1]
    #     y = y[:-1]
    #     ax = x + dist * np.sin(theta)
    #     ay = y + dist * np.cos(theta)
    #
    #     for x1, y1, x2, y2 in zip(x, y, ax - x, ay - y):
    #         axes.arrow(x1, y1, x2, y2, head_width=HEAD_WIDTH, head_length=HEAD_LEN)
    #     plt.show()
    #
    # arr = np.vstack([x, y])
    # ic(arr.shape)
    # draw_path(arr)

    # from matplotlib import pyplot as plt
    # import numpy as np
    # #
    # # # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # # # plt.rcParams["figure.autolayout"] = True
    # # # x = np.linspace(-2, 2, 100)
    # # # y = np.sin(x)
    # # # plt.plot(x, y, c='b', lw=1)
    # # # plt.arrow(0, 0, 0.01, np.sin(0.01), shape='full', lw=10,
    # # #           length_includes_head=True, head_width=.05, color='r')
    # # # plt.show()
    # #
    # def plot_line_seg_arrow(c1, c2, scale=0.01):
    #     coords = np.array([c1, c2])
    #     ic(coords)
    #     mean = coords.mean(axis=0)
    #     mags = (coords[1] - coords[0]) * scale
    #     ic(mean, mags)
    #     plt.arrow(*(mean-mags/2), *mags, head_width=0.05, length_includes_head=True, lw=0, overhang=0.2)
    #
    # plt.figure(figsize=(16, 9), constrained_layout=True)
    # x = [1, 2, 6]
    # y = [3, 4, 2]
    # plt.plot(x, y, ms=1, lw=0.5)
    # plot_line_seg_arrow(
    #     (x[0], y[0]),
    #     (x[1], y[1])
    # )
    # plt.gca().set_aspect('equal')
    # plt.show()

    # import numpy as np
    # keys = np.array([
    #     'b', 'a', 'c'
    # ])
    # vals = np.array([
    #     [1, 23],
    #     [32323, 1],
    #     [2, 5]
    # ])
    # idxs = np.argsort(keys)
    # ic(idxs, vals[idxs])

    # import numpy as np
    # # grid = np.mgrid[[slice(0, 1, 0.1)] * 6]
    # # ic(grid.shape)
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(0, 20, 0.25)
    # X, Y = np.meshgrid(X, Y)
    # ic(X.shape, Y.shape)
    # ic(X, Y)

    # https://stackoverflow.com/questions/55531760/is-there-a-way-to-label-multiple-3d-surfaces-in-matplotlib/55534939
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    plt.rcParams['legend.fontsize'] = 10

    # First constraint
    g2 = np.linspace(-5, 5, 2)
    g3 = np.linspace(-5, 5, 2)
    G2, G3 = np.meshgrid(g2, g3)
    G4_1 = -1.18301270189222 - 0.5 * G2 + 0.5 * G3
    ax = fig.gca(projection='3d')
    c1 = ax.plot_surface(G2, G3, G4_1, label="c1")
    c1._facecolors2d = c1._facecolor3d
    c1._edgecolors2d = c1._edgecolor3d

    # # Second
    # G3, G4 = np.meshgrid(g2, g3)
    # G2 = G3
    # c2 = ax.plot_surface(G2, G3, G4, label="c2")
    # c2._facecolors2d = c2._facecolors3d
    # c2._edgecolors2d = c2._edgecolors3d
    #
    # # Third
    # G2, G3 = np.meshgrid(g2, g3)
    # G4 = (0.408248290463863 * G2 + 0.408248290463863 * G3 - 0.707106781186548) / 1.63299316185545
    # c3 = ax.plot_surface(G2, G3, G4, label="c3")
    # c3._facecolors2d = c3._facecolors3d
    # c3._edgecolors2d = c3._edgecolors3d
    #
    # # And forth
    # G4 = (1.04903810567666 - (0.288675134594813 * G2 + 0.288675134594813 * G3)) / 0.577350269189626
    # c4 = ax.plot_surface(G2, G3, G4, label="c4")
    #
    # c4._facecolors2d = c4._facecolors3d
    # c4._edgecolors2d = c4._edgecolors3d

    ax.legend()  # -> error : 'AttributeError: 'Poly3DCollection' object has no attribute '_edgecolors2d''

    # labeling the figure
    fig.suptitle("Constraints")
    # plt.xlabel('g2', fontsize=14)
    # plt.ylabel('g3', fontsize=14)
    ax.set_xlabel(r'$g_2$', fontsize=15, rotation=60)
    ax.set_ylabel('$g_3$', fontsize=15, rotation=60)
    ax.set_zlabel('$g_4$', fontsize=15, rotation=60)
    plt.show()

