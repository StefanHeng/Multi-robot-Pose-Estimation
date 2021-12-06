from icecream import ic

from util import *


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

    # # https://stackoverflow.com/questions/55531760/is-there-a-way-to-label-multiple-3d-surfaces-in-matplotlib/55534939
    # import numpy as np
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    #
    # plt.rcParams['legend.fontsize'] = 10
    #
    # # First constraint
    # g2 = np.linspace(-5, 5, 2)
    # g3 = np.linspace(-5, 5, 2)
    # G2, G3 = np.meshgrid(g2, g3)
    # G4_1 = -1.18301270189222 - 0.5 * G2 + 0.5 * G3
    # ax = fig.gca(projection='3d')
    # c1 = ax.plot_surface(G2, G3, G4_1, label="c1")
    # c1._facecolors2d = c1._facecolor3d
    # c1._edgecolors2d = c1._edgecolor3d
    #
    # # # Second
    # # G3, G4 = np.meshgrid(g2, g3)
    # # G2 = G3
    # # c2 = ax.plot_surface(G2, G3, G4, label="c2")
    # # c2._facecolors2d = c2._facecolors3d
    # # c2._edgecolors2d = c2._edgecolors3d
    # #
    # # # Third
    # # G2, G3 = np.meshgrid(g2, g3)
    # # G4 = (0.408248290463863 * G2 + 0.408248290463863 * G3 - 0.707106781186548) / 1.63299316185545
    # # c3 = ax.plot_surface(G2, G3, G4, label="c3")
    # # c3._facecolors2d = c3._facecolors3d
    # # c3._edgecolors2d = c3._edgecolors3d
    # #
    # # # And forth
    # # G4 = (1.04903810567666 - (0.288675134594813 * G2 + 0.288675134594813 * G3)) / 0.577350269189626
    # # c4 = ax.plot_surface(G2, G3, G4, label="c4")
    # #
    # # c4._facecolors2d = c4._facecolors3d
    # # c4._edgecolors2d = c4._edgecolors3d
    #
    # ax.legend()  # -> error : 'AttributeError: 'Poly3DCollection' object has no attribute '_edgecolors2d''
    #
    # # labeling the figure
    # fig.suptitle("Constraints")
    # # plt.xlabel('g2', fontsize=14)
    # # plt.ylabel('g3', fontsize=14)
    # ax.set_xlabel(r'$g_2$', fontsize=15, rotation=60)
    # ax.set_ylabel('$g_3$', fontsize=15, rotation=60)
    # ax.set_zlabel('$g_4$', fontsize=15, rotation=60)
    # plt.show()

    # # Smooth 3D plot https://stackoverflow.com/questions/35157650/smooth-surface-plot-with-pyplot
    # import numpy as np
    # from scipy import interpolate
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import axes3d, Axes3D
    #
    # X, Y = np.mgrid[-1:1:20j, -1:1:20j]
    # Z = (X + Y) * np.exp(-6.0 * (X * X + Y * Y)) + np.random.rand(X.shape[0])
    # X = np.array([[-4., -0.75, 2.5, 5.75, 9.],
    #                [-4., -0.75, 2.5, 5.75, 9.],
    #                [-4., -0.75, 2.5, 5.75, 9.],
    #                [-4., -0.75, 2.5, 5.75, 9.],
    #                [-4., -0.75, 2.5, 5.75, 9.]])
    # Y = np.array([[-3., -3., -3., -3., -3.],
    #           [0.5, 0.5, 0.5, 0.5, 0.5],
    #           [4., 4., 4., 4., 4.],
    #           [7.5, 7.5, 7.5, 7.5, 7.5],
    #           [11., 11., 11., 11., 11.]])
    # Z = np.array([[-2.82540909, -3.00641217, -1.73795643, -3.14751604, -3.72014602],
    #           [-1.01205317, -0.10801578, -0.25385137, -0.50081401, -1.46194579],
    #           [-1.00205789, -0.23398504, -0.17424708, -0.97058985, -2.33291644],
    #           [-2.08968439, -0.81389158, -2.41424369, -1.2806653, -3.97050193],
    #           [-2.37117982, -1.38764977, -3.55121739, -4.35633471, -5.9975255]])
    #
    # xnew, ynew = np.mgrid[-1:1:80j, -1:1:80j]
    # ic(xnew, ynew)
    # n = 81
    # # y_, x_ = np.meshgrid(np.linspace(-1, 1, num=n+1), np.linspace(-1, 1, num=n+1))  # Inversed
    # # ic(y_, x_)
    # # assert ynew == y_
    # # assert xnew == x_
    # # ynew, xnew = np.meshgrid(np.linspace(-1, 1, num=17), np.linspace(-1, 1, num=17))  # Inversed
    # # ic(xnew, ynew)
    # # xnew, ynew = x_, y_
    # # tck = interpolate.bisplrep(X, Y, Z, s=30)
    # # znew = interpolate.bisplev(xnew[:, 0], ynew[0, :], tck)
    #
    # # Here
    # X_ = X.flatten()
    # Y_ = Y.flatten()
    # Z_ = Z.flatten()
    # xi = np.linspace(X_.min(), X_.max(), num=10)
    # yi = np.linspace(Y_.min(), Y_.max(), num=12)
    # ic(xi, yi)
    # # xi = np.linspace(-1, 1, num=n+1)
    # # yi = np.linspace(-1, 1, num=n+1)
    # ic(X_.shape, Y_.shape, xi.shape, yi.shape, xi[None, :].shape, yi[:, None].shape,  Z_.shape)
    # znew = interpolate.griddata((X_, Y_), Z_, (xi[None, :], yi[:, None]), method='linear')
    # ic(X.shape, Y.shape, znew.shape)
    # ic(xi[None, :], xi.reshape(1, -1))
    # np.testing.assert_almost_equal(xi[None, :], xi.reshape(1, -1))
    #
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='summer', rstride=1, cstride=1, alpha=None)
    # plt.show()
    #
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.gca(projection='3d')
    # xnew, ynew = np.meshgrid(xi, yi)
    # ax.plot_surface(xnew, ynew, znew, cmap='summer', rstride=1, cstride=1, alpha=0.6, antialiased=True)
    # plt.show()
    # exit(1)

    # https://stackoverflow.com/questions/33287620/creating-a-smooth-surface-plot-from-topographic-data-using-matplotlib
    # import os
    # import numpy as np
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # from scipy.interpolate import griddata
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # # my_data = np.genfromtxt('2014_0.01_v3_HDF5.txt', delimiter=',', skip_header=1)
    # # my_data[my_data == 0] = np.nan
    # # my_data = my_data[~np.isnan(my_data).any(axis=1)]
    # my_data = []
    # n1, n2 = X.shape
    # for i in range(n1):
    #     for j in range(n2):
    #         my_data.append([X[i][j], Y[i][j], Z[i][j]])
    # my_data = np.array(my_data)
    # X = my_data[:, 0]
    # Y = my_data[:, 1]
    # Z = my_data[:, 2]
    # xi = np.linspace(X.min(), X.max(), num=int(len(Z) / 3))
    # yi = np.linspace(Y.min(), Y.max(), num=int(len(Z) / 3))
    # zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='nearest')
    #
    # xig, yig = np.meshgrid(xi, yi)
    #
    # surf = ax.plot_surface(xig, yig, zi, cmap='gist_earth')
    # ic(xig.shape, yig.shape, zi.shape)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_title('2014 ATM Data 0.01 Degree Spacing')
    # ax.set_xlabel('Latitude')
    # ax.set_ylabel('Longitude')
    # ax.set_zlabel('Elevation (m)')
    # # ax.set_zlim3d(0, 8000)
    # plt.show()
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='cubic')
    # xig, yig = np.meshgrid(xi, yi)
    #
    # surf = ax.plot_surface(xig, yig, zi, cmap='gist_earth')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_title('2014 ATM Data 0.01 Degree Spacing')
    # ax.set_xlabel('Latitude')
    # ax.set_ylabel('Longitude')
    # ax.set_zlabel('Elevation (m)')
    # # ax.set_zlim3d(0, 8000)
    # plt.show()

    # # https://stackoverflow.com/a/70156524/10732321
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # x = np.linspace(0, 1, 100)
    # X, Y = np.meshgrid(x, x)
    # Z1 = .1 * np.sin(2 * X) * np.sin(4 * Y)
    # Z2 = .1 * np.sin(3 * X) * np.sin(4 * Y)
    # Z3 = .1 * np.sin(4 * X) * np.sin(5 * Y)
    #
    # levels = np.linspace(Z1.min(), Z1.max(), 100)
    # ax.contourf(X, Y, Z1, levels=levels, zdir='z', offset=0, cmap=plt.get_cmap('rainbow'))
    #
    # levels = np.linspace(Z2.min(), Z2.max(), 100)
    # ax.contourf(X, Y, Z2, levels=levels, zdir='z', offset=1, cmap=plt.get_cmap('rainbow'))
    #
    # levels = np.linspace(Z3.min(), Z3.max(), 100)
    # ax.contourf(X, Y, Z3, levels=levels, zdir='z', offset=2, cmap=plt.get_cmap('rainbow'))
    #
    # ax.set_xlim3d(0, 1)
    # ax.set_ylim3d(0, 1)
    # ax.set_zlim3d(0, 2)
    #
    # plt.show()

    # # https://matplotlib.org/stable/gallery/mplot3d/pathpatch3d.html
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Circle, PathPatch
    # from matplotlib.text import TextPath
    # from matplotlib.transforms import Affine2D
    # import mpl_toolkits.mplot3d.art3d as art3d
    #
    #
    # def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
    #     """
    #     Plots the string *s* on the axes *ax*, with position *xyz*, size *size*,
    #     and rotation angle *angle*. *zdir* gives the axis which is to be treated as
    #     the third dimension. *usetex* is a boolean indicating whether the string
    #     should be run through a LaTeX subprocess or not.  Any additional keyword
    #     arguments are forwarded to `.transform_path`.
    #
    #     Note: zdir affects the interpretation of xyz.
    #     """
    #     x, y, z = xyz
    #     if zdir == "y":
    #         xy1, z1 = (x, z), y
    #     elif zdir == "x":
    #         xy1, z1 = (y, z), x
    #     else:
    #         xy1, z1 = (x, y), z
    #
    #     text_path = TextPath((0, 0), s, size=size, usetex=usetex)
    #     trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])
    #
    #     p1 = PathPatch(trans.transform_path(text_path), **kwargs)
    #     ax.add_patch(p1)
    #     art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # # Draw a circle on the x=0 'wall'
    # p = Circle((5, 5), 3)
    # ax.add_patch(p)
    # art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")
    #
    # # Manually label the axes
    # text3d(ax, (4, -2, 0), "X-axis", zdir="z", size=.5, usetex=False,
    #        ec="none", fc="k")
    # text3d(ax, (12, 4, 0), "Y-axis", zdir="z", size=.5, usetex=False,
    #        angle=np.pi / 2, ec="none", fc="k")
    # text3d(ax, (12, 10, 4), "Z-axis", zdir="y", size=.5, usetex=False,
    #        angle=np.pi / 2, ec="none", fc="k")
    #
    # # Write a Latex formula on the z=0 'floor'
    # text3d(ax, (1, 5, 0),
    #        r"$\displaystyle G_{\mu\nu} + \Lambda g_{\mu\nu} = "
    #        r"\frac{8\pi G}{c^4} T_{\mu\nu}  $",
    #        zdir="z", size=1, usetex=True,
    #        ec="none", fc="k")
    #
    # ax.set_xlim(0, 10)
    # ax.set_ylim(0, 10)
    # ax.set_zlim(0, 10)
    #
    # plt.show()
    # exit(1)

    # # https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals#comment26758349_18228967
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import proj3d
    # from matplotlib.patches import Circle, Rectangle
    # from itertools import product
    # from mpl_toolkits.mplot3d import art3d
    #
    # def rotation_matrix(d):
    #     """
    #     Calculates a rotation matrix given a vector d. The direction of d
    #     corresponds to the rotation axis. The length of d corresponds to
    #     the sin of the angle of rotation.
    #
    #     Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    #     """
    #     sin_angle = np.linalg.norm(d)
    #
    #     if sin_angle == 0:
    #         return np.identity(3)
    #
    #     d /= sin_angle
    #
    #     eye = np.eye(3)
    #     ddt = np.outer(d, d)
    #     skew = np.array([[0, d[2], -d[1]],
    #                      [-d[2], 0, d[0]],
    #                      [d[1], -d[0], 0]], dtype=np.float64)
    #
    #     M = ddt + np.sqrt(1 - sin_angle ** 2) * (eye - ddt) + sin_angle * skew
    #     return M
    #
    #
    # def pathpatch_2d_to_3d(pathpatch, z=0, normal='z'):
    #     """
    #     Transforms a 2D Patch to a 3D patch using the given normal vector.
    #
    #     The patch is projected into they XY plane, rotated about the origin
    #     and finally translated by z.
    #     """
    #     if type(normal) is str:  # Translate strings to normal vectors
    #         index = "xyz".index(normal)
    #         # ic(normal, np.roll((1.0, 0, 0), index))
    #         normal = np.roll((1.0, 0, 0), index)
    #
    #     normal /= np.linalg.norm(normal)  # Make sure the vector is normalised
    #
    #     path = pathpatch.get_path()  # Get the path and the associated transform
    #     trans = pathpatch.get_patch_transform()
    #
    #     path = trans.transform_path(path)  # Apply the transform
    #
    #     pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    #     pathpatch._code3d = path.codes  # Copy the codes
    #     pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color
    #
    #     verts = path.vertices  # Get the vertices in 2D
    #
    #     d = np.cross(normal, (0, 0, 1))  # Obtain the rotation vector
    #     M = rotation_matrix(d)  # Get the rotation matrix
    #     # ic(normal, d, M)
    #
    #     pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])
    #
    #
    # def pathpatch_translate(pathpatch, delta):
    #     """
    #     Translates the 3D pathpatch by the amount delta.
    #     """
    #     pathpatch._segment3d += delta
    #     # ic(pathpatch._segment3d)
    #
    #
    # fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='3d'))
    # # ax = axes(projection='3d')  # Create axes
    #
    # p = Circle((0, 0), .2)  # Add a circle in the yz plane
    # ax.add_patch(p)
    # pathpatch_2d_to_3d(p, z=0.5, normal='x')
    # pathpatch_translate(p, (0, 0.5, 0))
    #
    # p = Circle((0, 0), .2, facecolor='r')  # Add a circle in the xz plane
    # ax.add_patch(p)
    # pathpatch_2d_to_3d(p, z=0.5, normal='y')
    # pathpatch_translate(p, (0.5, 1, 0))
    #
    # p = Circle((0, 0), .2, facecolor='g')  # Add a circle in the xy plane
    # ax.add_patch(p)
    # pathpatch_2d_to_3d(p, z=0, normal='z')
    # pathpatch_translate(p, (0.5, 0.5, 0))
    #
    # p = Rectangle((0, 0), 0.25, 1, fc='pink', alpha=0.5)
    # ax.add_patch(p)
    # # pathpatch_2d_to_3d(p, z=0, normal='y')
    # pathpatch_2d_to_3d(p, z=0, normal=(0, 0.1, 0))
    # pathpatch_translate(p, (0, 0, 1))
    # tsf = tsl_n_angle2tsf([0, 0, -0.15])
    # ic(p._segment3d)
    # ic(p._segment3d[:, :2].shape)
    # p._segment3d[:, :2] = apply_tsf_2d(p._segment3d, tsf)
    #
    # for normal in product((-1, 1), repeat=3):
    #     # ic(normal)
    #     p = Circle((0, 0), .2, facecolor='y', alpha=.2)
    #     ax.add_patch(p)
    #     pathpatch_2d_to_3d(p, z=0, normal=normal)
    #     pathpatch_translate(p, 0.5)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

    # if dict():
    #     ic()

    # tsf = tsl_n_angle2tsf(config('heuristics.pose_guess.actual_pose'))
    # ic(tsf, np.linalg.inv(tsf))

    # https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    import numpy as np
    from numpy import *
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d


    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        # def draw(self, renderer):
        #     xs3d, ys3d, zs3d = self._verts3d
        #     xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        #     self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        #     FancyArrowPatch.draw(self, renderer)
        #
        # def do_3d_projection(self, dummy):
        #     pass

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

            return np.min(zs)


    ####################################################
    # This part is just for reference if
    # you are interested where the data is
    # coming from
    # The plot is at the bottom
    #####################################################

    # Generate some example data
    mu_vec1 = np.array([0, 0, 0])
    cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)

    mu_vec2 = np.array([1, 1, 1])
    cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)

    # concatenate data for PCA
    samples = np.concatenate((class1_sample, class2_sample), axis=0)

    # mean values
    mean_x = mean(samples[:, 0])
    mean_y = mean(samples[:, 1])
    mean_z = mean(samples[:, 2])

    # eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(cov_mat1)

    ################################
    # plotting eigenvectors
    ################################

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(samples[:, 0], samples[:, 1], samples[:, 2], 'o', markersize=10, color='g', alpha=0.2)
    ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
    for v in eig_vec:
        # ax.plot([mean_x,v[0]], [mean_y,v[1]], [mean_z,v[2]], color='red', alpha=0.8, lw=3)
        # I will replace this line with:
        a = Arrow3D([mean_x, v[0]], [mean_y, v[1]],
                    [mean_z, v[2]], mutation_scale=20,
                    lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')

    plt.title('Eigenvectors')

    plt.draw()
    plt.show()

    # https://github.com/matplotlib/matplotlib/issues/21688
    import numpy as np
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d


    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

            return np.min(zs)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
    a = Arrow3D([0, 10], [0, 0], [0, 0], **arrow_prop_dict)
    ax.add_artist(a)

    plt.show()
