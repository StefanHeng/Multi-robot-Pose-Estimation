import glob
import json
import os.path
from math import pi, acos, degrees, sqrt
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib import transforms, rcParams
from matplotlib.widgets import Button
import seaborn as sns
import pint
from icecream import ic

from scripts.data_path import *

rcParams['figure.constrained_layout.use'] = True
sns.set_style('darkgrid')


reg = pint.UnitRegistry()


def unit_converter(m, src=reg.inch, dst=reg.meter):
    return (m * src).to(dst).magnitude


def json_load(fnm):
    f = open(fnm, 'r')
    scans = json.load(f)
    f.close()
    return scans


def get(dic, ks):
    # return reduce(lambda acc, elm: acc[elm], keys, dic)
    return reduce(lambda acc, elm: acc[elm] if elm in acc else None, ks.split('.'), dic)


def config(attr):
    """
    Retrieves the queried attribute value from the config file.

    Loads the config file on first call.
    """
    if not hasattr(config, 'config'):
        with open(f'{PATH_BASE}/{DIR_PROJ}/config.json') as f:
            config.config = json.load(f)
    return get(config.config, attr)


def eg_hsr_scan(k1=0, k2=77):
    """
    :return: Example HSR laser scan as 2D coordinates, given file name and measurement number
    """
    path = os.path.join(PATH_BASE, DIR_DATA)
    fls = sorted(glob.iglob(f'{path}/{config(f"{DIR_DATA}.eg.HSR.fmt")}', recursive=True))
    hsr_scans = json_load(fls[k1])
    s = hsr_scans[k2]
    return laser_polar2planar(s['angle_max'], s['angle_min'])(np.array(s['ranges']))


def pts2max_dist(pts):
    """
    :param pts: List of 2d points
    :return: The maximum distance between any two pairs of points
    """
    assert pts.shape[1] == 2

    def dist(a, b):
        return (a[0] - b[0])**2 + (a[1] - b[1])**2
    n = pts.shape[0]
    idxs = ((i, j) for i in range(n-1) for j in range(i, n))
    # ic(list(idxs))
    return sqrt(max(dist(pts[a], pts[b]) for a, b in idxs))


def get_kuka_pointcloud():
    d_dim = config('dimensions.KUKA')
    return get_rect_pointcloud(d_dim['length'], d_dim['width'])


def clipper(low, high):
    """
    :return: A clipping function for range [low, high]
    """
    return lambda x: max(min(x, high), low)


def get_3rd_side(a, b):
    """
    Returns hypotenuse of a right-angled triangle, given it's other sides
    """
    # return np.sqrt(np.sum(np.square(mags)))
    return sqrt(a**2 + b**2)


class JsonWriter:
    """
    Each time the object is called, the data is appended to the end of a list which is serialized to a JSON file
    """

    def __init__(self, fnm):
        self.fnm = fnm
        self.data = []
        self.fnm_ext = f'data/{self.fnm}.json'
        open(self.fnm_ext, 'a').close()  # Create file in OS

    def __call__(self, data):
        f = open(self.fnm_ext, 'w')
        self.data.append(data)
        json.dump(self.data, f, indent=4)
        # ic(self.data)


def laser_scan2dict(data):
    """
    :param data: Of type [sensor_msgs/LaserScan](https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html)
    """
    h = data.header
    d_h = dict(
        seq=h.seq,
        stamp=dict(
            secs=h.stamp.secs,
            nsecs=h.stamp.nsecs
        ),
        frame_id=h.frame_id
    )
    return dict(
        header=d_h,
        angle_min=data.angle_min,
        angle_max=data.angle_max,
        angle_increment=data.angle_increment,
        time_increment=data.time_increment,
        scan_time=data.scan_time,
        range_min=data.range_min,
        ranges=data.ranges,
        intensities=data.intensities
    )


def extend_1s(arr):
    """
    Return array with column of 1's appended
    :param arr: 2D array
    """
    return np.hstack([arr, np.ones([arr.shape[0], 1])])


def cartesian(arrs: list, out=None):
    """
    :param arrs: list of 1D arrays
    :param out: Array to place the cartesian product in.
    :return: Cartesian product of `arrs` of shape

    Modified from https://stackoverflow.com/a/1235363/10732321
    """
    arrs = [np.asarray(x) for x in arrs]
    n = np.prod([x.size for x in arrs])
    if out is None:
        out = np.zeros([n, len(arrs)], dtype=arrs[0].dtype)

    m = int(n / arrs[0].size)
    out[:, 0] = np.repeat(arrs[0], m)
    if arrs[1:]:
        cartesian(arrs[1:], out=out[0:m, 1:])
        for j in range(1, arrs[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def polar2planar(dist, angle):
    return (
        dist * np.cos(angle),
        dist * np.sin(angle)
    )


def laser_polar2planar(a_max, a_min, split=False):
    """
    :param a_max: Maximum angle
    :param a_min: Minimum angle
    :param split: If True, the function returns a 2-tuple of x and y coordinates
    :return: A function that returns an array of 2D points

    Assumes the angles are [a_min, a_max)
    """
    def _get(ranges):
        """
        :param ranges: Array of laser scan ranges

        Number of beams taken from size of `range`;
        """
        theta = np.linspace(a_min, a_max, num=ranges.size + 1)[:-1]
        x, y = polar2planar(ranges, theta)
        return (x, y) if split else np.vstack([x, y]).T
    return _get


def rot_mat(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s],
        [s, c]
    ])


def tsl_n_angle2tsf(tsl=np.array([0, 0]), theta=0):
    """
    Converts translation in 2D & an angle into matrix transformation

    :param tsl: 3-array of (translation_x, translation_y, theta),
        or 2-array of (translation_x, translation_y)
    :param theta: Angle in radians
    """
    tsf = np.identity(3)
    tsf[:2, 2] = tsl[:2]
    # ic(tsl[-1] if tsl.size == 3 else theta)
    tsf[:2, :2] = rot_mat(tsl[-1] if tsl.size == 3 else theta)
    return tsf


def tsf2tsl_n_angle(tsf):
    """
    :return: 2-tuple of 2D translation and angle in radians from transformation matrix
    """
    return tsf[:2, 2], acos(tsf[0][0])


def get_rect_pointcloud(w, h, n=240, visualize=False):
    """
    :param w: Width of rectangle
    :param h: Height of rectangle
    :param n: Number of points/beams
    :param visualize: If True, shows an illustration of the process 
    :return: Array of 2D points of a rectangular contour, as if by a 360 degree of beams
    """
    r = max(w, h)
    r = np.full(n, r)
    theta = np.linspace(0, 2 * pi, num=n+1)[:-1]
    x, y = polar2planar(r, theta)
    boundaries = (-w/2, -h/2, w/2, h/2)

    def intersec_rect(left, bot, right, top):
        """ :return: function that returns the intersection of point relative to a rectangle """
        def _get(x_, y_):
            """
            x, y should be outside of the rectangle
            """
            ct_x = (left + right) / 2
            ct_y = (bot + top) / 2
            slope = (ct_y - y_) / (ct_x - x_)

            if x_ <= ct_x:
                y__ = slope * (left - x_) + y_
                if bot <= y__ <= top:
                    return left, y__
            if x_ >= ct_x:
                y__ = slope * (right - x_) + y_
                if bot <= y__ <= top:
                    return right, y__
            if y_ <= ct_y:
                x__ = (bot - y_) / slope + x_
                if left <= x__ <= right:
                    return x__, bot
            if y_ >= ct_y:
                x__ = (top - y_) / slope + x_
                if left <= x__ <= right:
                    return x__, top
            if x_ == ct_x and y_ == ct_y:
                return x_, y_
        return _get

    if visualize:
        fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
        for x_i, y_i in zip(x, y):
            x_int, y_int = intersec_rect(*boundaries)(x_i, y_i)
            ax.add_patch(Rectangle((-w/2, -h/2), w, h, edgecolor='b', fill=False))
            ax.plot((0, x_int), (0, y_int), marker='o', c='c', ms=2, lw=0.5, ls='dotted')
            ax.plot((x_i, x_int), (y_i, y_int), marker='o', c='orange', ms=2, ls='dotted')
        plt.gca().set_aspect('equal')
        plt.show()
    intersec = intersec_rect(*boundaries)
    return np.apply_along_axis(lambda i: intersec(*i), 1, np.vstack([x, y]).T)


def save_fig(save, title):
    if save:
        fnm = f'{title}.png'
        plt.savefig(os.path.join(PATH_BASE, DIR_PROJ, 'plot', fnm), dpi=300)


def plot_icp_result(src, tgt, tsf, title=None, save=False, states=None, xlim=None, ylim=None, with_arrow=True,
                    init_tsf=np.identity(3), mode='static', scale=1):
    """
    :param src: Source coordinates
    :param tgt: Target coordinates
    :param tsf: ICP result transformation
    :param title: Plot title
    :param save: If true, plot saved as image
    :param states: A list of source-target matched points & transformation for each iteration
    :param xlim: X limit for plot, inferred if not given
    :param ylim: Y limit for plot, inferred if not given
    :param init_tsf: Initial transformation guess for ICP
    :param mode: Plotting mode, one of [`static`, `animate`, `control`]
    :param scale: Plot window scale

    .. note:: Assumes 2d data
    """
    def _plot_point_cloud(arr, **kwargs):
        kwargs_ = dict(
            c='orange',
            marker='.',
            ms=1,
            lw=0.5,
        )
        plt.plot(arr[:, 0], arr[:, 1], **(kwargs_ | kwargs))

    def _plot_line_seg_arrow(c1, c2, r=0.01, **kwargs):
        coords = np.array([c1, c2])
        mean = coords.mean(axis=0)
        mags = (coords[1] - coords[0]) * r

        width = 5 * get_3rd_side(*mags)
        if not hasattr(plot_icp_result, 'clp'):
            plot_icp_result.clp = clipper(0.01, 0.05)
        width = plot_icp_result.clp(width)

        kwargs_ = dict(
            alpha=0.5,
            # head_width=0.05,
            head_width=width,
            length_includes_head=True,
            lw=0,
            overhang=0.2,
        )
        plt.arrow(
            *(mean-mags/2), *mags,
            **(kwargs_ | kwargs)
        )

    def _plot_line_seg(c1, c2, **kwargs):
        kwargs_ = dict(
            marker='o',
            c='orange',
            ms=2,
            lw=1,
            ls='dotted'
        )
        kwargs = kwargs_ | kwargs
        plt.plot((c1[0], c2[0]), (c1[1], c2[1]), **kwargs)
        if with_arrow:
            _plot_line_seg_arrow(c1, c2, color=get(kwargs, 'c'), alpha=get(kwargs, 'alpha'))

    def _plot_matched_points(stt, **kwargs):
        src__, tgt__ = stt[0], stt[1]
        for s_, t_ in zip(src__, tgt__):
            _plot_line_seg(s_, t_, **kwargs)

    N_STT = len(states)

    x_rang, y_rang = (
        abs(xlim[0] - xlim[1]), abs(ylim[0] - ylim[1])
    ) if xlim and ylim else (
        np.ptp(tgt[:, 0]), np.ptp(tgt[:, 1])
    )
    ratio = 1 / x_rang * y_rang
    d = 12 * scale
    plt.figure(figsize=(d, d * ratio), constrained_layout=True)
    plt.xlabel('Target space dim 1 (m)')
    plt.ylabel('Target space dim 2 (m)')
    t = 'ICP results'
    if title:
        t = f'{t}, {title}'
    plt.suptitle(t)
    plt.gca().set_aspect('equal')

    def _step(idx_, t_):
        tsf_ = states[idx_][-1] if states else tsf
        plt.cla()
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        if mode == 'control':
            t_ = f'{t_}, iteration {idx_}'
            plt.suptitle(t_)

        tsl, theta = tsf2tsl_n_angle(tsf_)
        ic(tsf_, tsl, degrees(theta))

        unit_sqr = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0]
        ])
        unit_sqr_tsf = (extend_1s(unit_sqr) @ tsf_.T)[:, :2]

        cs = iter(sns.color_palette(palette='husl', n_colors=5))
        c = next(cs)
        _plot_point_cloud(src, c=c, alpha=0.5, label='Source points')
        if not np.array_equal(init_tsf, np.identity(3)):
            _plot_point_cloud(src @ init_tsf.T, c=c, alpha=0.5, label='Source points, initial guess')
        _plot_point_cloud(src @ tsf_.T, c=c, label='Source points, transformed')
        _plot_point_cloud(tgt, c=next(cs), label='Target points')

        c = next(cs)
        if states:
            _plot_matched_points(states[0], c=c, ms=1, ls='solid', alpha=0.5, label='Matched points, initial')
            _plot_matched_points(states[idx_], c=c, ms=1, ls='solid', label='Matched points, final')

        c = next(cs)
        plt.plot(0, 0, marker='o', c=c, ms=4)
        _plot_point_cloud(unit_sqr, ms=0, marker=None, c=c, alpha=0.6, label='Unit square')
        _plot_point_cloud(unit_sqr_tsf, ms=0.5, marker=None, c=c, alpha=0.9, label='Unit square, transformed')
        for i in zip(unit_sqr, unit_sqr_tsf):
            _plot_line_seg(*i, marker=None, c=c, alpha=0.5)

        handles, labels = plt.gca().get_legend_handles_labels()  # Distinct labels
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        save_fig(save, t_)

        if mode != 'static':
            plt.pause(1 if mode == 'animate' else 0.1)  # 'control'

    if mode == 'animate':
        plt.ion()
        for idx in range(N_STT):
            _step(idx, t)
    elif mode == 'control':
        class PlotFrame:
            def __init__(self, i=0):
                self.idx = i
                self.clp = clipper(0, N_STT-1)

            def next(self, event):
                prev_idx = self.idx
                self.idx = self.clp(self.idx+1)
                if prev_idx != self.idx:
                    _step(self.idx, t)

            def prev(self, event):
                prev_idx = self.idx
                self.idx = self.clp(self.idx-1)
                if prev_idx != self.idx:
                    _step(self.idx, t)
        init = 0
        pf = PlotFrame(i=init)
        ax = plt.gca()
        btn_next = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Next')
        btn_next.on_clicked(pf.next)
        btn_prev = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Previous')
        btn_prev.on_clicked(pf.prev)
        plt.sca(ax)

        _step(init, t)
    else:
        _step(N_STT-1, t)

    plt.ioff()  # So that window doesn't close
    plt.show()


def plot_cluster(data, labels, title=None, save=False):
    d_clusters = {lb: data[np.where(labels == lb)] for lb in np.unique(labels)}

    cs = iter(sns.color_palette(palette='husl', n_colors=len(d_clusters) + 1))
    x, y = data[:, 0], data[:, 1]
    fig, ax = plt.subplots(figsize=(12, 12 / np.ptp(x) * np.ptp(y)), constrained_layout=True)
    plt.plot(x, y, marker='o', ms=0.3, lw=0.25, c=next(cs), alpha=0.5, label='Whole')

    for lb, d in d_clusters.items():
        x_, y_ = d[:, 0], d[:, 1]
        c = next(cs)

        def confidence_ellipse(n_std=1., **kwargs):
            """
            Modified from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
            Create a plot of the covariance confidence ellipse of x and y

            :param n_std: number of standard deviations to determine the ellipse's radius'
            :return matplotlib.patches.Ellipse
            """
            cov = np.cov(x_, y_)
            pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)
            ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                              **(dict(fc='none') | kwargs))

            tsf = transforms.Affine2D().rotate_deg(45)
            tsf = tsf.scale(
                np.sqrt(cov[0, 0]) * n_std,
                np.sqrt(cov[1, 1]) * n_std
            )
            tsf = tsf.translate(np.mean(x_), np.mean(y_))

            ellipse.set_transform(tsf + ax.transData)
            return ax.add_patch(ellipse)

        if lb != -1:  # Noise as in DBSCAN
            confidence_ellipse(n_std=1.25, fc=c, alpha=0.25)

        lb = f'Cluster {lb + 1}'
        # plt.scatter(x_[:, 0], x_[:, 1], marker='.', s=2, label=lb)
        ax.plot(x_, y_, marker='o', ms=0.5, lw=0.25, c=c, label=lb)

    t = 'Clustering results'
    if title:
        t = f'{t}, {title}'
    plt.title(t)
    plt.legend()
    plt.gca().set_aspect('equal')
    save_fig(save, title)
    plt.show()


if __name__ == '__main__':
    def _create():
        jw = JsonWriter('jw')

        for d in [1, 'as', {1: 2, "a": "b"}, [12, 4]]:
            jw(d)

    def _check():
        with open('jw.json') as f:
            l = json.load(f)
            ic(l)

    # _create()
    # _check()

    def _kuka_pointcloud():
        # pc = get_rect_pointcloud(2, 0.8, visualize=False)
        ptc = get_kuka_pointcloud()
        plt.figure(figsize=(16, 9), constrained_layout=True)
        plt.plot(ptc[:, 0], ptc[:, 1], marker='o', ms=1, lw=0.5)
        plt.show()

    # ic(config('dimensions.KUKA.length'))
    ic(cartesian([[1, 2, 3], [4, 5], [6, 7]]))
