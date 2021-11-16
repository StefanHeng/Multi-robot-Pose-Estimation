import numpy as np
import json
from math import pi, acos
from functools import reduce

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.transforms as transforms
import seaborn as sns
from icecream import ic

sns.set_style('darkgrid')


def json_load(fnm):
    f = open(fnm, 'r')
    scans = json.load(f)
    f.close()
    return scans


def get(dic, keys):
    return reduce(lambda acc, elm: acc[elm], keys, dic)


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
    # ic(arr_x, arr_y)

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
            # x, y = np.random.randint(1, high=4, size=2)
            x_int, y_int = intersec_rect(*boundaries)(x_i, y_i)
            # ic(x, y)
            # ic(x_int, y_int)
            ax.add_patch(Rectangle((-w/2, -h/2), w, h, edgecolor='b', fill=False))
            ax.plot((0, x_int), (0, y_int), marker='o', c='c', ms=2, lw=0.5, ls='dotted')
            ax.plot((x_i, x_int), (y_i, y_int), marker='o', c='orange', ms=2, ls='dotted')
            # ax.plot(x_int, y_int, marker='o', c='orange', ms=4)
        plt.gca().set_aspect('equal')
        plt.show()
    intersec = intersec_rect(*boundaries)
    return np.apply_along_axis(lambda i: intersec(*i), 1, np.vstack([x, y]).T)


def plot_icp_result(src, tgt, tsf, title=None, save=False, states=None, xlim=None, ylim=None,
                    init_tsf=np.identity(3), animate=False):
    """
    Assumes 2d data
    """
    if animate:
        plt.ion()

    def _plot_point_cloud(arr, label, **kwargs):
        kwargs_ = dict(
            c='orange',
            marker='.',
            ms=1,
            lw=0.5,
        )
        plt.plot(arr[:, 0], arr[:, 1], label=label, **(kwargs_ | kwargs))

    def _plot_line_seg(c1, c2, **kwargs):
        kwargs_ = dict(
            marker='o',
            c='orange',
            ms=2,
            lw=1,
            ls='dotted'
        )
        plt.plot((c1[0], c2[0]), (c1[1], c2[1]), **(kwargs_ | kwargs))

    def _plot_matched_points(stt, **kwargs):
        src__, tgt__ = stt[0], stt[1]
        for s_, t_ in zip(src__, tgt__):
            _plot_line_seg(s_, t_, **kwargs)

    def _step(tsf_, states_):
        plt.cla()
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

        ori_tsl = tsf_[:2, 2]
        angle = acos(tsf_[0][0])
        ic(tsf_, ori_tsl, angle)
        unit_sqr = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0]
        ])
        unit_sqr_tsf = (extend_1s(unit_sqr) @ tsf_.T)[:, :2]
        plt.plot(0, 0, marker='o', c='orange', ms=4)
        _plot_point_cloud(unit_sqr, 'unit square', alpha=0.5, ms=0, marker=None)
        _plot_point_cloud(unit_sqr_tsf, 'unit square, ICP output', alpha=0.8, ms=0.5, marker=None)
        for i in zip(unit_sqr, unit_sqr_tsf):
            _plot_line_seg(*i, alpha=0.3, marker=None)
        if states_:
            _plot_matched_points(states_[0], c='g', ms=1, ls='solid', alpha=0.5)
            _plot_matched_points(states_[-1], c='g', ms=1, ls='solid')

        _plot_point_cloud(src, 'source', c='c', alpha=0.5)
        if not np.array_equal(init_tsf, np.identity(3)):
            _plot_point_cloud(src @ init_tsf.T, 'source, initial guess', c='c', alpha=0.5)
        _plot_point_cloud(tgt, 'target', c='m')
        _plot_point_cloud(src @ tsf_.T, 'source, transformed', c='c')

        handles, labels = plt.gca().get_legend_handles_labels()  # Distinct labels
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.pause(1)

    x_rang, y_rang = (
        abs(xlim[0] - xlim[1]), abs(ylim[0] - ylim[1])
    ) if xlim and ylim else (
        np.ptp(tgt[:, 0]), np.ptp(tgt[:, 1])
    )
    ratio = 1 / x_rang * y_rang
    ic(xlim, ylim, ratio)
    plt.figure(figsize=(12, 12 * ratio), constrained_layout=True)
    # exit(1)
    t = 'ICP results'
    if title:
        t = f'{t}, {title}'
    plt.suptitle(t)
    plt.gca().set_aspect('equal')
    if animate:
        for idx, (src_match, tgt_match, tsf) in enumerate(states):
            _step(tsf, states[:idx])
    else:
        _step(tsf, states)

    if save:
        plt.savefig(f'plot/{t}.png', dpi=300)

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
    if save:
        plt.savefig(f'plot/{t}.png', dpi=300)
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
        pc = get_rect_pointcloud(2, 0.8, visualize=False)
        plt.figure(figsize=(16, 9), constrained_layout=True)
        plt.plot(pc[:, 0], pc[:, 1], marker='o', ms=1, lw=0.5)
        plt.show()
