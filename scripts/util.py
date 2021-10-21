import numpy as np
import json
from math import pi
from functools import reduce

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from icecream import ic

sns.set_style('darkgrid')


def arr_idx(a, v):
    """
    :return: 1st occurrence index of `v` in `a`, a numpy 1D array
    """
    return np.where(a == v)[0][0]


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

    pc = get_rect_pointcloud(2, 0.8, visualize=False)
    plt.figure(figsize=(16, 9), constrained_layout=True)
    plt.plot(pc[:, 0], pc[:, 1], marker='o', ms=1, lw=0.5)
    plt.show()

