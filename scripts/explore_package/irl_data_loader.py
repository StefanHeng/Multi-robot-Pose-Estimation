"""
DatasetL Intel Research Lab (Seattle), raw log data provided by Dirk Hähnel

Get the laser scan data
"""

import numpy as np
from math import pi
from icecream import ic

from scripts.util import laser_polar2planar


class DataLoader:
    N_LSBM = 180  # Number of beams in a single laser measurement
    ANGLE_MAX = -pi / 2
    ANGLE_MIN = pi / 2
    ANGLE_INC = pi / N_LSBM

    def __init__(self, fnm='Intel Research Lab, SLAM raw.txt', prefix='.'):
        f = open(f'{prefix}/{fnm}', 'r')
        lines = list(map(lambda l: l.split(' '), f.readlines()))
        lines = list(filter(lambda l: l[0] == 'FLASER', lines))
        idx_strt = 2
        ranges = list(map(lambda l: np.array(list(map(float, l[idx_strt: idx_strt+self.N_LSBM]))), lines))
        self.ranges = np.vstack(ranges)
        self.range2polar = laser_polar2planar(self.ANGLE_MAX, self.ANGLE_MIN)

    def __getitem__(self, idx):
        """
        :return: Planar representation of a measurement
        """
        return self.range2polar(self.ranges[idx])


if __name__ == '__main__':
    dl = DataLoader()
    ic(dl[180][:10], dl[186][:10])
