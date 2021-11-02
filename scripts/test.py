import numpy as np
from icecream import ic

from util import *
from explore_package.irc_laser_data_eg import src_pts, tgt_pts
from explore_package.Align2D import Align2D
from robo_pose_estimator import Icp
from explore_package.irl_data_loader import DataLoader


class TestIcp:
    def __init__(self):
        self.mine = Icp
        self.ori = Align2D

    def __call__(self, src, tgt):
        m = self.mine(src[:, :2], tgt[:, :2])()
        ic()
        o = self.ori(src, tgt, np.identity(3)).transform
        np.testing.assert_array_almost_equal(m, o, decimal=12)


if __name__ == '__main__':
    np.random.seed(7)
    # ic(src_pts.shape, tgt_pts.shape)
    ti = TestIcp()
    # ti(src_pts, tgt_pts)

    dl = DataLoader(prefix='explore_package')

    def _test(i_strt, i_end):
        src = extend_1s(dl[i_strt])
        tgt = extend_1s(dl[i_end])
        ti(src, tgt)

    # n_max = 10
    n_max = 6
    n = 100
    n_measure = dl.ranges.shape[0]
    counts = np.arange(n)
    idx_strt = np.random.randint(n_measure - n_max, size=n)
    span = np.random.randint(1, high=n_max, size=n)
    for (count, i_s, s) in np.vstack([counts, idx_strt, span]).T:
        ic(count, i_s, s)
        _test(i_s, i_s+s)

    # _test(11644, 11644 + 4)  # An example with seed 77
    # _test(6600, 6600 + 4)  # An example with seed 7
