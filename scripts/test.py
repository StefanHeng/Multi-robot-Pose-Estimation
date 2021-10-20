import numpy as np
from icecream import ic

from explore_package.irc_laser_data_eg import src_pts, tgt_pts
from explore_package.Align2D import Align2D
from robo_pose_estimator import PoseEstimator
from explore_package.irl_data_loader import DataLoader


class TestIcp:
    def __init__(self):
        self.mine = PoseEstimator.Icp
        self.ori = Align2D

    def __call__(self, src, tgt):
        m = self.mine(src[:, :2], tgt[:, :2])()
        ic()
        o = self.ori(src, tgt, np.identity(3)).transform
        np.testing.assert_array_almost_equal(m, o, decimal=12)


if __name__ == '__main__':
    np.random.seed(77)
    # ic(src_pts.shape, tgt_pts.shape)
    ti = TestIcp()
    # ti(src_pts, tgt_pts)

    dl = DataLoader(prefix='explore_package')

    def _test(i_strt, i_end):
        src = ti.mine.extend_1s(dl[i_strt])
        tgt = ti.mine.extend_1s(dl[i_end])
        ti(src, tgt)

    # # n_max = 10
    # n_max = 6
    # n = 100
    # idx_strt = np.random.randint(dl.ranges.shape[0] - n_max, size=n)
    # span = np.random.randint(1, high=n_max, size=n)
    # for (i_s, s) in np.vstack([idx_strt, span]).T:
    #     ic(i_s, s)
    #     _test(i_s, i_s+s)

    _test(11644, 11644 + 4)
