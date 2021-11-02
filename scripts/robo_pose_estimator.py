import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from icecream import ic

from notebooks.util import *
from util import *


class Icp:
    """
    Implementation of ICP algorithm in 2D
    Given source and target point clouds, returns the translation matrix T s.t. T(source) ~= target
    """

    def __init__(self, src, tgt):
        """
        :param src:  Array of source points
        :param tgt: Array of target points
        """
        self.src = extend_1s(src)
        self.tgt = extend_1s(tgt)
        self.tree_tgt = KDTree(tgt)

    def __call__(self, tsf=np.identity(3), max_iter=20, min_d_err=1e-4, lst_match=None, match=False):
        """
        :param tsf: Initial transformation estimate
        :param max_iter: Max iteration, stopping criteria
        :param min_d_err: Minimal change in error, stopping criteria
        :param lst_match: List of source-target matched points in each iteration
        :param match: If true, keep track of source-target matched points in each iteration
        """
        err = float('inf')
        d_err = float('inf')
        n = 0
        # src = self.src @ tsf.T
        src = self.src
        # ic(tsf)

        # if match:
        #     lst_match = []

        while d_err > min_d_err and n < max_iter:
            src_match, tgt_match, idxs = self.nn_tgt(src)
            if match:
                lst_match.append((src_match, tgt_match))
            tsf = tsf @ self.svd(src_match, tgt_match)
            src = self.src @ tsf.T

            def _err():
                src_ = src[idxs[:, 0]]
                return np.sum(np.square(src_[:, :2] - tgt_match)) / idxs.shape[0]

            err_ = _err()
            d_err = abs(err - err_)
            err = err_
            n += 1
        return tsf

    def nn_tgt(self, src):
        """
        Finds nearest neighbors in the target
        """
        dist, idxs_ = self.tree_tgt.query(src[:, :2])
        # Keep pairs with distinct target points by removing pairs with larger distance
        idxs_pair = np.vstack([np.arange(idxs_.size), idxs_]).T  # Each element of [src_idx, tgt_idx]
        arg_idxs = dist.argsort()
        idxs_pair_sort = idxs_pair[arg_idxs]
        idxs_sort = idxs_[arg_idxs]

        def _get_idx(arr):
            def _get(i):
                """
                If arr[i[ is not the first occurrence, it's set to -1
                """
                i_ = arr_idx(arr, arr[i])
                return arr[i] if i_ == i else -1

            return _get

        idxs_1st_occ = np.vectorize(_get_idx(idxs_sort))(np.arange(idxs_sort.size))
        idxs_pair_uniq = idxs_pair_sort[np.where(idxs_1st_occ != -1)]
        return (
            src[idxs_pair_uniq[:, 0]][:, :2],
            self.tgt[idxs_pair_uniq[:, 1]][:, :2],
            idxs_pair_uniq
        )

    @staticmethod
    def svd(src, tgt):
        """
        Singular value decomposition for points in 2D

        :return: T: transformation matrix
        """

        def _centroid(pts):
            return np.sum(pts, axis=0) / pts.shape[0]

        c_src = _centroid(src)
        c_tgt = _centroid(tgt)
        src_c = src - c_src
        tgt_c = tgt - c_tgt
        M = tgt_c.T @ src_c
        U, W, V_t = np.linalg.svd(M)
        rot_mat = U @ V_t
        tsl = c_tgt - rot_mat @ c_src
        tsf = np.identity(3)
        tsf[:2, 2] = tsl
        tsf[:2, :2] = rot_mat
        return tsf


class PoseEstimator:
    """
    Various laser-based pose estimation algorithms between KUKA iiwa and HSR robot
    """
    L_KUKA = 2
    W_KUKA = 0.8

    class FusePose:
        """
        1) For each robot `r`, get its multiple pose estimates of the other robot `r_`
        based on `r_`'s point-cloud representation

        2) Pick the pair that makes sense as relative pose
        Note that those should be inverse of each other
        """
        def __init__(self, pc_a=None, pc_b=None, rg_a=None, rg_b=None):
            """
            :param pc_a: Point-cloud representation of `robot_a`
            :param pc_b: Point-cloud representation of `robot_b`
            :param rg_a: Range of laser sensor of `robot_a`
            :param rg_b: Range of laser sensor of `robot_b`
            """
            self.pc_a = pc_a
            self.pc_b = pc_b
            self.rg_a = rg_a
            self.rg_b = rg_b

        def __call__(self, pts_a=None, pts_b=None):
            # tsf = Icp(pts_a, self.pc_b)()
            # plot_icp_result(extend_1s(pts_a), self.pc_b, tsf, title='default init', save=True)

            def visualize(a, b, title, tsf=np.identity(3)):
                l_m = []
                tsf = Icp(a, b)(tsf=tsf, lst_match=l_m, match=True, max_iter=100, min_d_err=1e-6)
                ic('final ICP output', tsf)
                plot_icp_result(extend_1s(a), b, tsf, title=title, save=True, lst_match=l_m, split=False)

            # visualize(pts_a, self.pc_b, 'default init from HSR')
            # visualize(self.pc_b, pts_a, 'default init from KUKA shape')
            visualize(self.pc_b, pts_a, 'default init from KUKA shape, best translation guess', tsf=np.array([
                [1, 0, 3],
                [0, 1, -0.5],
                [0, 0, 1]
            ]))

    class FuseLaser:
        """
        1) Fuse the laser scan for robots `r` and `r_` for their relative transformation with multiple
        random initializations

        2) Pick the one s.t. robot `r`'s transformed point cloud matches robot `r_`'s base point-cloud representation
        """
        pass


if __name__ == '__main__':
    # from explore_package.irc_laser_data_eg import *
    #
    # src_pts = src_pts[:, :2]  # Expect 2-dim data points
    # tgt_pts = tgt_pts[:, :2]
    # # ic(src_pts.shape, tgt_pts.shape)
    #
    # t = PoseEstimator.Icp(src_pts, tgt_pts)()
    # ic(t)

    # Empirically have `robot_a` as HSR, `robot_b` as KUKA
    fp = PoseEstimator.FusePose(pc_b=get_rect_pointcloud(
                PoseEstimator.L_KUKA,
                PoseEstimator.W_KUKA,
    ))
    ic(fp.pc_b.shape)

    hsr_scans = get_scans('../data/HSR laser 2.json')
    ic(len(hsr_scans))
    s = hsr_scans[77]
    # del s['ranges']
    # del s['intensities']
    # ic(s)
    pts = laser_polar2planar(s['angle_max'], s['angle_min'])(np.array(s['ranges']))
    fp(pts_a=pts)

