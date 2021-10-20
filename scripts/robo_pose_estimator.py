import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from icecream import ic

from util import *


class PoseEstimator:
    """
    Various laser-based pose estimation algorithms between KUKA iiwa and HSR robot
    """

    class Icp:
        """
        Implementation of ICP algorithm in 2D
        Given source and target point clouds, returns the translation matrix T s.t. T(source) ~= target
        """
        @staticmethod
        def extend_1s(arr):
            """
            Return array with column of 1's appended
            :param arr: 2D array
            """
            return np.hstack([arr, np.ones([arr.shape[0], 1])])

        def __init__(self, src, tgt):
            """
            :param src:  Array of source points
            :param tgt: Array of target points
            """
            self.src = self.extend_1s(src)
            self.tgt = self.extend_1s(tgt)
            self.tree_tgt = KDTree(tgt)

        def __call__(self, transf=np.identity(3), max_iter=20, min_d_err=1e-4):
            """
            :param transf: Initial transformation estimate
            :param max_iter: Max iteration, stopping criteria
            :param min_d_err: Minimal change in error, stopping criteria
            """
            err = float('inf')
            d_err = float('inf')
            n = 0
            src = self.src

            while d_err > min_d_err and n < max_iter:
                src_match, tgt_match, idxs = self.nn_tgt(src)
                transf = transf @ self.svd(src_match, tgt_match)
                ic(transf)
                src = self.src @ transf.T

                def _err():
                    src_ = src[idxs[:, 0]]
                    return np.sum(np.square(src_[:, :2] - tgt_match[:, :2])) / idxs.shape[0]

                err_ = _err()
                # assert err > err_
                d_err = abs(err - err_)
                ic(err_, d_err)
                err = err_
                n += 1
            return transf

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

            # idxs_pair_uniq = idxs_pair_uniq[np.argsort(idxs_pair_uniq[:, 0])]
            # ic(idxs_pair_uniq)
            # exit(1)
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
            transla = c_tgt - rot_mat @ c_src
            transf = np.identity(3)
            transf[:2, 2] = transla
            transf[:2, :2] = rot_mat
            return transf

            # # get the point clouds in reference to their centroids
            # source_centered = src - c_src
            # target_centered = tgt - c_tgt
            #
            # # get cross covariance matrix M
            # M = np.dot(target_centered.T, source_centered)
            #
            # # get singular value decomposition of the cross covariance matrix
            # U, W, V_t = np.linalg.svd(M)
            #
            # # get rotation between the two point clouds
            # R = np.dot(U, V_t)
            #
            # # get the translation (simply the difference between the point cloud centroids)
            # t = np.expand_dims(c_src, 0).T - np.dot(R, np.expand_dims(c_tgt, 0).T)
            #
            # # assemble translation and rotation into a transformation matrix
            # T = np.identity(3)
            # T[:2, 2] = np.squeeze(t)
            # T[:2, :2] = R
            # return T


if __name__ == '__main__':
    from explore_package.irc_laser_data_eg import *

    src_pts = src_pts[:, :2]  # Expect 2-dim data points
    tgt_pts = tgt_pts[:, :2]
    # ic(src_pts.shape, tgt_pts.shape)

    t = PoseEstimator.Icp(src_pts, tgt_pts)()
    ic(t)
