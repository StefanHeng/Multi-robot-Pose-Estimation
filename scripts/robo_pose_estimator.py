import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from icecream import ic


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

        def __init__(self, src, tgt, t=np.identity(3)):
            """
            :param src:  Array of source points
            :param tgt: Array of target points
            :param t: Initial estimate of T
            """
            self.src = self.extend_1s(src)
            self.tgt = self.extend_1s(tgt)
            self.t = t
            self.tree_tgt = KDTree(tgt)

        def __call__(self, max_iter=20, min_d_err=1e-4):
            err = 10e6  # init with large values
            d_err = 10e6  # change in error (used in stopping condition)
            t = self.t
            n = 0
            src = self.src

            # def _err():

            while d_err > min_d_err and n < max_iter:
                tgt_match, src_match, idxs = self.nn_tgt(src)
                # ic(tgts.shape, srcs.shape, idxs.shape)
                t = t @ self.svd(src_match, tgt_match)
                src = self.src @ t.T

                # find mean squared error between transformed source points and target points
                new_err = 0
                for i in range(len(idxs)):
                    if idxs[i] != -1:
                        diff = src[i, :2] - self.tgt[idxs[i], :2]
                        new_err += np.dot(diff, diff.T)

                new_err /= float(len(tgt_match))

                # update error and calculate delta error
                d_err = abs(err - new_err)
                err = new_err

                n += 1

            return t

        def nn_tgt(self, src):
            """
            Finds nearest neighbors in the target

            :param src: Array of source points
            """
            matched_src_pts = src[:, :2]
            dist, idxs = self.tree_tgt.query(matched_src_pts)
            # ic(dist.shape, indices.shape)
            # ic(dist, indices)
            ic(idxs)
            # Keep pairs with distinct target points by removing pairs with larger distance
            arg_idxs = dist.argsort()  # Reorder matched pairs based on ascending distance
            dist_sort = dist[arg_idxs]
            idxs_sort = idxs[arg_idxs]
            src_sort = src[arg_idxs]
            tgt_sort = self.tgt[idxs_sort]
            # ic(idxs)
            # ic(src_s, tgt_s)
            # ic(dist_sort[:10])
            # d = np.linalg.norm(src_sort - tgt_sort, axis=-1)
            # ic(d[:10])
            # ic(idxs_sort, idxs_sort.size)
            # 1st occurrence must've been that of smallest distance
            idxs_unique = np.unique(idxs_sort, return_index=True)[1]
            # ic(np.unique(idxs_s, return_index=True), np.unique(idxs_s).size)
            src_f = src_sort[idxs_unique]
            tgt_f = tgt_sort[idxs_unique]
            ic(dist_sort[idxs_unique][:10])
            dists_ = np.linalg.norm(src_f - tgt_f, axis=-1)
            ic(dists_[:10])
            ic(idxs_unique.size)

            # remove multiple associations from index list
            # only retain closest associations
            unique = False
            while not unique:
                unique = True
                for i in range(len(idxs)):
                    if idxs[i] == -1:
                        continue
                    for j in range(i + 1, len(idxs)):
                        if idxs[i] == idxs[j]:
                            if dist[i] < dist[j]:
                                idxs[j] = -1
                            else:
                                idxs[i] = -1
                                break
            # build array of nearest neighbor target points
            # and remove unmatched source points
            point_list = []
            src_idx = 0
            for idx in idxs:
                if idx != -1:
                    point_list.append(self.tgt[idx, :])
                    src_idx += 1
                else:
                    matched_src_pts = np.delete(matched_src_pts, src_idx, axis=0)

            matched_pts = np.array(point_list)

            ic(sum(map(lambda x: x != -1, idxs)))
            tgt_ps = matched_pts[:, :2]
            dists = np.linalg.norm(tgt_ps - matched_src_pts, axis=-1)

            ic(dists[20:40])
            ic(dists_[20:40])
            np.testing.assert_equal(dists.sort(), dists_.sort())

            exit(1)
            return matched_pts[:, :2], matched_src_pts, idxs

        def svd(self, source, target):
            """
            Singular value decomposition to find the transformation from the target to the source point cloud
            assumes source and target point clouds are ordered such that corresponding points are at the same indices
            in each array

            :param source: numpy array representing source pointcloud
            :param target: numpy array representing target pointcloud
            :return: T: transformation between the two point clouds
            """

            # first find the centroids of both point clouds

            def _mean(pts):
                return np.sum(pts, axis=0) / pts.shape[0]

            c_src = _mean(source)
            c_tgt = _mean(target)

            # get the point clouds in reference to their centroids
            src_c = source - c_src
            tgt_c = target - c_tgt

            # get cross covariance matrix M
            M = np.dot(tgt_c.T, src_c)

            # get singular value decomposition of the cross covariance matrix
            U, W, V_t = np.linalg.svd(M)

            # get rotation between the two point clouds
            R = np.dot(U, V_t)

            # get the translation (simply the difference between the point cloud centroids)
            t = np.expand_dims(c_tgt, 0).T - np.dot(R, np.expand_dims(c_src, 0).T)

            # assemble translation and rotation into a transformation matrix
            T = np.identity(3)
            T[:2, 2] = np.squeeze(t)
            T[:2, :2] = R
            # ic(t.shape, t, T[:2, 2])
            # ic(R.shape, R, T[:2, :2])

            return T


if __name__ == '__main__':
    from explore_package.irc_laser_data_eg import *

    src_pts = src_pts[:, :2]  # Expect 2-dim data points
    tgt_pts = tgt_pts[:, :2]
    ic(src_pts.shape, tgt_pts.shape)

    t = PoseEstimator.Icp(src_pts, tgt_pts)()
    ic(t)
