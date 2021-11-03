import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from icecream import ic

from util import *


class Cluster:
    RDS = 77  # Random state

    @staticmethod
    def cluster(x, approach='spectral', **kwargs):
        d_kwargs = dict(
            spectral=dict(
                assign_labels='discretize',
                random_state=Cluster.RDS
            ),
            hierarchical=dict(
                n_clusters=None,
                linkage='average'
            ),
            gaussian=dict(
                random_state=Cluster.RDS
            ),
            dbscan=dict()
        )

        kwargs = d_kwargs[approach] | kwargs

        def spectral():
            assert 'n_clusters' in kwargs
            return SpectralClustering(**kwargs).fit(x).labels_

        def hierarchical():
            assert 'distance_threshold' in kwargs
            return AgglomerativeClustering(**kwargs).fit(x).labels_

        def gaussian():
            assert 'n_components' in kwargs
            return GaussianMixture(**kwargs).fit(x).predict(x)

        def dbscan():
            assert 'eps' in kwargs and 'min_samples' in kwargs
            return DBSCAN(**kwargs).fit(x).labels_

        d_f = dict(
            spectral=spectral,
            hierarchical=hierarchical,
            gaussian=gaussian,
            dbscan=dbscan
        )
        return d_f[approach]()

    def __init__(self, n_clusters=6):
        self.spectral = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=self.RDS)
        self.hierarchical = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=2)
        self.gaussian = GaussianMixture(n_components=n_clusters, random_state=self.RDS)
        self.dbscan = DBSCAN(eps=0.5, min_samples=16)

    def __call__(self, x, approach):
        """
        :param x: Array of 2D points to cluster
        """
        def spectral():
            return self.spectral.fit(x).labels_

        def hierarchical():
            return self.hierarchical.fit(x).labels_

        def gaussian():
            return self.gaussian.fit(x).predict(x)

        def dbscan():
            return self.dbscan.fit(x).labels_

        d_f = dict(
            spectral=spectral,
            hierarchical=hierarchical,
            gaussian=gaussian,
            dbscan=dbscan
        )
        return d_f[approach]()


def extend_1s(arr):  # The jupyter notebook `explore_fuse_pose` seems to require this???
    """
    Return array with column of 1's appended
    :param arr: 2D array
    """
    return np.hstack([arr, np.ones([arr.shape[0], 1])])


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

    def __call__(self, tsf=np.identity(3), max_iter=20, min_d_err=1e-4, lst_match=None):
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
        src = self.src

        while d_err > min_d_err and n < max_iter:
            src_match, tgt_match, idxs = self.nn_tgt(src)
            if lst_match:
                lst_match.append((src_match, tgt_match))
            # ic(tsf)
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

        def arr_idx(a, v):
            """
            :return: 1st occurrence index of `v` in `a`, a numpy 1D array
            """
            return np.where(a == v)[0][0]

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

    def __init__(self):
        pass

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
    from explore_package.irc_laser_data_eg import src_pts, tgt_pts

    def icp_sanity_check():
        s_pts = src_pts[:, :2]  # Expect 2-dim data points
        t_pts = tgt_pts[:, :2]
        # ic(src_pts.shape, tgt_pts.shape)

        t = Icp(s_pts, t_pts)()
        ic(t)

    def check_icp_hsr():
        # Empirically have `robot_a` as HSR, `robot_b` as KUKA
        fp = PoseEstimator.FusePose(pc_b=get_rect_pointcloud(
                    PoseEstimator.L_KUKA,
                    PoseEstimator.W_KUKA,
        ))
        ic(fp.pc_b.shape)

        hsr_scans = json_load('../data/HSR laser 2.json')
        ic(len(hsr_scans))
        s = hsr_scans[77]
        # del s['ranges']
        # del s['intensities']
        # ic(s)
        pts = laser_polar2planar(s['angle_max'], s['angle_min'])(np.array(s['ranges']))
        fp(pts_a=pts)

    check_icp_hsr()

    def clustering_sanity_check():
        hsr_scans = json_load('../data/HSR laser 2.json')
        s = hsr_scans[77]
        pts = laser_polar2planar(s['angle_max'], s['angle_min'])(np.array(s['ranges']))

        c = Cluster.cluster

        def sp():
            lbs = c(pts, approach='spectral', n_clusters=8)
            ic(lbs)
            plot_cluster(pts, lbs, title='Spectral on HSR', save=True)

        def hi():
            d = 2
            lbs = c(pts, approach='hierarchical', distance_threshold=d)
            plot_cluster(pts, lbs, title=f'Hierarchical on HSR, avg threshold={d}', save=True)

        def ga():
            lbs = c(pts, approach='gaussian', n_components=6)
            plot_cluster(pts, lbs, title='Gaussian on HSR, eps=0.5', save=True)

        def db():
            lbs = c(pts, approach='dbscan', eps=0.5, min_samples=16)
            plot_cluster(pts, lbs, title='DBSCAN on HSR, eps=0.5', save=True)

        # sp()
        hi()

    # clustering_sanity_check()



