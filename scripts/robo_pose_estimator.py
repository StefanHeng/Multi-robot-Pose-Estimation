from copy import deepcopy

import math
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from icecream import ic

from scripts.util import *


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

    Modified from [LIDAR Odometry with ICP](https://andrewjkramer.net/lidar-odometry-with-icp/)
    """

    def __init__(self, src, tgt):
        """
        :param src:  Array of source points
        :param tgt: Array of target points
        """
        self.src = extend_1s(src)
        self.tgt = extend_1s(tgt)
        self.tree_tgt = KDTree(tgt)

    def pose_error(self, tsf):
        """
        :param tsf: Proposed translation, given by 3-array of (translation_x, translation_y, theta),
            or list of proposed translations
        :return: The error based on closest pair of matched points
        """
        def _pose_error(tsf_):
            src_match, tgt_match, idxs = self.nn_tgt((self.src @ tsl_n_angle2tsf(tsf_).T)[:, :2])
            err = self.pts_match_error(src_match, tgt_match)
            # err *= idxs.shape[0] / self.src.shape[0] * 2**4
            # if err <= 0.5:
            #     ic(err, tsf_, idxs.shape)
            return err
        if len(tsf.shape) == 2:
            return np.apply_along_axis(_pose_error, 1, tsf)
        else:
            return _pose_error(tsf)

    @staticmethod
    def pts_match_error(pts1, pts2):
        """
        :return: Pair wise euclidian distance/l2-norm between two lists of matched 2d points,
            normalized by number points
        """
        pts1 = pts1[:, :2]
        pts2 = pts2[:, :2]
        n = np.linalg.norm(pts1 - pts2, ord=2, axis=1)
        # ic(n.shape)
        # return n.sum() / (n.size**2)
        return n.mean()

    def __call__(self, tsf=np.identity(3), max_iter=20, min_d_err=1e-4, verbose=False):
        """
        :param tsf: Initial transformation estimate
        :param max_iter: Max iteration, stopping criteria
        :param min_d_err: Minimal change in error, stopping criteria
        :param verbose: If true,
            return additionally a list of source-target matched points & transformation for each iteration
        """
        tsf = deepcopy(tsf)
        err = float('inf')
        d_err = float('inf')
        n = 0
        # src = self.src  # TODO: implementation wrong if non-identity initial guess?
        src = self.src @ tsf.T
        states = []

        while d_err > min_d_err and n < max_iter:
            src_match, tgt_match, idxs = self.nn_tgt(src)

            def _err():
                src_ = src[idxs[:, 0]]
                return np.sum(np.square(src_[:, :2] - tgt_match)) / idxs.shape[0]

            err_ = _err()
            d_err = abs(err - err_)
            err = err_

            if verbose:
                states.append((src_match, tgt_match, tsf))

            tsf = tsf @ self.svd(src_match, tgt_match)
            src = self.src @ tsf.T
            n += 1
        return (tsf, states) if verbose else tsf

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
                If arr[i] is not the first occurrence, it's set to -1
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

        def _centroid(pts_):
            return np.sum(pts_, axis=0) / pts_.shape[0]

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


def visualize(a, b, init_tsf=np.identity(3), mode='static', **kwargs):
    ic('Initial guess', init_tsf)
    tsf, states = Icp(a, b)(tsf=init_tsf, max_iter=100, min_d_err=1e-6, verbose=True)
    plot_icp_result(extend_1s(a), b, tsf, states=states, init_tsf=init_tsf, mode=mode, **kwargs)


class PoseEstimator:
    """
    Various laser-based pose estimation algorithms between KUKA iiwa and HSR robot
    """

    def __init__(self):
        pass

    class FusePose:
        """
        1) For each robot `r`, get its multiple pose estimates of the other robot `r_`
        based on `r_`'s point-cloud representation

        2) Pick the pair that makes sense as relative pose
        Note that those should be inverse of each other
        """
        def __init__(self, pcr_a=None, pcr_b=None, pts_a=None, pts_b=None):
            """
            :param pcr_a: Point-cloud representation of `robot_a`
            :param pcr_b: Point-cloud representation of `robot_b`
            :param pts_a: List of points of laser readings of `robot_a`
            :param pts_b: List of points of laser readings of `robot_b`
            """
            self.pcr_a = pcr_a
            self.pcr_b = pcr_b
            self.pts_a = pts_a
            self.pts_b = pts_b

        def __call__(self):
            # visualize(self.ptc_b, pts_a, tsf=np.array([
            #     [1, 0, 3],
            #     [0, 1, -0.5],
            #     [0, 0, 1]
            # ]))
            pass

        def grid_search(self, plot=False, plot_kwargs=None):
            # self._grid_search(self.pcr_a, self.pts_b)
            ret = self._grid_search(self.pcr_b, self.pts_a)
            if plot:
                plot_grid_search(self.pcr_b, self.pts_a, *ret, **plot_kwargs)
            return ret

        @staticmethod
        def _grid_search(pcr, pts, precision=None):
            """
            :param pcr: List of 2D points for robot A's shape
            :param pts: List of 2D points for laser scan from another robot B that potentially detects robot A
            :param precision: Dict containing precision of translation in meters and angle in radians to search
            :return: 4-tuple of (X translation options, Y translation options, Angle options,
                Corresponding error for each setup combination)

            Systematically search for the confidence of robot A's pose, for each x, y, theta setting
            """
            print(f'Grid-searching for pose estimates on...')
            print(f'   target of [{pts.shape[0]}] points, and')
            print(f'   source of [{pcr.shape[0]}] points...')
            if precision is None:
                # precision = dict(tsl=5e-1, angle=1 / 9)
                precision = dict(tsl=1e0, angle=1 / 2)
                # precision = dict(tsl=3e0, angle=1)
            # ic(pts.max(axis=0), pts.min(axis=0))
            x_max, y_max = pts.max(axis=0)
            x_min, y_min = pts.min(axis=0)
            edge = pts2max_dist(pcr)
            x_ran = [math.floor(x_min-edge), math.ceil(x_max+edge)]
            y_ran = [math.floor(y_min-edge), math.ceil(y_max+edge)]

            prec_tsl = precision['tsl']
            prec_ang = precision['angle']
            opns_x = np.linspace(*x_ran, num=int((x_ran[1]-x_ran[0]) / prec_tsl+1))
            opns_y = np.linspace(*y_ran, num=int((y_ran[1]-y_ran[0]) / prec_tsl+1))
            opns_ang = np.linspace(-1, 1, num=int(2 / prec_ang+1))[1:]
            opns = cartesian([opns_x, opns_y, opns_ang])
            print(f'    x range {x_ran}, at precision {prec_tsl} => {opns_x.size} candidates and ')
            print(f'    y range {y_ran}, at precision {prec_tsl} => {opns_y.size} candidates and ')
            print(f'    angle with precision {round(prec_ang, 3)} => {opns_ang.size} candidates, ')
            print(f'    for a combined [{opns.size}] candidates')

            errs = Icp(pcr, pts).pose_error(opns)
            print(f'... completed')
            return opns_x, opns_y, opns_ang, errs

    class FuseLaser:
        """
        1) Fuse the laser scan for robots `r` and `r_` for their relative transformation with multiple
        random initializations

        2) Pick the one s.t. robot `r`'s transformed point cloud matches robot `r_`'s base point-cloud representation
        """
        pass


if __name__ == '__main__':
    def icp_sanity_check():
        from explore_package.irc_laser_data_eg import src_pts, tgt_pts
        s_pts = src_pts[:, :2]  # Expect 2-dim data points
        t_pts = tgt_pts[:, :2]
        # ic(src_pts.shape, tgt_pts.shape)

        t = Icp(s_pts, t_pts)()
        ic(t)
        visualize(
            s_pts, t_pts,
            title='Sample data sanity check',
            # save=True
        )
    # icp_sanity_check()

    ptc_kuka = get_kuka_pointcloud()
    pts = eg_hsr_scan()

    def check_icp_hsr():
        # Empirically have `robot_a` as HSR, `robot_b` as KUKA
        fp = PoseEstimator.FusePose(pcr_b=ptc_kuka)
        ic(fp.pcr_b.shape)

        title = 'default init from KUKA shape, good translation guess'
        init_tsf = np.array([
            [1, 0, 3],
            [0, 1, -0.5],
            [0, 0, 1]
        ])
        visualize(ptc_kuka, pts, tsf=init_tsf, title=title, xlim=[-2, 6], ylim=[-2, 2], mode='static', save=True)
    # check_icp_hsr()

    cls = Cluster.cluster

    def clustering_sanity_check():
        def sp():
            lbs = cls(pts, approach='spectral', n_clusters=8)
            plot_cluster(pts, lbs, title='Spectral on HSR', save=True)

        def hi(d):
            lbs = cls(pts, approach='hierarchical', distance_threshold=d)
            plot_cluster(pts, lbs, title=f'Hierarchical on HSR, avg threshold={d}', save=True)

        def ga():
            lbs = cls(pts, approach='gaussian', n_components=6)
            plot_cluster(pts, lbs, title='Gaussian Mixture on HSR', save=True)

        def db():
            lbs = cls(pts, approach='dbscan', eps=0.5, min_samples=16)
            plot_cluster(pts, lbs, title='DBSCAN on HSR, eps=0.5', save=True)

        sp()
        hi(1)
        hi(2)
        ga()
        db()
    # clustering_sanity_check()

    def icp_after_cluster():
        cls = Cluster.cluster
        # A good clustering result by empirical inspection
        lbs = cls(pts, approach='hierarchical', distance_threshold=1)
        d_clusters = {lb: pts[np.where(lbs == lb)] for lb in np.unique(lbs)}

        cls = d_clusters[11]  # The cluster indicating real location of KUKA

        # visualize(
        #     ptc_kuka, cls,
        #     title='HSR locates KUKA, from the real cluster',
        #     init_tsf=tsl_n_angle2tsf(tsl=cls.mean(axis=0)),
        #     xlim=[-2, 6], ylim=[-2, 3], mode='control', save=False
        # )
        visualize(
            ptc_kuka, cls,
            title='HSR locates KUKA, from the real cluster, good translation estimate',
            init_tsf=tsl_n_angle2tsf(tsl=[2.5, -0.5]),
            xlim=[-1, 5], ylim=[-2, 2], mode='control',
            save=False
        )
    # icp_after_cluster()

    def grid_search():
        fp = PoseEstimator.FusePose(pts_a=pts, pcr_b=ptc_kuka)
        fp.grid_search(plot=True, plot_kwargs=dict(inverse=True, save=True))
    # grid_search()

    def pick_cmap():
        cmaps = [
            'mako',
            'CMRmap',
            'RdYlBu',
            'Spectral',
            'bone',
            'gnuplot',
            'gnuplot2',
            'icefire',
            'rainbow',
            'rocket',
            'terrain',
            'twilight',
            'twilight_shifted'
        ]
        # cmap='mako_r',
        # cmap='CMRmap',
        # cmap='RdYlBu',
        # cmap='Spectral',
        # cmap='Spectral_r',
        # cmap='bone',
        # cmap='gnuplot',
        # cmap='gnuplot2',
        # cmap='icefire',
        # cmap='rainbow',
        # cmap='rocket',
        # cmap='terrain_r',
        # cmap='twilight',
        # cmap='twilight_shifted',
        fp = PoseEstimator.FusePose(pts_a=pts, pcr_b=ptc_kuka)
        ret = fp.grid_search()
        for cmap in cmaps:
            ic(cmap)
            for cm in [cmap, f'{cmap}_r']:
                plot_grid_search(
                    fp.pcr_b, fp.pts_a, *ret,
                    inverse=True, save=True, title=cm,
                    plot3d_kwargs=dict(cmap=cm)
                )
    pick_cmap()


