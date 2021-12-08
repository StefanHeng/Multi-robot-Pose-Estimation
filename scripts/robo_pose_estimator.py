from copy import deepcopy
import math
import pickle

import numpy as np
from sklearn import linear_model
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
    arr = np.asarray(arr)
    return np.hstack([arr, np.ones([arr.shape[0], 1])])


class Loss:
    """
    Various metrics to compute the fitness of two point clouds
    """
    class NearestNeighbor:
        """
        Given a list of 2D query points, return the corresponding target points matched
        """
        def __init__(self, tgt):
            self.tgt = tgt
            self.tree_tgt = KDTree(tgt)
            self.n = self.tgt.shape[0]

        def __call__(self, src, with_dist=False):
            """
            Finds nearest neighbors in the target
            """
            dist, idxs_ = self.tree_tgt.query(src[:, :2])
            # Keep pairs with distinct target points by removing pairs with larger distance
            idxs_pair = np.vstack([np.arange(idxs_.size), idxs_]).T  # Each element of [src_idx, tgt_idx]
            arg_idxs = dist.argsort()
            idxs_pair_sort = idxs_pair[arg_idxs]
            # idxs_sort = idxs_[arg_idxs]
            dist_sort = dist[arg_idxs]


            # Find the unique targets with the lowest distances
            idxs_tgt_sort = idxs_pair_sort[:, 1]
            idxs_tgt_uniq = np.unique(idxs_tgt_sort, return_index=True)[1]  # Index of 1st occurrence
            idxs_pair_uniq = idxs_pair_sort[idxs_tgt_uniq]

            # Sort the pairs by distance
            dist_by_tgt = dist_sort[idxs_tgt_uniq]
            idxs_tgt_sort = dist_by_tgt.argsort()
            idxs_pair_uniq = idxs_pair_uniq[idxs_tgt_sort]

            # def arr_idx(a, v):
            #     """
            #     :return: 1st occurrence index of `v` in `a`, a numpy 1D array
            #     """
            #     return np.where(a == v)[0][0]
            #
            # def _get_idx(arr):
            #     def _get(i):
            #         """
            #         If arr[i] is not the first occurrence, it's set to -1
            #         """
            #         i_ = arr_idx(arr, arr[i])
            #         return arr[i] if i_ == i else -1
            #
            #     return _get
            #
            # idxs_1st_occ = np.vectorize(_get_idx(idxs_sort))(np.arange(idxs_sort.size))
            # idxs_pair_uniq = idxs_pair_sort[np.where(idxs_1st_occ != -1)]
            # np.testing.assert_array_equal(idxs_pair_uniq_tgt[idxs_tgt_sort], idxs_pair_uniq)

            ret = [
                src[idxs_pair_uniq[:, 0]][:, :2],
                self.tgt[idxs_pair_uniq[:, 1]][:, :2],
                idxs_pair_uniq
            ]
            if with_dist:
                ret += [dist[idxs_pair_uniq[:, 0]]]
            return ret

    def __init__(self, tgt):
        """
        :param tgt: List of 2d points, a base set of points to compute loss
        """
        self.nn = Loss.NearestNeighbor(tgt)
        self.tgt = self.nn.tgt  # Save memory

    def pose_error(
            self, src, tsf,
            labels=None, bias=False, n=2, plot=False, dist_thresh=inf, cls_frac=-inf, adjust_inf=True, is_mat=False
    ):
        """
        :param src: List of 2d points to match target points
        :param tsf: Proposed translation, given by 3-array of (translation_x, translation_y, rotation angle),
            or translation matrix,
            or list of them
        :param is_mat: If `tsf` is a 3-tuple of translation or translation matrix
        :param labels: 1D list, cluster assignments for source points,
            If specified, the target points are split by clusters where loss is computed for each cluster
        :param bias: If true, bias transformations with more points matched linearly
        :param n: Order of norm

        :param plot: If true, and only one `tsf` option is given, plot the matched points
        :param dist_thresh: Threshold for minimum distance between matched points, if larger, error is set to infinity
        :param cls_frac: Threshold for fraction of points matched relative to cluster size,
            if smaller, error is set to infinity
        :param adjust_inf: If true, loss of infinity are set to a value below global minimum

        :return: If `labels` unspecified, the error based on closest pair of matched points;
            If `labels` specified, 2-tuple of (label-to-index mapping, error for each cluster in the mapping order)
        """
        src = np.asarray(src)
        tsf = np.asarray(tsf)
        if not is_mat:
            ic(tsf.shape, tsf)
            tsf = np.apply_along_axis(tsl_n_angle2tsf, 1, tsf)
            ic(tsf.shape)
            exit(1)

        def adjust(arr):
            if arr.size > 1 and adjust_inf:
                from numpy import inf
                arrs_ = arr[arr != inf]
                ma, mi = arrs_.max(), arrs_.min()
                arr[arr == inf] = ma + (ma-mi) / 2**4

        def _pose_error(cluster):
            num = cluster.shape[0] * cls_frac
            # ic('then im here', tsf)

            def __pose_error(tsf_):
                # ic(tsf_)
                src_mch, tgt_mch, idxs, dist = self.nn(apply_tsf_2d(cluster, tsf_), with_dist=True)
                # if len(dist) > 20:
                #     ic(tsf_)
                #     exit(1)
                if dist[0] > dist_thresh or idxs.shape[0] < num:  # dist[0] is the smallest value
                    return inf
                err = self.pts_match_error(src_mch, tgt_mch, n=n)
                return err / math.sqrt(idxs.shape[0]) if bias else err
            if len(tsf.shape) == 3:
                return np.apply_along_axis(__pose_error, 1, tsf)
            else:
                err = __pose_error(tsf)
                # ic('here', tsf, err)
                return err
        if labels is None:
            if plot:
                src_ = apply_tsf_2d(src, tsf)
                src_mch_, tgt_mch_, _ = self.nn(src_)
                plot_2d([src_, self.tgt], label=['Source points, transformed', 'Target points'], show=False)
                for s, t in zip(src_mch_, tgt_mch_):
                    plot_line_seg(s, t)
                plt.show()
            errs = _pose_error(src)
            # ic('am i here?')
            adjust(errs)
            return errs
        else:
            def _get(idx, c):
                print(f'{now()}| Getting errors for cluster #{idx+1}... ')
                return _pose_error(c)
            label_idxs = {lb: idx for idx, lb in enumerate(np.unique(labels))}
            clusters = [src[np.where(labels == lb)] for lb in label_idxs]
            errs = np.stack([_get(idx, c) for idx, c in enumerate(clusters)])
            adjust(errs)
            return label_idxs, errs.T

    @staticmethod
    def pts_match_error(pts1, pts2, n=2):
        """
        :return: Pair wise euclidian distance/l2-norm between two lists of matched 2d points,
            normalized by number points
        """
        n = np.linalg.norm(pts1[:, :2] - pts2[:, :2], ord=n, axis=1)
        return n.mean()


class Search:
    @staticmethod
    def grid_search(pts, grid=None, reverse=False, save=False, err_kwargs=None):
        """
        :param pts: 2-tuple of list of 2D points to match, (source points, target points)
        :param grid: Dict containing a `precision` and a `range` key
            `precision` contains a dict containing precision of translation in meters and angle in radians to search
            `range` contains a dict containing range of translation in meters and angle in radians to search
                angle range defaults to 2-tuple within range [-1, 1], the angle to search through
                    For shapes of rotational symmetry, don't need to explore entire 2pi degree space,
                        e.g. For rectangles, any range covering 1pi suffice
        :param reverse: If true, `pts` are treated as source and targets inversely,
            i.e. in the order (target points, source points)
        :param save: If true, errors and configuration settings are saved to `pickle`
        :param err_kwargs: Arguments passed to `Loss.pose_error`
        :return: 4-tuple of (X translation options, Y translation options, Angle options,
            Corresponding error for each setup combination)

        Systematically search for the confidence of robot A's pose, for each x, y, theta setting
        """
        src, tgt = reversed(pts) if reverse else pts
        print(f'{now()}| Grid-searching for pose estimates on ... ')
        print(f'    target of [{tgt.shape[0]}] points, and ')
        print(f'    source of [{src.shape[0]}] points ... ')

        def default_ran():
            x_max, y_max = tgt.max(axis=0)
            x_min, y_min = tgt.min(axis=0)
            edge = pts2max_dist(src)
            return dict(
                x=(math.floor(x_min-edge), math.ceil(x_max+edge)),
                y=(math.floor(y_min-edge), math.ceil(y_max+edge)),
                angle=(-1, 1)
            )

        def default_prec():
            # return dict(tsl=3, angle=1)
            # return dict(tsl=5e-1, angle=1/9)
            return dict(tsl=1e0, angle=1/2)

        prec, ran = 'precision', 'range'
        dft_prec, dft_ran = default_prec(), default_ran()
        if grid is not None:
            grid[prec] = (prec in grid and (dft_prec | grid[prec])) or dft_prec
            grid[ran] = (ran in grid and (dft_ran | grid[ran])) or dft_ran
        else:
            grid = {prec: dft_prec, ran: dft_ran}
        x_ran, y_ran, angle_ran = grid[ran]['x'], grid[ran]['y'], grid[ran]['angle']
        # ic(pts.max(axis=0), pts.min(axis=0))
        prec_tsl = grid[prec]['tsl']
        prec_ang = grid[prec]['angle']
        opns_x = np.linspace(*x_ran, num=int((x_ran[1]-x_ran[0]) / prec_tsl+1))
        opns_y = np.linspace(*y_ran, num=int((y_ran[1]-y_ran[0]) / prec_tsl+1))
        opns_ang = np.linspace(*angle_ran, num=int((angle_ran[1]-angle_ran[0]) / prec_ang+1))[1:]
        opns = cartesian([opns_x, opns_y, opns_ang])
        print(f'    x range {x_ran}, at precision {prec_tsl} => {opns_x.size} candidates and ')
        print(f'    y range {y_ran}, at precision {prec_tsl} => {opns_y.size} candidates and ')
        print(f'    angle range {angle_ran}, at precision {round(prec_ang, 3)} => {opns_ang.size} candidates, ')
        print(f'    for a combined [{opns.shape[0]}] candidate setups... ')

        if err_kwargs is None:
            err_kwargs = dict()
        errs = Loss(tgt).pose_error(src, opns, **err_kwargs)
        has_label = 'labels' in err_kwargs
        if has_label:
            label_idxs, errs = errs
        print(f'{now()}| ... completed')

        if save:
            d = dict(
                options_x=opns_x,
                options_y=opns_y,
                options_angle=opns_ang,
                errors=errs
            )
            if has_label:
                d['label_indices'] = label_idxs
            fnm = f'gird-search, {[x_ran, y_ran, prec_tsl]}, {[angle_ran, prec_ang]}, {now()}.pickle'
            with open(fnm, 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'{now()}| Grid search result written to pickle ')
        return opns_x, opns_y, opns_ang, ((label_idxs, errs) if has_label else errs)


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
        src = np.asarray(src)
        tgt = np.asarray(tgt)
        self.nn = Loss.NearestNeighbor(tgt)
        # Should stay unchanged across program duration
        self.src = extend_1s(src)
        self.tgt = self.nn.tgt

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
            src_match, tgt_match, idxs = self.nn(src)

            def _err():
                src_ = src[idxs[:, 0]]
                # ic(np.square(src_[:, :2] - tgt_match))
                err = np.sum(np.square(src_[:, :2] - tgt_match)) / idxs.shape[0]
                # err_ = np.linalg.norm(src_[:, :2] - tgt_match, axis=0, ord=2).mean()
                # ic(err, err_)
                # assert err == err_
                return err

            err_ = _err()
            d_err = abs(err - err_)
            err = err_

            if verbose:
                states.append((src_match, tgt_match, tsf, err))

            tsf = tsf @ self.svd(src_match, tgt_match)
            src = self.src @ tsf.T
            n += 1
        return (tsf, states) if verbose else tsf

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
        rot_mat_ = U @ V_t
        tsl = c_tgt - rot_mat_ @ c_src
        tsf = np.identity(3)
        tsf[:2, 2] = tsl
        tsf[:2, :2] = rot_mat_
        return tsf


def visualize(a, b, init_tsf=np.identity(3), mode='static', **kwargs):
    ic('Initial guess', init_tsf)
    tsf, states = Icp(a, b)(tsf=init_tsf, max_iter=100, min_d_err=1e-6, verbose=True)
    plot_icp_result(extend_1s(a), b, tsf, states=states, init_tsf=init_tsf, mode=mode, **kwargs)
    return tsf, states


class TsfInitializer:
    """
    Given a set of 2D laser scan points, generate possible candidates of initial transformation
    """
    def __init__(self):
        pass

    @staticmethod
    def rect_tsf_cands(pts, labels=None, kwargs_cands=None):
        """
        :param pts: A list of 2d points
        :param labels: If given, `pts` are first separated, and RANSAC is performed on each cluster
        :param kwargs_cands: See `TsfInitializer.ln2rect_tsf_cands`
        :return: A list of rectangle transformation candidates based on RANSAC linear regression
            If `labels` given, a dict of list
        """
        if kwargs_cands is None:
            kwargs_cands = dict()

        def _rect_tsf_cands(cluster):
            cands = [TsfInitializer.ransac_linear(cluster, reverse=r) for r in [False, True]]
            # ic(cands)
            # ic(sum([TsfInitializer.ln2rect_tsf_cands(**kwargs_cands) for cand in cands], []))
            # exit(1)
            return sum([TsfInitializer.ln2rect_tsf_cands(cand, **kwargs_cands) for cand in cands], [])
        if labels is None:
            return _rect_tsf_cands(pts)
        else:
            d_cls = {lb: pts[np.where(labels == lb)] for lb in np.unique(labels)}
            return {lb: _rect_tsf_cands(c) for lb, c in d_cls.items()}

    @staticmethod
    def ransac_linear(pts, labels=None, plot=False, reverse=False, return_ln=False):
        """
        Fit the set of points linearly

        :param pts: A list of 2d points
        :param labels: If given, `pts` are first separated, and RANSAC is performed on each cluster
        :param reverse: If reversed, the independent and dependent axis in `pts` are flipped
        :param plot: If true, the result is visualized
            Allowed only for single cluster
        :param return_ln: If true, the fitted RANSAC line is returned as array of start & end points
        :return: 2-tuple of (coefficient, centroid in the range of inlier points), or list of them
        """
        def _ransac_linear(cluster):
            cluster = np.asarray(cluster)
            ransac = linear_model.RANSACRegressor()
            if reverse:
                cluster = cluster[:, ::-1]

            x, y = cluster[:, 0], cluster[:, 1]
            ransac.fit(x.reshape(-1, 1), y.reshape(-1, 1))

            inlier_mask = ransac.inlier_mask_
            inliers = cluster[inlier_mask]
            x_in, y_in = inliers[:, 0], inliers[:, 1]

            x_ = np.linspace(x_in.min(), x_in.max(), num=2)[:, np.newaxis]
            y_ = ransac.predict(x_)
            center = [x_.mean(), y_.mean()]
            end_pts = [x_, y_]
            if reverse:
                end_pts.reverse()
                center.reverse()

            if labels is None and plot:
                outlier_mask = np.logical_not(inlier_mask)
                outliers = cluster[outlier_mask]
                ins = np.array([x_in, y_in]).T
                outs = np.array([outliers[:, 0], outliers[:, 1]]).T
                center_in = inliers.mean(axis=0)
                if reverse:
                    ins[:] = ins[:, ::-1]
                    outs[:] = outs[:, ::-1]
                    center_in[:] = center_in[::-1]
                d = 9
                ratio = (y.max()-y.min()) / (x.max()-x.min())
                ratio = clipper(2**-2, 2**2)(ratio)
                plt.figure(figsize=(d, d/ratio if reverse else d*ratio))
                cs = iter(sns.color_palette(palette='husl', n_colors=7))
                plot_points(ins, c=next(cs), label='Inliers')
                plot_points(outs, c=next(cs), label='Outliers')
                plot_points([center_in], ms=8, c=next(cs), label='Centroid or inliers')
                plot_points([center], ms=8, c=next(cs), label='Centroid of regression')
                plt.plot(*end_pts, lw=0.5, c=next(cs), label='Regression')

                plt.gca().set_aspect('equal')
                plt.legend()
                plt.title('RANSAC illustration on HSR scan cluster')
                plt.show()
            coef = ransac.estimator_.coef_
            assert coef.size == 1
            coef = coef[0][0]
            ret = [(1/coef if reverse else coef), center]
            if return_ln:
                ret.append(end_pts)
            return tuple(ret)
        if labels is None:
            return _ransac_linear(pts)
        else:
            d_cls = {lb: pts[np.where(labels == lb)] for lb in np.unique(labels)}
            return {lb: _ransac_linear(c) for lb, c in d_cls.items()}

    @staticmethod
    def ln2rect_tsf_cands(line_seg, rect_dim=config('dimensions.KUKA'), return_mat=True, plot=False, no_flip=False):
        """
        Given a line segment, propose possible transformation candidates such that
            the rectangle, transformed from centroid of origin, matches the line segment

        :param line_seg: 2-tuple of (coefficient, centroid)
        :param rect_dim: 2-tuple of (length, width) of dict with keys `length` and `width`
        :return: List of transformations
        :param return_mat: If false, returns 3-tuple of (translation_x, translation_y, rotation angle);
            Otherwise, returns transformation matrices
        :param no_flip: If true, only 2 of the 4 cases which are distinct 90 degree rotations are returned
            Intended for illustration purposes
        :param plot: If true, the result is visualized
        """
        coef, center = line_seg
        theta = math.atan(coef)
        theta_comp = math.pi/2-theta
        x, y = center
        ln_, wd_ = rect_dim['length'], rect_dim['width'] if isinstance(rect_dim, dict) else rect_dim

        def _get(ln, wd):
            flipped = abs(ln) < abs(wd)
            l, w = ln/2, wd/2

            bl = (x - w*math.cos(theta), y - w*math.sin(theta))  # Bottom left corner
            x_tl, y_tl = (x + w*math.cos(theta), y + w*math.sin(theta))  # x and y for top left corner
            tr = (x_tl + ln*math.cos(theta_comp), y_tl - ln*math.sin(theta_comp))
            rect_cent = np.mean(np.array([bl, tr]), axis=0)
            return get_rect_pointcloud(rect_dim), tuple([*rect_cent, theta if flipped else -theta_comp])

        args = [
            (ln_, wd_),
            (wd_, ln_),
            (-ln_, -wd_),
            (-wd_, -ln_)
        ]
        if no_flip:
            args = args[:2]
        cands = [_get(*a) for a in args]

        if plot:
            hypot = np.linalg.norm(np.array([ln_, wd_]), ord=2)  # hypotenuse
            diff_x_seg, diff_y_seg = hypot * math.cos(theta), hypot * math.sin(theta)
            pts_seg = np.array([
                [x - diff_x_seg, y - diff_y_seg],
                [x, y],
                [x + diff_x_seg, y + diff_y_seg]
            ])

            d = 12
            plt.figure(figsize=(d, d))
            cs = iter(sns.color_palette(palette='husl', n_colors=7))
            plot_points(pts_seg, ms=2**3, c=next(cs), label='Line segment')
            for idx, (pts, tsf) in enumerate(cands):
                tsf = tsl_n_angle2tsf(tsf)
                plot_points(apply_tsf_2d(pts, tsf), c=next(cs), label=f'Rectangle candidate {idx+1}')
            plt.legend()
            plt.gca().set_aspect('equal', 'box')
            plt.title('Rectangle candidates that fits the line segment')
            plt.show()
        cands = [tsf for (_, tsf) in cands]
        return [tsl_n_angle2tsf(tsf) for tsf in cands] if return_mat else cands


class PoseEstimator:
    """
    Various laser-based pose estimation algorithms between KUKA iiwa and HSR robot

    The two robots should potentially detect each other
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

        def grid_search(self, precision=None, reverse_pts=False, plot=False, plot_kwargs=None, err_kwargs=None):
            def get_pts(a, b):
                return (b, a) if reverse_pts else (a, b)

            # ret1 = Search.grid_search(get_pts(self.pcr_a, self.pts_b), precision=precision, labels=labels)
            ret2 = Search.grid_search(get_pts(self.pcr_b, self.pts_a), **err_kwargs)
            if plot:
                plot_grid_search(self.pcr_b, self.pts_a, *ret2, **plot_kwargs)
            return ret2

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

    pcr_kuka = get_kuka_pointcloud()
    pts_hsr = eg_hsr_scan()
    tsf_ideal = tsl_n_angle2tsf(config('heuristics.pose_guess.actual_pose'))

    def check_icp_hsr():
        # Empirically have `robot_a` as HSR, `robot_b` as KUKA
        fp = PoseEstimator.FusePose(pcr_b=pcr_kuka)
        ic(fp.pcr_b.shape)

        title = 'default init from KUKA shape, good translation guess'
        init_tsf = np.array([
            [1, 0, 3],
            [0, 1, -0.5],
            [0, 0, 1]
        ])
        visualize(
            pcr_kuka, pts_hsr, init_tsf=init_tsf,
            title=title, save=False,
            xlim=[-2, 6], ylim=[-2, 2],
            mode='static'
        )
    # check_icp_hsr()

    cls = Cluster.cluster

    def clustering_sanity_check():
        def sp():
            lbs = cls(pts_hsr, approach='spectral', n_clusters=8)
            plot_cluster(pts_hsr, lbs, title='Spectral on HSR', save=True)

        def hi(d):
            lbs = cls(pts_hsr, approach='hierarchical', distance_threshold=d)
            plot_cluster(pts_hsr, lbs, title=f'Hierarchical on HSR, avg threshold={d}', save=True)

        def ga():
            lbs = cls(pts_hsr, approach='gaussian', n_components=6)
            plot_cluster(pts_hsr, lbs, title='Gaussian Mixture on HSR', save=True)

        def db():
            lbs = cls(pts_hsr, approach='dbscan', eps=0.5, min_samples=16)
            plot_cluster(pts_hsr, lbs, title='DBSCAN on HSR, eps=0.5', save=True)

        sp()
        hi(1)
        hi(2)
        ga()
        db()
    # clustering_sanity_check()

    def icp_after_cluster():
        # A good clustering result by empirical inspection
        lbs = Cluster.cluster(pts_hsr, approach='hierarchical', distance_threshold=1)
        d_clusters = {lb: pts_hsr[np.where(lbs == lb)] for lb in np.unique(lbs)}

        pts_cls = d_clusters[11]  # The cluster indicating real location of KUKA

        # visualize(
        #     ptc_kuka, pts_cls,
        #     title='HSR locates KUKA, from the real cluster',
        #     init_tsf=tsl_n_angle2tsf(tsl=pts_cls.mean(axis=0)),
        #     xlim=[-2, 6], ylim=[-2, 3], mode='control', save=False
        # )
        visualize(
            pcr_kuka, pts_cls,
            title='HSR locates KUKA, from the real cluster, good translation estimate',
            init_tsf=tsl_n_angle2tsf(tsl=[2.5, -0.5]),
            xlim=[-1, 5], ylim=[-2, 2], mode='control',
            save=False
        )
    # icp_after_cluster()

    def grid_search():
        ret = Search.grid_search(
            (pcr_kuka, pts_hsr),
            # precision=dict(tsl=0.25, angle=1/20),
            # angle_range=(0, 1)
        )
        plot_grid_search(
            pcr_kuka, pts_hsr, *ret,
            inverse_loss=True,
            # save=True,
            tsf_ideal=tsf_ideal,
            zlabel='Normalized L2 norm from matched points',
            # interp=False,
            interp_kwargs=dict(
                # method='linear',
                # factor=2**2
            )
        )
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
        fp = PoseEstimator.FusePose(pts_a=pts_hsr, pcr_b=pcr_kuka)
        ret = fp.grid_search()
        for cmap in cmaps:
            ic(cmap)
            for cm in [cmap, f'{cmap}_r']:
                plot_grid_search(
                    fp.pcr_b, fp.pts_a, *ret,
                    inverse_loss=True, save=True, title=cm,
                    plot3d_kwargs=dict(cmap=cm)
                )
    # pick_cmap()

    dim_kuka = config('dimensions.KUKA')

    d_cls_res = config('heuristics.cluster_results.good')
    lbs = d_cls_res['labels']
    d_clusters = d_cls_res['clusters']
    d_clusters = {int(k): np.array(v) for k, v in d_clusters.items()}

    def grid_search_clustered():
        # ic(pcr_kuka.shape, pts_hsr.shape)
        ret = Search.grid_search(
            (pcr_kuka, pts_hsr),
            reverse=True,
            save=True,
            # grid=dict(precision=dict(tsl=0.25, angle=1/20), range=dict(x=(-6, 6), y=(-6, 6), angle=(0, 1))),
            grid=dict(precision=dict(tsl=0.25, angle=1/20), range=dict(angle=(0, 1))),
            err_kwargs=dict(labels=lbs, bias=False, n=2, dist_thresh=2, cls_frac=0.25),
        )
        plot_grid_search(
            pts_hsr, pcr_kuka, *ret,
            inverse_loss=True,
            labels=lbs,
            inverse_pts=True,
            interp=False,
            save=True,
            tsf_ideal=tsf_ideal,
            zlabel='Normalized L2 norm from matched points in best cluster',
            # interp_kwargs=dict(method='linear')
        )
    # grid_search_clustered()

    def profile():
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()
        grid_search_clustered()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
    # profile()

    def explore_visualize_reversed_icp():
        # A good cluster
        lbs = Cluster.cluster(pts_hsr, approach='hierarchical', distance_threshold=1)
        d_clusters = {lb: pts_hsr[np.where(lbs == lb)] for lb in np.unique(lbs)}
        pts_cls = d_clusters[11]

        x, y = config('heuristics.pose_guess.good_no_rotation')
        tsf, states = visualize(
            pts_cls, pcr_kuka,
            title='HSR locates KUKA, from the real cluster, good translation estimate, reversed',
            init_tsf=tsl_n_angle2tsf([-x, -y]),
            # xlim=[-2, 6], ylim=[-2, 3], mode='static',
            # save=True
        )
        tsf_rev = np.linalg.inv(tsf)  # Looks like
        plot_icp_result(pcr_kuka, pts_cls, tsf_rev, init_tsf=tsl_n_angle2tsf([x, y]))
    # explore_visualize_reversed_icp()

    def check_grid_search_cluster():
        fnm = 'gird-search, [(-5, 5), (-5, 5), 0.25], [(0, 1), 0.05], 2021-12-06 22:54:01.pickle'
        with open(fnm, 'rb') as handle:
            d = pickle.load(handle)
            opns_x = d['options_x']
            opns_y = d['options_y']
            opns_ang = d['options_angle']
            errs = d['errors']
            label_idxs = d['label_indices']
            ic(label_idxs)
            opns = cartesian([opns_x, opns_y, opns_ang])
            ic(opns.shape, errs.shape)
            idxs_sort = np.argsort(errs.min(axis=-1))
            ic(idxs_sort.shape, idxs_sort[:20])
            opns = opns[idxs_sort]
            errs = errs[idxs_sort]
            n = 5
            for opn, error in zip(opns[:n], errs[:n]):
                idx = np.argmin(error)
                ic(opn, label_idxs[idx], error[idx])
            ic()
            for opn, error in zip(opns[-n:], errs[-n:]):
                idx = np.argmin(error)
                ic(opn, label_idxs[idx], error[idx])
    # check_grid_search_cluster()

    def check_pose_error_plot():
        # opn = [-2.75, -0.75, 0.35]
        # opn = [-13., 0., -0.5]
        opn = [-0.75, 2., 0.05]
        pts_cls = d_clusters[12]
        Loss(pcr_kuka).pose_error(pts_cls, opn, plot=True)
    # check_pose_error_plot()

    def check_ransac():
        pts_cls = d_clusters[13]
        ti = TsfInitializer(pts_cls)
        coef, center = ti.ransac_linear(plot=True, reverse=True)
        ic(coef, center, type(coef), coef.shape)
        ic(coef, math.degrees(math.atan(coef)), center)
    # check_ransac()

    def check_init_proposal():
        pts_cls = d_clusters[11]
        ti = TsfInitializer(pts_cls)
        coef, center = ti.ransac_linear()
        ic(ti.ln2rect_tsf_cands((coef, center), dim_kuka, plot=True, return_mat=False))
    # check_init_proposal()

    def visualize_proposals():
        ti = TsfInitializer(pts_hsr)
        n_cls = len(d_clusters)
        cp = sns.color_palette(palette='husl', n_colors=n_cls)
        cs = iter(cp)
        plt.figure(figsize=(9, 9))
        plot_cluster(
            pts_hsr, lbs,
            cls_kwargs=[dict(label='Cluster', c=c) for c in cp],
            new_fig=False, show_eclipse=True,
            line_kwargs=dict(alpha=0.4, c=next(cs))
        )
        for r in [False, True]:
            cp = sns.color_palette(palette='husl', n_colors=n_cls)
            cs = iter(cp)
            for idx, (lb, (coef, center, end_pts)) in enumerate(
                    ti.ransac_linear(labels=lbs, reverse=r, return_ln=True).items()
            ):
                c = next(cs)
                plt.plot(*end_pts, lw=1, c=c, label='Regression' + (', reversed' if r else ''))
                for tsf in ti.ln2rect_tsf_cands((coef, center), dim_kuka, no_flip=True):
                    plot_points(
                        apply_tsf_2d(pcr_kuka, tsf),
                        lw=0.2, ms=0.2, alpha=0.3, c=c, label='Rectangle candidates'
                    )
        plot_points(
            apply_tsf_2d(pcr_kuka, tsf_ideal),
            lw=0.5, ms=0.25, c='black', label='Actual pose'
        )
        plt.gca().set_aspect('equal')
        handles, labels_ = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title('Transformation proposals on HSR scan cluster, without flipping')
        plt.show()
    # visualize_proposals()

    def icp_on_good_proposal():
        pts_cls = d_clusters[11]

        ti = TsfInitializer()
        cands = ti.rect_tsf_cands(pts_cls, kwargs_cands=dict(return_mat=False))
        cand = cands[0]  # This one is close to the hand-selected good initial transformation
        ic(cand)
        tsf1 = np.linalg.inv(tsl_n_angle2tsf(cand))
        tsf2 = tsl_n_angle2tsf([-e for e in cand])

        visualize(
            pts_cls, pcr_kuka,
            title='HSR locates KUKA, from the real cluster, translation proposed from RANSAC',
            init_tsf=tsf1,
            # mode='control'
        )
    # icp_on_good_proposal()

    def icp_on_proposals():
        ti = TsfInitializer()
        d_tsfs = ti.rect_tsf_cands(pts_hsr, labels=lbs, kwargs_cands=dict(rect_dim=dim_kuka, return_mat=False))
        err_by_tsl = []
        for label, tsfs in d_tsfs.items():
            # ic(len(tsfs), tsfs[:4])
            pts_cls = d_clusters[label]
            for tsf in tsfs:
                tsf_ = np.linalg.inv(tsl_n_angle2tsf(tsf))
                src, tgt = pts_cls, pcr_kuka
                tsf_ = Icp(src, tgt)(tsf_)
                error = Loss(tgt).pose_error(src, tsf_, is_mat=True)
                # ic(tsf[:2], error)
                err_by_tsl.append((*tsf[:2], error))
        err_by_tsl = np.array(err_by_tsl)
        err_thresh = 0.5  # Remove noise, exploding error
        err_by_tsl = err_by_tsl[err_by_tsl[:, 2] < err_thresh]
        ic(err_by_tsl.shape, err_by_tsl[:5])

        x, y, z = err_by_tsl[:, 0], err_by_tsl[:, 1], err_by_tsl[:, 2]

        # plt.figure(figsize=(9, 9))
        ic(pts2bins(np.stack([x, y]).T, prec=0.5))
        exit(1)
        # plot_points(np.stack([x, y]).T, lw=0, ms=8)
        # plt.show()
        # ic(z)
        # z = z.max() - z
        # z = scipy.special.softmax(z)
        # ic(z)
        # ic(x, y, z)
        prec = 0.25
        mi_x, ma_x = (math.floor(x.min() / prec)-1)*prec, (math.ceil(x.max()/prec)+1)*prec
        mi_y, ma_y = (math.floor(y.min() / prec)-1)*prec, (math.ceil(y.max()/prec)+1)*prec
        ic(mi_x, ma_x, mi_y, ma_y)
        x_grid = np.linspace(mi_x, ma_x, num=int((ma_x-mi_x)/prec + 1))
        y_grid = np.linspace(mi_y, ma_y, num=int((ma_y-mi_y)/prec + 1))
        ic(x_grid, y_grid)

        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        # triang = tri.Triangulation(x, y)
        # interpolator = tri.LinearTriInterpolator(triang, z)
        # Xi, Yi = np.meshgrid(xi, yi)
        # zi = interpolator(Xi, Yi)

        # Note that scipy.interpolate provides means to interpolate data on a grid
        # as well. The following would be an alternative to the four lines above:
        # from scipy.interpolate import griddata
        Z_ = scipy.interpolate.griddata((x, y), z, (x_grid[None, :], y_grid[:, None]), method='nearest')
        ic(Z_.shape)
        # for row in z_i:
        #     ic(row)

        d = 9

        ord_3d, ord_2d = 1, 20
        fig, ax = plt.subplots(figsize=(d, d), subplot_kw=dict(projection='3d'))
        X, Y = np.meshgrid(x_grid, y_grid)
        kwargs_surf = dict(
            zorder=ord_3d, antialiased=True,
            alpha=0.9, cmap='Spectral_r', edgecolor='black', lw=0.3
        )
        Z_ = -Z_

        bot, top = get_offset(Z_, frac=8)

        surf = ax.plot_surface(X, Y, -Z_, **kwargs_surf)
        # ax.contourf(X, Y, -Z_, levels=14, linewidths=0.5, colors='k')
        kwargs_cont = dict(
            zorder=ord_3d, antialiased=True,
            linewidths=1, levels=np.linspace(Z_.min(), Z_.max(), 2 ** 4), offset=bot, zdir='z',
            cmap='Spectral_r'
        )
        ct = ax.contour(X, Y, Z_, **kwargs_cont)
        # ax.plot(x, y, z)
        # ic(np.stack([x, y]).shape)
        plot_points(np.stack([x, y]).T, zs=-z, ms=4, lw=0)

        lb_tgt = 'Laser scan, target'

        cp = sns.color_palette(palette='husl', n_colors=7)
        # cp = list(reversed(cp)) if inverse_loss else cp
        cs = iter(cp)
        pcr = pcr_kuka
        plot_points([[0, 0]], zs=top, zorder=ord_2d, ms=10, alpha=0.5)
        c = next(cs)
        plot_points(pcr, zorder=ord_2d, zs=top, c=c, alpha=0.5, label='Point cloud representation, source')
        labels = lbs
        pts = pts_hsr
        if labels is not None:
            plot_cluster(pts, labels, new_fig=False, show_eclipse=False, line_kwargs=dict(
                zorder=ord_2d, zs=top, label=lb_tgt
            ))
        else:
            plot_points(pts, zorder=ord_2d, zs=top, c=next(cs), label=lb_tgt)
        if tsf_ideal is not None:  # Illustrate the ideal translation+-
            plot_points(
                apply_tsf_2d(pcr, tsf_ideal),
                zorder=ord_2d, zs=top, c=c, alpha=0.7,
                label='Point cloud representation at actual pose'
            )
        # fig.colorbar(ct, shrink=0.5, aspect=2 ** 5, pad=2 ** -4)
        plt.xlabel('Translation in X (m)')
        plt.ylabel('Translation in y (m)')
        ax.set_zlabel('Mean Squared Error')
        t = 'MSE by translation, interpolated'
        plt.title(t)
        plt.show()

        # tsfs = sum(d_tsfs.values(), [])
        # ic(len(tsfs))
    icp_on_proposals()
