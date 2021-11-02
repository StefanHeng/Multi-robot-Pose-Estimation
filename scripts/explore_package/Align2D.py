"""
Modified from [LIDAR Odometry with ICP](http://andrewjkramer.net/lidar-odometry-with-icp/)
"""
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

sns.set_style('dark')


class Align2D:
    def __init__(self, source_points, target_points, initial_T):
        """

        :param source_points:  numpy array containing points to align to the target set points should be homogeneous,
                                with one point per row
        :param target_points: numpy array containing points to which the source points are to be aligned,
                                points should be homogeneous with one point per row
        :param initial_T: initial estimate of the transform between target and source
        """
        self.source = source_points
        self.target = target_points
        self.init_T = initial_T
        self.target_tree = KDTree(target_points[:, :2])
        self.transform = self.align_icp(20, 1.0e-4)

    def align_icp(self, max_iter, min_delta_err):
        """
        uses the iterative closest point algorithm to find the transformation between the source and target point clouds
        that minimizes the sum of squared errors between nearest neighbors in the two point clouds
        :param max_iter: int, max number of iterations
        :param min_delta_err: float, minimum change in alignment error
        """
        mean_sq_error = 1.0e6  # initialize error as large number
        delta_err = 1.0e6  # change in error (used in stopping condition)
        T = self.init_T
        num_iter = 0  # number of iterations
        tf_source = self.source

        while delta_err > min_delta_err and num_iter < max_iter:
            # find correspondences via nearest-neighbor search
            matched_trg_pts, matched_src_pts, indices = self.find_correspondences(tf_source)

            # find alignment between source and corresponding target points via SVD
            # note: svd step doesn't use homogeneous points
            new_T = self.align_svd(matched_src_pts, matched_trg_pts)
            ic(new_T)

            # update transformation between point sets
            np.testing.assert_equal(np.dot(T, new_T), T @ new_T)
            T = np.dot(T, new_T)

            # apply transformation to the source points
            np.testing.assert_equal(np.dot(self.source, T.T), self.source @ T.T)
            tf_source = np.dot(self.source, T.T)

            # find mean squared error between transformed source points and target points
            new_err = 0
            for i in range(len(indices)):
                if indices[i] != -1:
                    diff = tf_source[i, :2] - self.target[indices[i], :2]
                    new_err += np.dot(diff, diff.T)

            new_err /= float(len(matched_trg_pts))

            # update error and calculate delta error
            delta_err = abs(mean_sq_error - new_err)
            ic(new_err, delta_err)
            mean_sq_error = new_err

            num_iter += 1

        return T

    def find_correspondences(self, src_pts):
        """
        finds nearest neighbors in the target point for all points in the set of source points
        :param src_pts: array of source points for which we will find neighbors points are assumed to be homogeneous
        :return: array of nearest target points to the source points (not homogeneous)
        """
        # get distances to nearest neighbors and indices of nearest neighbors
        matched_src_pts = src_pts[:, :2]
        dist, indices = self.target_tree.query(matched_src_pts)

        # remove multiple associations from index list
        # only retain closest associations
        unique = False
        while not unique:
            unique = True
            for i in range(len(indices)):
                if indices[i] == -1:
                    continue
                for j in range(i + 1, len(indices)):
                    if indices[i] == indices[j]:
                        if dist[i] < dist[j]:
                            indices[j] = -1
                        else:
                            indices[i] = -1
                            break
        # build array of nearest neighbor target points
        # and remove unmatched source points
        point_list = []
        src_idx = 0
        for idx in indices:
            if idx != -1:
                point_list.append(self.target[idx, :])
                src_idx += 1
            else:
                matched_src_pts = np.delete(matched_src_pts, src_idx, axis=0)

        matched_pts = np.array(point_list)

        return matched_pts[:, :2], matched_src_pts, indices

    def align_svd(self, source, target):
        """
        uses singular value decomposition to find the transformation from the target to the source point cloud
        assumes source and target point clouds are ordered such that corresponding points are at the same indices
        in each array

        :param source: numpy array representing source pointcloud
        :param target: numpy array representing target pointcloud
        :return: T: transformation between the two point clouds
        """

        # first find the centroids of both point clouds
        src_centroid = self.get_centroid(source)
        trg_centroid = self.get_centroid(target)

        # get the point clouds in reference to their centroids
        source_centered = source - src_centroid
        target_centered = target - trg_centroid

        # get cross covariance matrix M
        M = np.dot(target_centered.T, source_centered)

        # get singular value decomposition of the cross covariance matrix
        U, W, V_t = np.linalg.svd(M)

        # get rotation between the two point clouds
        R = np.dot(U, V_t)

        # get the translation (simply the difference between the point cloud centroids)
        t = np.expand_dims(trg_centroid, 0).T - np.dot(R, np.expand_dims(src_centroid, 0).T)

        # assemble translation and rotation into a transformation matrix
        T = np.identity(3)
        T[:2, 2] = np.squeeze(t)
        T[:2, :2] = R
        return T

    @staticmethod
    def get_centroid(points):
        point_sum = np.sum(points, axis=0)
        return point_sum / float(len(points))


if __name__ == '__main__':
    import math

    from irc_laser_data_eg import *
    # from scripts.robo_pose_estimator import Icp

    ic(src_pts[:5], tgt_pts[:5])
    # ic(source_points[:, -1], target_points[:, -1])
    ic(src_pts.shape, tgt_pts.shape)
    a2d = Align2D(src_pts, tgt_pts, np.identity(3))
    tsf = a2d.transform
    ic(tsf)
    # tsf = Icp(src_pts, tgt_pts)(np.identity(3))
    # ic(tsf)

    def _plot_clouds(p_s, p_t, title=None, save=False):
        plt.figure(figsize=(16, 9), constrained_layout=True)
        plt.scatter(p_s[:, 0], p_s[:, 1], marker='.', s=9, c='c', label='source')
        plt.scatter(p_t[:, 0], p_t[:, 1], marker='.', s=9, c='m', label='target')
        t = 'Point Clouds'
        if title:
            t = f'{t}, {title}'
        plt.title(t)
        plt.legend()
        if save:
            plt.savefig(f'plot/{t}.png', dpi=300)
        plt.show()


    # source_ = np.dot(src_pts, T.T)
    source_ = src_pts @ tsf.T
    # ic(source_[:, -1])

    rot = tsf[:2, :2]
    tsl = tsf[:2, 2]
    angle = math.acos(rot[0][0])
    ic(math.degrees(angle), tsl)
    t_ = np.array(list(tsl) + [1])
    s = src_pts[:, :-1]
    ic(s.shape, tsl.shape)
    ic((s @ rot.T).shape, tsl.shape)
    ic(source_[:10])
    s_ = s @ rot.T + tsl.reshape(1, -1)
    ic(s_[:10])

    # Break down into rotation & translation
    np.testing.assert_equal(source_[:, :-1], s @ rot.T + tsl.reshape(1, -1))

    _plot_clouds(src_pts, tgt_pts, title='Initial', save=True)
    _plot_clouds(source_, tgt_pts, title='Final', save=True)


