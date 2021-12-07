from icecream import ic

from util import *
from data_path import *
from robo_pose_estimator import *


config = {
    DIR_DATA: dict(
        eg=dict(
            HSR=dict(
                fmt='*.json'
            )
        )
    ),
    'dimensions': dict(
        KUKA=dict(
            length=unit_converter(41),  # In meters
            width=unit_converter(25)
        )
    ),
    'heuristics': dict(
        pose_guess=dict(  # (translation_x, translation_y, rotation angle)
            good_no_rotation=[2.5, -0.5],
            actual_pose=[2.5, -0.75, -0.15]
        ),
        cluster_results=dict()
    )
}


pts_hsr = eg_hsr_scan()
lbs = Cluster.cluster(pts_hsr, approach='hierarchical', distance_threshold=1)
# ic(lbs[0])
# lst = lbs.tolist()
# ic(type(lst), type(lst[0]), lst[:5])
d_clusters = {int(lb): pts_hsr[np.where(lbs == lb)].tolist() for lb in np.unique(lbs)}
config['heuristics']['cluster_results']['good'] = dict(
    labels=lbs.tolist(),
    clusters=d_clusters
)
# k = list(d_clusters.keys())
# ic(type(k[0]))
# ic(k, d_clusters)
# ic(config)
# exit(1)


if __name__ == '__main__':
    import json

    fl_nm = 'config.json'
    ic(config)
    with open(f'{PATH_BASE}/{DIR_PROJ}/{fl_nm}', 'w') as f:
        json.dump(config, f, indent=4)

