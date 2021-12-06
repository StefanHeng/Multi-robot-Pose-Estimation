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
        cluster_eg=dict()
    )
}

# res =
# ic(type(res), type(res[0]), res[:5])
config['heuristics']['cluster_eg']['good'] = [
    int(lb) for lb in Cluster.cluster(eg_hsr_scan(), approach='hierarchical', distance_threshold=1)
]
# ic(type(config['heuristics']['cluster_result']))
# exit(1)


if __name__ == '__main__':
    import json

    fl_nm = 'config.json'
    ic(config)
    with open(f'{PATH_BASE}/{DIR_PROJ}/{fl_nm}', 'w') as f:
        json.dump(config, f, indent=4)

