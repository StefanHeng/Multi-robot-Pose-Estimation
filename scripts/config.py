from icecream import ic

from util import *
from data_path import *


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
    )
}


if __name__ == '__main__':
    import json

    fl_nm = 'config.json'
    ic(config)
    with open(f'{PATH_BASE}/{DIR_PROJ}/{fl_nm}', 'w') as f:
        json.dump(config, f, indent=4)

