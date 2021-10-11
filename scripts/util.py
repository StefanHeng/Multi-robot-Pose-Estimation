import json
from icecream import ic


class JsonWriter:
    def __init__(self, fnm):
        self.fnm = fnm
        self.data = []
        self.fnm_ext = f'{self.fnm}.json'
        open(self.fnm_ext, 'a').close()  # Create file in OS

    def __call__(self, data):
        f = open(self.fnm_ext, 'w')
        self.data.append(data)
        json.dump(self.data, f, indent=4)
        ic(self.data)


if __name__ == '__main__':
    def _create():
        jw = JsonWriter('jw')

        for d in [1, 'as', {1: 2, "a": "b"}, [12, 4]]:
            jw(d)

    def _check():
        with open('jw.json') as f:
            l = json.load(f)
            ic(l)

    _create()
    # _check()
