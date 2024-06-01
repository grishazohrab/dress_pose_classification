import os

import pandas as pd

from detect_person import is_person
from detect_position import detect_position


def process_data_plt(dir_path, csv_path):
    dress_names = os.listdir(dir_path)

    res_info = {"path": [], "type": [], "pose": []}

    for name in dress_names:
        images = [os.path.join(dir_path, name, img_name) for img_name in os.listdir(os.path.join(dir_path, name))]

        for img in images:
            res_info["path"].append(img)
            if is_person(img):
                res_info["type"].append("person")
                p = detect_position(img)
                if p == 1:
                    res_info["pose"].append("front")
                elif p == 0:
                    res_info["pose"].append("back")
                else:
                    res_info["pose"].append("unknown")
            else:
                res_info["type"].append("dress")
                res_info["pose"].append("unknown")

    df = pd.DataFrame.from_dict(res_info)
    df.to_csv(csv_path)


if __name__ == '__main__':
    pass
