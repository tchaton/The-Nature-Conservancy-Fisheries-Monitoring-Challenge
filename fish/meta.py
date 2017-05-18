from tqdm import tqdm_notebook as tq

import json
from utils import *

classes_dictionary = {"NoF": 0, "OTHER": 1, "ALB": 2, "BET": 3, "DOL": 4, "LAG": 5, "SHARK": 6, "YFT": 7}


def find_dic(f_name, list_of_dics):
    for dic in list_of_dics:
        file = dic["filename"].split("/")[-1]
        if file == f_name:
            return dic


def remove_class(list_of_dicts):
    new_list = []
    for dic in list_of_dicts:
        dic.pop("class")
        new_list.append(dic)
    return new_list


def is_in_rect(x, y, rect, margin=0):
    """ This function check if a point is in a rectangle, 
        the rectangle can be a dictionary or a tuple of 4 points"""

    m = margin
    if type(rect) == dict:
        ass1 = rect["x"] - m < x < rect["x"] + rect["width"] + m
        ass2 = rect["y"] - m < y < rect["y"] + rect["height"] + m
    else:
        ass1 = rect[0] - m < x < rect[2] + m
        ass2 = rect[1] - m < y < rect[3] + m
    return ass1 and ass2


def overlap(r1, r2):
    """Overlapping rectangles overlap both horizontally & vertically
    """
    return (range_overlap(r1["x"], r1["x"] + r1["width"], r2[0], r2[2]) and
            range_overlap(r1["y"], r1["y"] + r1["height"], r2[1], r2[3]))


def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)


def head_tail_first(img_rect, ht):
    """ This function take a list of rectangles and put first 
        in list the rectangle that has the annotated head and tail."""
    x1 = ht[0]["x"]
    y1 = ht[0]["y"]
    x2 = ht[1]["x"]
    y2 = ht[1]["y"]
    index = None

    for i, rect in enumerate(img_rect):
        # We need to verify if the two points are in the rectangle. 
        # We add a margin m.
        m = 20

        ass1 = is_in_rect(x1, y1, rect, m)
        ass2 = is_in_rect(x2, y2, rect, m)
        if ass1 and ass2:
            index = i

            # If the good rectangle was found
    if index is not None:

        # We put it first 
        rectangle_with_ht = img_rect.pop(index)
        img_rect.insert(0, rectangle_with_ht)
        return img_rect, True

    # If we didn't find the rectangle, we specify it as a second variable
    else:
        return img_rect, False


def find_angle(ht):
    x = ht[0]["x"] - ht[1]["x"]
    y = ht[0]["y"] - ht[1]["y"]
    return np.angle(complex(x, y), True)


def create_metadata():
    mk("mask/")
    paths = h_gen.get_all_images()
    jsons_r = h_gen.get_all_images(regex="json/*")
    jsons_ht = h_gen.get_all_images(regex="heads_tails/*")

    # Loading the rectangles
    rectangles = []
    for json_path in jsons_r:
        with open(json_path, 'r+') as f:
            labels = json.load(f)
            rectangles += labels

    # Loading the heads and tails
    heads_tails = []
    for json_path in jsons_ht:
        with open(json_path, 'r+') as f:
            labels = json.load(f)
            heads_tails += labels

    # Fill the infos
    results = {}
    for path in tq(paths):
        dic = {"path": path, "filename": path.split("/")[-1], "class": path.split("/")[-2]}
        dic["code"] = classes_dictionary[dic["class"]]
        size = h_gen.resolution(path)
        dic["height"] = size[1]
        dic["width"] = size[0]

        # The image doesn't necessarily has a fish
        ht = find_dic(dic["filename"], heads_tails)
        if ht is not None:
            ht = remove_class(ht["annotations"])
            if len(ht) == 2:
                dic["head_tail"] = ht
                dic["angle"] = find_angle(ht)
            else:
                ht = None

        img_rect = find_dic(dic["filename"], rectangles)
        if img_rect is not None:
            img_rect = remove_class(img_rect["annotations"])
            if ht is not None:
                img_rect, found = head_tail_first(img_rect, ht)
                dic["ht_first"] = found
            dic["rectangles"] = img_rect
        results[dic["filename"]] = dic

    return results


def create_metadata_test(sub_fold):
    paths = h_gen.get_all_images(regex=sub_fold + "test_data/*.jpg")
    filenames = [x.split("/")[-1] for x in paths]
    heights = [h_gen.resolution(x)[1] for x in paths]
    widths = [h_gen.resolution(x)[0] for x in paths]

    return list(zip(paths, filenames, heights, widths))
