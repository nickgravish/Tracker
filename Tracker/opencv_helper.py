

import numpy as np

def opencv_contour_to_list(contour):
    return [[int(x[0][0]), int(x[0][1])] if x != [] else [] for x in contour]


def list_to_opencv_contours(lst):
    return np.array([[[x, y]] for x, y in lst], dtype = np.int32)

