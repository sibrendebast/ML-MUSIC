import numpy as np


def angle_vectors(a, b):
    return np.arccos((a[0]*b[0] + a[1]*b[1])/(np.sqrt(a[0]**2 + a[1]**2) * np.sqrt(b[0]**2 + b[1]**2)))


def angle_points(a, b, c, d):
    e = (b[0] - a[0], b[1] - a[1])
    f = (d[0] - c[0], d[1] - c[1])
    return angle_vectors(e, f)


def angle_array_point(array_direction, array_center, point):
    user_direction = (point[0] - array_center[0], point[1] - array_center[1])
    return angle_vectors(array_direction, user_direction)

# test bench
# a = (0, 0)
# b = (1, -1)
# c = (0, 0)
# d = (200, 100)
#
# print(angle_points(a, d, c, b)/np.pi*180)
