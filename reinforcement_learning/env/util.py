import math


def absolute_bearing(p, minus=False):
    delta_x, delta_y, *_ = p
    if delta_y == 0.0:
        if delta_x == 0.0:
            return math.inf
        angle = 90.0 if delta_x > 0 else 270.0
    elif delta_x == 0.0:
        angle = 0.0 if delta_y > 0 else 180.0
    else:
        tan_v = delta_x / delta_y
        angle = math.atan(tan_v) * 180.0 / math.pi
        if tan_v > 0:
            angle += int(delta_x < 0) * 180.0
        else:
            angle -= int(delta_x > 0) * 180.0

    angle = (angle + 360) % 360
    # angle: [0, 360) --> (-180, 180]
    if minus and angle >= 180.0:
        angle -= 360
    return angle


def relative_bearing(p0, p1, reference=None):
    angle = absolute_bearing((p1[0] - p0[0], p1[1] - p0[1]))
    if reference is not None and reference != math.inf:
        angle = (angle - reference + 360) % 360
    if angle >= 180.0:
        angle -= 360
    return angle


def relative_bearing_z(p0, p1, reference=None):
    delta_p = (-(p1[2] - p0[2]), math.sqrt(math.pow(p0[0]-p1[0], 2)+math.pow(p0[1]-p1[1], 2)))
    angle = absolute_bearing(delta_p, minus=True)
    if reference is not None and reference != math.inf:
        angle = (angle - reference + 360) % 360
    if angle >= 180.0:
        angle -= 360
    return angle
