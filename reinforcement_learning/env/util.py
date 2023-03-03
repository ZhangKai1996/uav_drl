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


if __name__ == '__main__':
    """
    测试角度计算函数：
        line = [px, py, pz, 
                pos.x_val, pos.y_val, pos.z_val, 
                vel.x_val, vel.y_val, vel.z_val, 
                x_z_view, x_y_view]
    """
    with open('./record.csv', 'r', newline='') as f:
        for line in f.readlines():
            print(line)
            line = line.strip('\r\n').split(',')
            line = [float(v) for v in line]
            """        
                [px, py, pz, pos.x_val, pos.y_val, pos.z_val, 
                 vel.x_val, vel.y_val, vel.z_val, x_z_view, x_y_view]
            """
            print(line)
            print(line[6:9])
            y_z_vel = absolute_bearing((line[7], line[8]))
            x_z_vel = absolute_bearing((line[6], line[8]))
            x_y_vel = absolute_bearing((line[6], line[7]))
            print('x v:', y_z_vel)
            print('y v:', x_z_vel)
            print('z v:', x_y_vel)
            print()

            p1, p0 = (line[0], line[1]), (line[3], line[4])
            print(p0, p1)
            print('\tNED z:', relative_bearing(p0=p0, p1=p1, reference=x_y_vel))
            print('\tNED z:', relative_bearing(p0=p0, p1=p1))
            print('\tNED z o:', absolute_bearing(p0))
            print('\tNED z p:', absolute_bearing(p1))
            print(relative_bearing(p0=p0, p1=p1, reference=x_y_vel) == line[-1])
            print()

            p1, p0 = (line[0], line[2]), (line[3], line[5])
            print(p0, p1)
            print('\tNED y:', relative_bearing(p0=p0, p1=p1))
            print('\tNED y o:', absolute_bearing(p0))
            print('\tNED y p:', absolute_bearing(p1))
            print()

            p1, p0 = (line[1], line[2]), (line[4], line[5])
            print(p0, p1)
            print('\tNED x:', relative_bearing(p0=p0, p1=p1))
            print('\tNED x o:', absolute_bearing(p0))
            print('\tNED x p:', absolute_bearing(p1))
            print()
