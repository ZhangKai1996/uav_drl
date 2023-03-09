# import numpy as np
# import math
# import matplotlib.pyplot as plt
#
# beta_list = [2.0, 1.0, 0.1]
# x = np.linspace(0.0, 100.0, 100)
# for beta in beta_list:
#     y = [-math.log(beta * (v+1e-8)) for v in x]
#     # y = [math.exp(beta * v) for v in x]
#     plt.plot(x, y, label="{:>+4.2f}".format(beta))
#
#
# plt.legend()
# plt.show()

# import threading
# import time
#
#
# # 新线程执行的代码:
# def loop():
#     print('thread %s is running...' % threading.current_thread().name)
#     n = 0
#     while n < 5:
#         n = n + 1
#         print('thread %s >>> %s' % (threading.current_thread().name, n))
#         time.sleep(1)
#     print('thread %s ended.' % threading.current_thread().name)
#
#
# print('thread %s is running...' % threading.current_thread().name)
# t = threading.Thread(target=loop, name='LoopThread')
# t.start()
# t.join()
# print('thread %s ended.' % threading.current_thread().name)

# import torch as th
# import numpy as np
#
# a = np.random.randn(32, 3, 10)
# b = np.random.randn(32, 3, 10)
#
# c = (a, b, )
# print(th.from_numpy(np.array(c)).shape)

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


a = (1.0, 1.0)
b = (-1.0, 1.0)
c = (-1.0, -1.0)
d = (1.0, -1.0)
e = (1.0, 0)
f = (-1.0, 0)
g = (0, -1.0)
h = (0, 1.0)
i = (0, 0)
print(absolute_bearing(a, minus=True))
print(absolute_bearing(b, minus=True))
print(absolute_bearing(c, minus=True))
print(absolute_bearing(d, minus=True))
print(absolute_bearing(e, minus=True))
print(absolute_bearing(f, minus=True))
print(absolute_bearing(g, minus=True))
print(absolute_bearing(h, minus=True))
print(absolute_bearing(i, minus=True))
