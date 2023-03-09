import math

import numpy as np
import cv2
import random
import copy

color_dict = {0: (0, 0, 255),
              1: (0, 255, 0),
              2: (255, 0, 0)}


def add_ADI(image, view, width, height, limits):
    """
    飞行姿态指示仪（Attitude Director Indicator, ADI）
    """
    # horizontal
    height_h = int(height // 4)
    img_h = np.ones((height_h, width, 3), dtype=np.uint8)
    center_x_h, center_y_h = width // 2, height_h // 2
    # img_h = cv2.rectangle(img_h,
    #                       (0, 0),
    #                       (width - 1, height_h),
    #                       (255, 255, 255), 1)
    img_h = cv2.line(img_h,
                     (center_x_h, 0),
                     (center_x_h, height_h),
                     (255, 255, 255), 1)
    # vertical
    width_v = int(height // 4)
    img_v = np.ones((height, width_v, 3), dtype=np.uint8)
    center_x_v, center_y_v = width_v // 2, height // 2
    # img_v = cv2.rectangle(img_v,
    #                       (0, 0),
    #                       (width_v, height - 1),
    #                       (0, 0, 0), 1)
    img_v = cv2.line(img_v,
                     (0, center_y_v),
                     (width_v, center_y_v),
                     (255, 255, 255), 1)
    # blank area
    img_blank = np.ones((height_h, width_v, 3), dtype=np.uint8)
    # img_blank = cv2.putText(img_blank,
    #                         str(1),
    #                         (width_v // 2, height_h // 2),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.4,
    #                         (0, 0, 0), 1, cv2.LINE_AA)
    # ball
    radius_h = 5
    radius_v = 5
    tmp_h, tmp_v = height_h - radius_h, width_v - radius_v
    [horizontal, vertical] = view
    j = 0

    color = (255, 255, 255)
    limit = limits['h']
    if horizontal == math.inf:
        img_h = cv2.circle(img_h,
                           (center_x_h, int(tmp_h * (j + 1 / 2))),
                           radius_h, color, 1)
    else:
        if abs(horizontal) >= limit:
            horizontal = abs(horizontal) / horizontal * limit
        delta = int(horizontal / limit * width / 2)
        img_h = cv2.circle(img_h,
                           (center_x_h + delta, int(tmp_h * (j + 1 / 2))),
                           radius_h, color, -1)

    limit = limits['v']
    if vertical == math.inf:
        img_v = cv2.circle(img_v,
                           (int(tmp_v * (j + 1 / 2)), center_y_v),
                           radius_v, color, 1)
    else:
        if abs(vertical) >= limit:
            vertical = abs(vertical) / vertical * limit
        delta = int(vertical / limit * height / 2)
        img_v = cv2.circle(img_v,
                           (int(tmp_v * (j + 1 / 2)), center_y_v - delta),
                           radius_v, color, -1)

    return np.hstack([np.vstack([image, img_h]),
                      np.vstack([img_v, img_blank])])


border_property = {'color': (100, 100, 100), 'thickness': 2}  # BGR
segment_property = {'color': (107, 55, 19), 'thickness': 1}  # BGR

decimal: int = 1


# ---------
# functions
# ---------
def resolution(border, scale: int):
    """
    分辨率（可视化界面的长和宽）
    假设border为[1.0, 9.0, 1.0, 7.0]，scale为100，则分辨率为800x600
    """
    min_x, max_x, min_y, max_y, *_ = border
    return (
        int((max_x - min_x) * scale),
        int((max_y - min_y) * scale)
    )


def convert_coord_to_pixel(points, border, scale: int):
    """
    将点坐标（lng, lat）转化为像素点的位置（x, y）
    """
    min_x, max_x, min_y, max_y, *_ = border
    scale_x = (max_x - min_x) * scale
    scale_y = (max_y - min_y) * scale
    return [
        (
            int((x - min_x) / (max_x - min_x) * scale_x),
            int((max_y - y) / (max_y - min_y) * scale_y)
        )
        for [x, y, *_] in points
    ]


class CVRender:
    def __init__(self, video_path, border=(-100.0, 100.0, -100.0, 100.0), scale=3, channel=3, fps=8,
                 extra_shape=(0, 0, 0)):
        """
        用于录制视频
        """
        self.scale = scale
        self.channel = channel
        self.border = border
        self.width, self.height = resolution(border, scale)
        # 白色底图
        image = np.ones((self.height, self.width, self.channel), np.uint8) * 255
        lines = [[(border[0], 0), (border[1], 0)],
                 [(0, border[2]), (0, border[3])]]
        self.base_image = self.__add_lines(lines, image, thickness=1)
        height_h = self.height - extra_shape[1]
        width_v = extra_shape[2]
        assert height_h > 0
        self.img_blank = np.ones((height_h, width_v, 3), dtype=np.uint8) * 255
        cv2.rectangle(self.img_blank, (0, 0), (width_v, height_h), (0, 0, 0), 1)

        self.video = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'MJPG'), fps,
            (self.width + extra_shape[2], self.height + extra_shape[1])
        )

    def __add_lines(self, lines, image, color=(0, 0, 0), thickness=-1):
        if len(lines) <= 0:
            return image

        border = self.border
        scale = self.scale
        for [p0, p1, *_] in lines:
            [start, end] = convert_coord_to_pixel([p0, p1], border=border, scale=scale)
            cv2.line(image, start, end, color, thickness)
        return image

    def __add_points(self, points, image, color=(0, 0, 0), name='point', radius=5, thickness=-1, font_scale=0.4,
                     font=cv2.FONT_HERSHEY_SIMPLEX, has_arrow=False, scale_arrow=10):
        """
        在image里增加点（图标为圆）
        """
        border = self.border
        scale = self.scale
        for idx, [x, y, z, *other] in enumerate(points):
            point = (x, y)
            coord = convert_coord_to_pixel([point, ], border=border, scale=scale)[0]
            if isinstance(color, tuple):
                cv2.circle(image, coord, radius, color, thickness)
            elif color == 'z':
                range_mixed = min(510, max(-z / 100 * 510, 0))
                if range_mixed <= 255:
                    cv2.circle(image, coord, radius, (0, 255, range_mixed), thickness)
                else:
                    cv2.circle(image, coord, radius, (0, 510 - range_mixed, 255), thickness)
            else:
                raise NotImplementedError
            cv2.putText(image, name + str(idx + 1), (coord[0], coord[1] + radius * 2), font, font_scale, (0, 0, 0), 1)
            text = '({},{},{})'.format(round(x, decimal), round(y, decimal), round(z, decimal))
            cv2.putText(image, text, (coord[0], coord[1] + radius * 5), font, font_scale, (0, 0, 0), 1)
            if has_arrow and len(other) >= 2:
                [x_vel, y_vel, z_vel, *_] = other
                point_ = (x + x_vel * scale_arrow, y + y_vel * scale_arrow)
                image = self.__add_lines([[point, point_, ]], image, color=(255, 0, 0), thickness=1)
                point_ = (x, y + z_vel * scale_arrow)
                image = self.__add_lines([[point, point_, ]], image, color=(255, 0, 255), thickness=1)

        return image

    def render(self, targets, locations, obs=None, mode='human', wait=1):
        if self.video is None:
            return

        # 底图
        base_image = copy.deepcopy(self.base_image)
        # 目标点
        image = self.__add_points(targets, base_image, name='target', color='z', radius=10)
        # 无人机位置点
        image = self.__add_points(locations, image, name='drone', color='z', radius=10,
                                  thickness=1, has_arrow=True)
        # 无人机和目标点之间增加连线
        lines = [(target, loc) for target, loc in zip(targets, locations)]
        image = self.__add_lines(lines, image, thickness=1)

        if obs is not None:
            image = np.hstack([image, np.vstack([obs, self.img_blank])])

        # 图片渲染
        cv2.namedWindow(mode, cv2.WINDOW_NORMAL)
        cv2.imshow(mode, image)
        cv2.waitKey(wait)
        # while True:
        #     if cv2.waitKey(0) == 113:
        #         break
        self.video.write(image)

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None
