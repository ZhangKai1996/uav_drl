import setup_path

import numpy as np
import airsim

from .airsim_env import AirSimEnv
from .render import add_ADI, CVRender
from .util import *


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape, step_length=1.0):
        super().__init__(image_shape)
        self.image_shape = image_shape
        self.step_length = step_length

        # 启动与连接仿真
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        # 选取无人机的摄像头
        self.image_requests = [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False), ]
        # 目的地（会随机变化）
        self.target_point = None
        # ADI的方向显示限制范围
        self.limit = {'h': 90.0, 'v': 45.0}
        # 记录训练飞行过程（视频）
        self.cv_render = CVRender(video_path='scenario.avi', extra_shape=image_shape)

    def reset(self):
        client = self.client
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.moveToZAsync(-10.0, 3).join()

        self.target_point = np.concatenate(
            [np.random.randint(-100, 100, size=(1, 2)),
             np.random.randint(-100, -10, size=(1, 1))],
            axis=1
        )[0]
        print('Reset is completed!', self.target_point)
        return self.__get_obs()[0]

    def step(self, action, duration=3, render=False):
        client = self.client
        step_size = self.step_length
        state = client.getMultirotorState().kinematics_estimated
        vel = state.linear_velocity
        new_x_val, new_y_val = vel.x_val + action[0] * step_size, vel.y_val + action[1] * step_size
        client.moveByVelocityAsync(
            new_x_val,
            new_y_val,
            vel.z_val + action[2],
            duration=duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, absolute_bearing((new_x_val, new_y_val)))
        ).join()

        reward, done = self.__compute_reward()
        next_state, obs_rgb = self.__get_obs()
        if render:
            self.render(obs_rgb=obs_rgb)
        return next_state, reward, done, {}

    def render(self, obs_rgb=None, mode='human', wait=1):
        client = self.client

        state = client.getMultirotorState().kinematics_estimated
        vel = state.linear_velocity
        pos = state.position
        location = [pos.x_val, pos.y_val, pos.z_val,
                    vel.x_val, vel.y_val, vel.z_val]
        self.cv_render.render(np.array([self.target_point, ]),
                              np.array([location, ]),
                              obs=obs_rgb, wait=wait)

    def __get_obs(self):
        client = self.client
        image_requests = self.image_requests
        p = self.target_point

        state = client.getMultirotorState().kinematics_estimated
        pos = state.position
        vel = state.linear_velocity
        x_y_view = relative_bearing((pos.x_val, pos.y_val), (p[0], p[1]),
                                    reference=absolute_bearing((vel.x_val, vel.y_val)))  # z
        x_z_view = relative_bearing_z((pos.x_val, pos.y_val, pos.z_val), p,
                                      reference=0)  # y
        view = [x_y_view, x_z_view]
        # print('view:', view)
        # print(vel.x_val, vel.y_val, absolute_bearing((vel.x_val, vel.y_val)), view)

        response = client.simGetImages(image_requests)[0]
        height, width = response.height, response.width
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = np.reshape(img1d, (height, width, 3))
        img_rgb = add_ADI(img_rgb, view, width, height, limits=self.limit)

        return np.transpose(img_rgb, (2, 0, 1)), img_rgb

    def __compute_reward(self):
        client = self.client
        f_point = self.target_point

        pos = client.getMultirotorState().kinematics_estimated.position

        x_square = math.pow(pos.x_val - f_point[0], 2)
        y_square = math.pow(pos.y_val - f_point[1], 2)
        z_square = math.pow(pos.z_val - f_point[2], 2)
        dist = math.sqrt(x_square + y_square + z_square)
        rew = -dist * 0.001

        collision = client.simGetCollisionInfo().has_collided
        if collision:
            rew -= 100
        return rew, collision

    def close(self):
        client = self.client
        client.armDisarm(False)  # lock
        client.enableApiControl(False)  # release control
