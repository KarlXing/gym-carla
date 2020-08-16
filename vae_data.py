#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla
import numpy as np
import argparse
import random 

parser = argparse.ArgumentParser()
parser.add_argument('--task', default=0, type=int)
parser.add_argument('--num-imgs', default=10000, type=int)
parser.add_argument('--points-path', default='./points.npy', type=str)
parser.add_argument('--action-repeat', default=10, type=int)
args = parser.parse_args()

weathers = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.CloudySunset]
weather = weathers[args.task]
points = np.load(args.points_path, allow_pickle=True).item()
start_point = points['start'][args.task]
end_point = points['end'][args.task]
route = points['routes'][args.task]

def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town07',  # which town to simulate
    # 'task_mode': 'random',  # removed
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 16,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 1,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': True,  # whether to output PIXOR observation
    'start_point': start_point,
    'end_point': end_point,
    'weather': weather,
    'ip': 'localhost'
  }

  # Set gym-carla environment
  save_imgs = []
  env = gym.make('carla-v0', params=params)
  obs = env.reset()
  save_imgs.append(obs['camera'])

  while(len(save_imgs) < args.num_imgs):
    action = env.action_space.sample()
    action[0] = 2
    for i in range(args.action_repeat):
      obs,r,done,info = env.step(action)
      if done:
        break

    save_imgs.append(obs['camera'])
    if done:
      env.update_start_point(random.choice(route))
      obs = env.reset()

  np.save('task%d-imgs' % args.task, save_imgs)

if __name__ == '__main__':
  main()