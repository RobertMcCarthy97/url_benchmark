import pybullet_multigoal_gym as pmg
import minitouch.env
import gym
from bullet_envs import gym_wrappers
# import numpy as np

class MultiGoalObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = gym.spaces.Dict(
                {
                    "pixels": env.observation_space['observation'],
                    "state": env.observation_space['state']
                }
            )
    
    def observation(self, obs):
        obs_dict = {'pixels': obs['observation'], 'state': obs['state']}
        return obs_dict
    
class MiniTouchObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = gym.spaces.Dict(
                {
                    "pixels": env.observation_space[0],
                    "state": env.observation_space[1]
                }
            )
    
    def observation(self, obs):
        obs_dict = {'pixels': obs[0], 'state': obs[1]}
        return obs_dict

def create_multigoal_gym_env(task='reach'):
    # TODO: change image size, include depth image (and make sure compatible with code)
    
    camera_setup = [
        {
            'cameraEyePosition': [-1.0, 0.25, 0.6],
            'cameraTargetPosition': [-0.6, 0.05, 0.2],
            'cameraUpVector': [0, 0, 1],
            'render_width': 84,
            'render_height': 84
        }
    ]
    
    env = pmg.make_env(
        # task args ['reach', 'push', 'slide', 'pick_and_place', 
        #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
        task=task,
        gripper='parallel_jaw',
        num_block=4,  # only meaningful for multi-block tasks
        render=False,
        binary_reward=False,
        max_episode_steps=100,
        # image observation args
        image_observation=True,
        depth_image=False,
        goal_image=False,
        visualize_target=True,
        camera_setup=camera_setup,
        observation_cam_id=0,
        goal_cam_id=0,
        # curriculum args
        use_curriculum=False,
        num_goals_to_generate=90)
    
    env = MultiGoalObsWrapper(env)
    env = gym_wrappers.DMEnvFromGym(env)
    
    return env

def create_minitouch_env(task="Pushing-v0"):
    env = gym.make(task)
    env = MiniTouchObsWrapper(env)
    env = gym_wrappers.DMEnvFromGym(env)
    
    return env