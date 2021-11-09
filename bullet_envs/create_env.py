import pybullet_multigoal_gym as pmg
import minitouch.env
import gym
from bullet_envs import gym_wrappers
import numpy as np

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
                    "pixels": gym.spaces.Box(low=0.0, high=255.0, shape=self.reset()['pixels'].shape, dtype=np.float32),
                    "state": env.observation_space[1]
                }
            )
        print('WARNING:manually creating obs space!!!')
        
        
    def observation(self, obs):
        obs_dict = {'pixels': obs[0], 'state': obs[1]}
        # is grayscale so increase to 3 image dims
        obs_dict['pixels'] = np.tile(obs_dict['pixels'], (3,1,1))
        # move image channel
        obs_dict['pixels'] = obs_dict['pixels'].transpose(1,2,0)
        
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
        # task args ['reach', 'push', 'slide', 'pick_and_place']
        task=task,
        render=False,
        binary_reward=True,
        max_episode_steps=100,
        # image observation args
        image_observation=True,
        depth_image=False,
        camera_setup=camera_setup,
        )
    
    env = MultiGoalObsWrapper(env)
    env = gym_wrappers.DMEnvFromGym(env)
    
    return env


def create_minitouch_env(task="Pushing-v0"):
    env = gym.make(task)
    env = MiniTouchObsWrapper(env)
    env = gym_wrappers.DMEnvFromGym(env)
    
    return env