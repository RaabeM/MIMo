""" This module contains a simple reaching experiment in which MIMo tries to touch a hovering ball.

The scene consists of MIMo and a hovering ball located within reach of MIMos right arm. The task is for MIMo to
touch the ball.
MIMo is fixed in position and can only move his right arm. His head automatically tracks the location of the ball,
i.e. the visual search for the ball is assumed.
Sensory input consists of the full proprioceptive inputs. All other modalities are disabled.

The ball hovers stationary. An episode is completed successfully if MIMo touches the ball, knocking it out of
position. There are no failure states. The position of the ball is slightly randomized each trial.

Reward shaping is employed, with a negative reward based on the distance between MIMos hand and the ball. A large fixed
reward is given when he touches the ball.

The class with the environment is :class:`~mimoEnv.envs.reach.MIMoReachEnv` while the path to the scene XML is defined
in :data:`REACH_XML`.
"""
import os
import numpy as np
import copy
import mujoco_py
import cv2
from scipy.spatial.transform import Rotation as R
import math


from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VISION_PARAMS


SACCADE_XML = os.path.join(SCENE_DIRECTORY, "saccade_scene.xml")
""" Path to the reach scene.

:meta hide-value:
"""


class MIMoSaccadeEnv(MIMoEnv):
    """ MIMo reaches for an object.

    Attributes and parameters are the same as in the base class, but the default arguments are adapted for the scenario.

    Due to the goal condition we do not use the :attr:`.goal` attribute or the interfaces associated with it. Instead,
    the reward and success conditions are computed directly from the model state, while
    :meth:`~mimoEnv.envs.reach.MIMoReachEnv._sample_goal` and
    :meth:`~mimoEnv.envs.reach.MIMoReachEnv._get_achieved_goal` are dummy functions.

    """
    def __init__(self,
                 model_path=SACCADE_XML,
                 initial_qpos={},
                 n_substeps=1,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=DEFAULT_VISION_PARAMS,
                 vestibular_params=None,
                 goals_in_observation=False,
                 done_active=True):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)


        self._last_pos_eye_left = np.zeros(3)
        self._last_pos_eye_right = np.zeros(3)
        self._hist = np.array([[0,0,0]])

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward.

        A negative reward is given based on the distance between MIMos fingers and the ball.
        If contact is made a fixed positive reward of 100 is granted. The achieved and desired goal parameters are
        ignored.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """
        # MR - TEST: Mimo has to set eyes into neutral position
        obs = self._get_obs() 
        img_left = obs['eye_left']
        max_saliency_left = np.where(img_left == np.amax(img_left))[0]

        # reward is invers distance from max saliency to middle of the frame 
        epsilon = 0.001
        reward = 100/(np.linalg.norm(max_saliency_left-128) + epsilon)

        #for i in range(128-30, 128+30):
          # for j in range(128-30, 128+30):
                 # reward += saliencyMap[i][j] 
        # TODO: If use the summed up saliency as reward normalize with sliency of whole img
        # => reward = reward / sum(sliencyMap)
    


        return reward

    def _is_success(self, achieved_goal, desired_goal):
        """ Determines the goal states.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `True` if the ball is knocked out of position.
        """

        return False

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns `False`.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `False`
        """
        return False

    def _sample_goal(self):
        """ Dummy function. Returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros((0,))

    def _get_achieved_goal(self):
        """ Dummy function. Returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros((0,))

    def _reset_sim(self):
        """ Resets the simulation.

        We reset the simulation and then slightly move both MIMos arm and the ball randomly. The randomization is
        limited such that MIMo can always reach the ball.

        Returns:
            bool: `True`
        """

        self.sim.set_state(self.initial_state)
        self.sim.forward()

        # perform 10 random actions
        for _ in range(10):
            action = self.action_space.sample()
            self._set_action(action)
            self.sim.step()
            self._step_callback()

        # reset target in random initial position and velocities as zero
        qpos = self.sim.data.qpos
        qpos[[-7, -6, -5]] = np.array([
            self.initial_state.qpos[-7] + self.np_random.uniform(low=-0.1, high=0, size=1)[0],
            self.initial_state.qpos[-6] + self.np_random.uniform(low=-0.2, high=0.1, size=1)[0],
            self.initial_state.qpos[-5] + self.np_random.uniform(low=-0.1, high=0, size=1)[0]
        ])
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )

        self.sim.set_state(new_state)
        self.sim.forward()
        self.target_init_pos = copy.deepcopy(self.sim.data.get_body_xpos('target'))
        return True

    def _step_callback(self):

        last_eye_pos_left = self._hist[-1]
        print(last_eye_pos_left)

        curr_eye_pos_left = self.sim.data.qpos[16:19]


        self._hist = np.concatenate(( self._hist, np.array([curr_eye_pos_left]) ))

        #print(f"horizontal: {self.sim.data.qpos[16]}, vertical {self.sim.data.qpos[17]}, torsional {self.sim.data.qpos[18]}")
        #print(f"last: {last_eye_pos_left}")
        #print(f"curr: {curr_eye_pos_left}")
        print(f"angle: {get_rot_angle(last_eye_pos_left, curr_eye_pos_left)}" )
        
        #for x in self._hist:
         #  print(x)


        print(self._hist)
        #self._last_pos_eye_left = curr_eye_pos_left



    def _get_obs(self):
        """Returns the observation.
            Changed Base function from Memo_Env. The Vision goes through the cv2.saliency pipeline first before safed to observation_dict
        """
        # robot proprioception:
        proprio_obs = self._get_proprio_obs()
        observation_dict = {
            "observation": proprio_obs,
        }
        if self.vision:
            vision_obs = self._get_vision_obs()
            for sensor in vision_obs:
                # Saliency is applied to vision so Mimo only sees the saliency
                saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
                (success, saliencyMap_left) = saliency.computeSaliency(vision_obs[sensor])
                observation_dict[sensor] = saliencyMap_left
                #observation_dict[sensor] = vision_obs[sensor]
        if self.goals_in_observation:
            achieved_goal = self._get_achieved_goal()
            observation_dict["achieved_goal"] = copy.deepcopy(achieved_goal)
            observation_dict["desired_goal"] = copy.deepcopy(self.goal)

        return observation_dict








def rotation(rotation_vec : '3D vector with, [0] horizontal, [1] vertical, [2] torsional rotation angle in radians'): 

    angle_vert = rotation_vec[0]
    angle_horz = rotation_vec[1]
    angle_tor  = rotation_vec[2]

    # Vertical Rotation along y-axis
    r_y = R.from_matrix([
        [math.cos(angle_vert), 0, math.sin(angle_vert)],
        [0, 1, 0],
        [-math.sin(angle_vert),0,math.cos(angle_vert)]
    ])


    # Horizontal Rotation along z-axis
    r_z = R.from_matrix([
        [math.cos(angle_horz), -math.sin(angle_horz), 0],
        [math.sin(angle_horz),  math.cos(angle_horz), 0],       
        [0,0,1]
    ])

    # Torsional Rotation along x-axis
    r_x = R.from_matrix([
        [1,0,0],
        [0, math.cos(angle_tor), -math.sin(angle_tor)],
        [0, math.sin(angle_tor),  math.cos(angle_tor)]
    ])


    r = r_x * r_y * r_z

    unit_vec = np.array([1,0,0])

    return r.apply(unit_vec)
    

def get_rot_angle(a,b):

    a_rotated = rotation(a)
    b_rotated = rotation(b)

    dot_product = np.dot(a_rotated,b_rotated) 

    if dot_product > 1: 
        dot_product = 1

    return math.acos( dot_product )