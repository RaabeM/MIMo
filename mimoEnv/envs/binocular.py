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
from pickletools import float8
import numpy as np
import copy
import mujoco_py
from scipy.spatial.transform import Rotation as R
import math
import random


from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VISION_PARAMS#

import torch
from torchvision import transforms
from models import Autoencoder_bw5

#import sys 
#sys.path.append('..')
import mimoEnv.utils as utils

import math

SZENE_XML = os.path.join(SCENE_DIRECTORY, "binocular_scene.xml")
""" Path to the reach scene.

:meta hide-value:
"""

TEXTURES = ["texture"+str(idx) for idx in range(1,31)]


CUSTOM_VISION_PARAMS = {
    "eye_left": {"width": 512, "height": 512},
    "eye_right": {"width": 1, "height": 1},
}



class MIMoBinocularEnv(MIMoEnv):
    """ MIMo reaches for an object.

    Attributes and parameters are the same as in the base class, but the default arguments are adapted for the scenario.

    Due to the goal condition we do not use the :attr:`.goal` attribute or the interfaces associated with it. Instead,
    the reward and success conditions are computed directly from the model state, while
    :meth:`~mimoEnv.envs.reach.MIMoReachEnv._sample_goal` and
    :meth:`~mimoEnv.envs.reach.MIMoReachEnv._get_achieved_goal` are dummy functions.

    """
    def __init__(self,
                 model_path=SZENE_XML,
                 initial_qpos={},
                 n_substeps=1,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=CUSTOM_VISION_PARAMS,
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


        self.eye_pos_world_hist = np.array([[1,0,0]]) # 1st view along x axis
        self.angle_hist = []

        self.model = Autoencoder_bw5() 
        self.MODELPATH = "/home/marcel/master/autoencoder/models/test-bw5-1.pt100"
        self.model.load_state_dict(torch.load(self.MODELPATH))

        IMG_SIZE = 256
        self.PATCH_SIZE = 64 

        # Preload target textures:
        self.target_textures = {}
        for texture in TEXTURES:
            tex_id = utils.texture_name2id(self.sim.model, texture)
            self.target_textures[texture] = tex_id


        target_material_names = ["target-texture"+str(idx) for idx in range(1,8)]
        self.target_material_id = [utils.material_name2id(self.sim.model, material_name) for material_name in target_material_names]

        #self.T = 1

        
    def _step_callback(self):


        max_saliency = self._get_max_saliency_coord()

        eye_movement = [0,0,0]
        eye_movement[0] = (128 - max_saliency[0]) * math.pi/(3*256) # left eye -  horizontal
        eye_movement[1] = (128 - max_saliency[1]) * math.pi/(3*256) # left eye -  vertical   
        eye_movement[2] = 0
        self.do_eye_movement(eye_movement)    

         


    def do_eye_movement(self, movement_array):
        
        self.sim.data.qpos[16] +=  movement_array[0] # left eye -  horizontal
        self.sim.data.qpos[17] +=  movement_array[1] # left eye -  vertical   
        #self.sim.data.qpos[18] +=  movement_array[2] # left eye - torsion ?
        self.sim.data.qpos[18]  =  0 # keep torsion at 0
        self.sim.forward()  


        left_eye_id   = utils.get_body_id(self.sim.model, body_name='left_eye')
        rightt_eye_id = utils.get_body_id(self.sim.model, body_name='right_eye')

        curr_eye_pos_left_in_world = utils.body_rot_to_world(self.sim.data, [1,0,0], left_eye_id)
        last_eye_pos_left_in_world = self.eye_pos_world_hist[-1]

        #angle between curr and last pos
        dot_product = np.dot(curr_eye_pos_left_in_world, last_eye_pos_left_in_world) 
        angle = np.arccos( dot_product )


        # Add Curr eye pos and angle to history
        self.eye_pos_world_hist = np.concatenate(( self.eye_pos_world_hist, np.reshape(curr_eye_pos_left_in_world, (1,3))), axis=0)
        self.angle_hist.append(angle)




    def get_curr_pos_left(self):
        return self.sim.data.qpos[16:19]


    def _get_reconstruction(self, patch=False):

        obs = self._get_obs()
        img = obs['eye_left']

        # convert image to grayscale and in an interval [0,1]
        img_gray =  rgb2gray(img)
        img_gray /= 255

        # np to (1,W,H) tensor 
        img_gray = np.reshape(img_gray, (1, img_gray.shape[0], img_gray.shape[1])) # img_gray shape (1,W,H)
        img_tensor = torch.from_numpy(img_gray.copy()).float()


        if patch :
            # transform img_tensor in tensor of img patches
            PATCH_SIZE =  self.PATCH_SIZE 
            patches = img_tensor.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
            patches = patches.reshape(1, -1, PATCH_SIZE, PATCH_SIZE)
            patches.transpose_(0, 1)

            # forward each patch through autoencoder model
            forward_list = []
            for im in patches:
                forward_list.append(self.model(im))
           
            # stack patches with numpy back together to one single image
            forward_patch = torch.stack(forward_list)
            for_num = forward_patch.detach().numpy() 
            for_num = np.squeeze(for_num)

            arr = []
            for i in range(4):
                arr.append(np.hstack([for_num[0+4*i], for_num[1+4*i], for_num[2+4*i], for_num[3+4*i]]))
            forward_img = np.vstack(arr)

        else:
            forward_tensor = self.model.forward(img_tensor)
            forward_img = forward_tensor.detach().numpy()

        return img_gray, forward_img




    def _get_saliency_map(self, T=1, patch=False):


        img_gray, forward_img = self._get_reconstruction(patch)                        
        diff = img_gray - forward_img
        norm = np.abs(diff)

        return softmax(norm, T)



    def _get_max_saliency_coord(self, T=1):
        """Get coordinates of max value of saliency map"""
        saliency_map = np.squeeze(self._get_saliency_map(T))

        border_px = 8
        saliency_map = saliency_map[border_px:-border_px, border_px:-border_px]
        max_saliency_coord = np.where(saliency_map == np.amax(saliency_map))
        max_saliency_coord = [x[0]+border_px for x in max_saliency_coord] # get values out of weird shape
        max_saliency_coord = (max_saliency_coord[1], max_saliency_coord[0])

        return max_saliency_coord


    def reset(self):

        # ORIGINAL MIMO ENV
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal()
        obs = self._get_obs()

        # MR EDIT
        self._eye_pos_world_hist = np.zeros((1,3))
        self.angle_hist = []

        return obs


    def swap_target_texture(self, target, texture):
        """ Changes target texture. Valid emotion names are in self.target_textures, which links readable
        texture names to their associated texture ids """
        assert texture in self.target_textures, "{} is not a valid texture!".format(texture)

        new_tex_id = self.target_textures[texture]
        self.sim.model.mat_texid[self.target_material_id[target]] = new_tex_id


    def move_target(self, target_name, position_new):
        """Changes the target position in qpos. Make sure posotion_new matches the desired lenght of the target qpos interval, which is n_qpos. """

        body_id = self.sim.model.body_name2id(target_name)
        body_jntadr = self.sim.model.body_jntadr[body_id]
        joint_qpos_addr = self.sim.model.jnt_qposadr[body_jntadr]
        joint_type = self.sim.model.jnt_type[body_jntadr]
        n_qpos = utils.MUJOCO_JOINT_SIZES[joint_type]

        assert len(position_new) == n_qpos, f"{self.__class__}.{self.move_target.__name__}: input:{position_new} has not the desired length which is: {n_qpos}"

        self.sim.data.qpos[joint_qpos_addr:joint_qpos_addr + n_qpos] = position_new
        self.sim.forward()


    def random_switch_targets(self):

        for target_n in range(len(self.target_material_id)):

            self.swap_target_texture(target_n, random.choice(TEXTURES))

            random_pos = np.random.uniform(-5,5, (2))
            #random_pos[3:] = [0,0,0,0]

            target_name = 'target' + str(target_n+1)
            body_id = self.sim.model.body_name2id(target_name)
            body_jntadr = self.sim.model.body_jntadr[body_id]
            joint_qpos_addr = self.sim.model.jnt_qposadr[body_jntadr]
            random_pos = np.array([self.sim.data.qpos[joint_qpos_addr], random_pos[0], random_pos[1],0,0,0,0])

            self.move_target(target_name, random_pos)


        #if np.sum( self.vision.get_vision_obs()['eye_left'] < 50 ):
         #   self.random_switch_targets()






    ####################
    # GETTER FUNCITONS #
    ####################

    def get_reconstruction(self, patch=True):
        return self._get_reconstruction()

    def get_saliency_map(self, T=1, patch=True):
        return self._get_saliency_map(T)

    def get_saliency_coord(self, T=1):
        return self._get_max_saliency_coord(T)

    def get_hist(self):
        return self._hist




    #######################################################
    # Functions that just have to be defined but not used #
    #######################################################

    def _sample_goal(self):
        return np.zeros((0,))

    def _get_achieved_goal(self):
        return np.zeros((0,))

    def _is_success(self, achieved_goal, desired_goal):
        return False

    def _is_failure(self, achieved_goal, desired_goal):
        return False

    
    def compute_reward(self, achieved_goal, desired_goal, info):
        return 0





def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def softmax(_x, T=1):

    x = _x
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        y[i,:,:] = np.exp(x[i,:,:]/T) / (np.exp(x[i,:,:]/T).sum())
    return y





            