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

SZENE_XML = os.path.join(SCENE_DIRECTORY, "pictures_scene.xml")
""" Path to the reach scene. """

TEXTURES = ["texture"+str(idx) for idx in range(1,31)]


# Foveated camera resolutions
CUSTOM_VISION_PARAMS = {
    "eye_left1": {"width": 144, "height": 144},
    "eye_left2": {"width": 128, "height": 128},
    "eye_left3": {"width":  80, "height":  80},
    "eye_left4": {"width":  64, "height":  64},

    "eye_right1": {"width": 144, "height": 144},
    "eye_right2": {"width": 128, "height": 128},
    "eye_right3": {"width":  80, "height":  80},
    "eye_right4": {"width":  64, "height":  64},
}



class MIMoBinoc_SaccadeEnv(MIMoEnv):
    """ MIMo reaches for an object.

        Class to demonstrate saccades with binocular saliency map by Marcel Raabe
    """
    def __init__(self,
                 model_path=SZENE_XML,
                 initial_qpos={},
                 n_substeps=1,
                 proprio_params=None,
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


    def _get_reconstruction(self, patch=False, cropscales=False):


        obs = self._get_obs()
        img = obs['eye_left']

        # convert image to grayscale and in an interval [0,1]
        img_gray =  rgb2gray(img)
        img_gray /= 255

        # np to (1,W,H) tensor 
        img_gray = np.reshape(img_gray, (1, img_gray.shape[0], img_gray.shape[1])) # img_gray shape (1,W,H)

        if cropscales: 
            img_tensor = self.get_cropscales(img_gray)
        else:
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

        return img_tensor.detach().numpy(), forward_img




    def _get_saliency_map(self, T=1, patch=False, cropscales=False):


        img_gray, forward_img = self._get_reconstruction(patch, cropscales)                        
        diff = img_gray - forward_img
        norm = np.abs(diff)

        return softmax(norm, T)



    def _get_max_saliency_coord(self, T=1, cropscales=False):
        """Get coordinates of max value of saliency map"""
        

        saliency_map = self._get_saliency_map(T, cropscales=cropscales)
        border_px = 8

        if len(saliency_map.shape) == 4:

            saliency_map = np.squeeze(saliency_map)           

            # Set saliency at borders to zero
            mask = np.zeros((saliency_map.shape[0], saliency_map.shape[1], saliency_map.shape[2]))
            border_px = 4
            mask[:,border_px:-border_px, border_px:-border_px] = 1
            saliency_map *= mask

            # Set saliency at center to zero, but do not do for the most inner picture
            start_center = int(saliency_map.shape[1]/4   + border_px)
            end_center   = int(saliency_map.shape[1]*3/4 - border_px)
            for i in range(saliency_map.shape[0]-1):
                saliency_map[i, start_center:end_center, start_center:end_center] = 0
                

            max_saliency_coord_list = []
            for i in range(saliency_map.shape[0]):
                max_saliency_coord = np.where(saliency_map[i] == np.amax(saliency_map[i]))
                max_saliency_coord_list.append((max_saliency_coord[1][0], max_saliency_coord[0][0]))

            return max_saliency_coord_list, saliency_map
                
        
        else: 
            saliency_map = np.squeeze(saliency_map)

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


    def get_cropscales(self, img_gray) -> torch.tensor: 

        #img_gray, _ = self.get_reconstruction()
        img_tensor  = torch.from_numpy(img_gray)

        WIDTH_L  = self.vision_params['eye_left']['width']
        HEIGHT_L = self.vision_params['eye_left']['height']

        output_scale = (int(WIDTH_L/4), int(HEIGHT_L/4))

        tensor_crop1 = transforms.functional.resized_crop(img_tensor, 0, 0, WIDTH_L,  HEIGHT_L, output_scale)
        tensor_crop2 = transforms.functional.resized_crop(img_tensor, int(WIDTH_L/2-WIDTH_L/4), int(HEIGHT_L/2-HEIGHT_L/4), int(WIDTH_L/2), int(HEIGHT_L/2),output_scale)
        tensor_crop3 = transforms.functional.resized_crop(img_tensor, int(WIDTH_L/2-WIDTH_L/8), int(HEIGHT_L/2-HEIGHT_L/8), int(WIDTH_L/4), int(HEIGHT_L/4),output_scale)

        # Joint img together 
        #img_joint = torch.zeros((1,WIDTH_L,HEIGHT_L))
        #img_joint1 = transforms.functional.resize(tensor_crop1, (WIDTH_L, HEIGHT_L))
        #img_joint2 = transforms.functional.resize(tensor_crop2, (WIDTH_L/2, HEIGHT_L/2))
        
        #img_joint = img_joint1
        #img_joint[0,   WIDTH_L/4 : 3*WIDTH_L/4,   WIDTH_L/4 : 3*WIDTH_L/4] = img_joint2
        #img_joint[0, 3*WIDTH_L/8 : 5*WIDTH_L/8, 3*WIDTH_L/8 : 5*WIDTH_L/4] = tensor_crop3

        return torch.stack((tensor_crop1.float(), tensor_crop2.float(), tensor_crop3.float()))

    





    ####################
    # GETTER FUNCITONS #
    ####################

    def get_reconstruction(self, patch=False, cropscales=False):
        return self._get_reconstruction(patch=patch, cropscales=cropscales)

    def get_saliency_map(self, T=1, patch=False, cropscales=False):
        return self._get_saliency_map(T, patch, cropscales)

    def get_saliency_coord(self, T=1, cropscales=False):
        return self._get_max_saliency_coord(T, cropscales=cropscales)

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

    # loop over cropscales
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                 y[i,j,:,:] = np.exp(x[i,j,:,:]/T) / (np.exp(x[i,j,:,:]/T).sum())
    else: # loop over color channel
        for i in range(x.shape[0]):
            y[i,:,:] = np.exp(x[i,:,:]/T) / (np.exp(x[i,:,:]/T).sum())
    
    
    return y





            