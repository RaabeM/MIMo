import gym
import time
import mimoEnv
import argparse

import matplotlib.pyplot as plt
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore")


def test(env, test_for=1000, model=None):
    env.seed(42)
    obs = env.reset()
    for idx in range(test_for):

        print(f"\n>>> TIMESTEP {idx}")

        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs)

            #MR TODO code action given by the image of the left eye
            #move eye the max saliency 

        obs, _, done, _ = env.step(action)
        
        TEST = False
        if TEST:
            img_left  = obs['eye_left']
            img_right = obs['eye_right']
            
            # MR Get Saliency Map
            #saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            #(success, saliencyMap_left) = saliency.computeSaliency(img_left)
            #(success, saliencyMap_right) = saliency.computeSaliency(img_right)

            saliencyMap_left = env._get_saliency_map()
            saliencyMap_right= env._get_saliency_map()

            # MR: Search Max of sliency map
            max_saliency_left = np.where(saliencyMap_left == np.amax(saliencyMap_left))
            circle1 = plt.Circle( (max_saliency_left[1][0], max_saliency_left[0][0]), radius=3, color='r', fill=False)

            max_saliency_right = np.where(saliencyMap_right == np.amax(saliencyMap_right))
            circle2 = plt.Circle( (max_saliency_right[1][0], max_saliency_right[0][0]), radius=3, color='r', fill=False)

            #MR: Plot 

            line_h1 = plt.Line2D(xdata=np.linspace(0,255,256, dtype=int), ydata=np.repeat(128,256), color='r', linestyle='--', alpha=.3)
            line_v1 = plt.Line2D(ydata=np.linspace(0,255,256, dtype=int), xdata=np.repeat(128,256), color='r', linestyle='--', alpha=.3)

            line_h2 = plt.Line2D(xdata=np.linspace(0,255,256, dtype=int), ydata=np.repeat(128,256), color='r', linestyle='--', alpha=.3)
            line_v2 = plt.Line2D(ydata=np.linspace(0,255,256, dtype=int), xdata=np.repeat(128,256), color='r', linestyle='--', alpha=.3)

            
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
            plt.ion()
            plt.show()

            #ax1.cla()
            ax1.set_title("Left eye")
            ax1.imshow(img_left)
            ax1.imshow(saliencyMap_left, alpha=0.8)            
            ax1.add_patch(circle1)
            ax1.add_patch(line_h1)
            ax1.add_patch(line_v1)

            #ax2.cla()
            ax2.set_title("Right eye")
            ax2.imshow(img_right)
            ax2.imshow(saliencyMap_right, alpha=0.8)            
            ax2.add_patch(circle2)
            ax2.add_patch(line_h2)
            ax2.add_patch(line_v2)
            
            plt.draw()
            plt.pause(.001)   

            #plt.savefig(f"./plots/{idx}.png")

            input("Press any key to go to next timestep.")
            plt.close('all')
            
 
        else:   
            env.render() # MR opens the mujoco-env 

    env.reset()


def main():

    env = gym.make('MIMoBinocular-v0')
    _ = env.reset()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_for', default=0, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing of trained policy')               
    parser.add_argument('--save_every', default=100000, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--algorithm', default=None, type=str, 
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C', 'HER'],
                        help='RL algorithm from Stable Baselines3')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_model', default='', type=str,
                        help='Name of model to save')
    
    args = parser.parse_args()
    algorithm = args.algorithm
    load_model = args.load_model
    save_model = args.save_model
    save_every = args.save_every
    train_for = args.train_for
    test_for = args.test_for

    if algorithm == 'PPO':
        from stable_baselines3 import PPO as RL
    elif algorithm == 'SAC':
        from stable_baselines3 import SAC as RL
    elif algorithm == 'TD3':
        from stable_baselines3 import TD3 as RL
    elif algorithm == 'DDPG':
        from stable_baselines3 import DDPG as RL
    elif algorithm == 'A2C':
        from stable_baselines3 import A2C as RL

    # load pretrained model or create new one
    if algorithm is None:
        model = None
    elif load_model:
        model = RL.load("models/binocular" + load_model, env)
    else:
        model = RL("MultiInputPolicy", env, tensorboard_log="models/tensorboard_logs/", verbose=1, device='cuda') #TODO

    # train model
    counter = 0
    while train_for > 0:
        counter += 1
        train_for_iter = min(train_for, save_every)
        train_for = train_for - train_for_iter
        model.learn(total_timesteps=train_for_iter)
        model.save("models/reach" + save_model + "_" + str(counter))
    
    test(env, model=model, test_for=test_for)


if __name__ == '__main__':
    main()
