# MIMo

## The MIMo environment.

Base class. 

MuJoCo xmls.

The action space is auomatically generated from the underlying MuJoCo xml. Each actuator whose name starts with 'act:' is included in the action space. Each actuator has a range from -1 to 1, with full torque in opposite directions at -1 and 1 and a linear response in between.

The observation space is a dictionary built automatically based on the configuration of the sensor modules. An entry 'observation' is always included and always returns relative joint positions. Enabling more sensor modules adds extra entries. For example, each camera of the vision system will store its image in a separate entry in the observation space, named after the associated camera.

### Observation spaces and `done`

By default this environment follows the behaviour of the `Robot` environments in gym. This means that the `done` return value from `step` is always False, and the calling method has to figure out when to stop or reset. In addition the observation space includes two entries with the desired and the currently achieved goal (populated by `_sample_goal` and `_get_achieved_goal`).

This behaviour can be changed with two parameters during initialization of the environment. 
  1. `goals_in_observation` : If this parameter is False, the goal entries in the observation space will not be populated. Note that the space still contains these entries, but they will be size zero. By default set to True.
  2. `done_active` : If this parameter is True, `done` is True if either `_is_success` or `_is_failure` returns True. If set to False, `done` is always False. By default set to False. Note that this behaviour is defined in the `_is_done` function. If you overwrite this function you can ignore this parameter.

## Installation:

First install mujoco and mujoco-py following their instructions.
Then clone this repository, install other dependencies with `pip install -r requirements.txt` and finally run `pip install -e .`

## Sensor modules

All of the sensor modules follow the same pattern. They are initialized with a MuJoCo gym environment and a dictionary of parameters and their observations can be collected by calling their `get_<modality>_touch` function. The return of this function is generally a single array containing the flattened/concatenated output. Each module also has an attribute `sensor_outputs` that stores the unflattened outputs as a dictionary. The parameter structure and workings of each module are described in more detail below.

### Proprioception

### Touch

### Vestibular

### Vision

## Sample Environments

We provide several sample environments with some simple tasks for demonstration purposes. These come with both an openAI environment in `mimoEnv/envs` as well as simple training scripts using stable-baselines3, in `mimoEnv`. These environments include:

  1. `reach` - A stripped down version where MIMo is tasked with reaching for a ball hovering in front of him. By default only the proprioceptive sensors are used. MIMo can only move his right arm and his head is manually fixed to track the ball. The initial position of both the ball and MIMo is slightly randomized.
  2. `standup` - MIMo is tasked with standing up. At the start he is in a low crouch with his hands gripping the bars of a crib. Proprioception and the vestibular sensors are included by default.
  3. `test` - This is a simple dummy environment set up to demonstrate and visualize most of the sensor modalities. MIMo is set to fall from a short height. During this, the visual and haptic outputs are rendered and saved to hard drive.
