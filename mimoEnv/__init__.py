import gym

def register(id, entry_point, max_episode_steps=200, force=True):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps, 
    )

register(id='MIMoBench-v0',
         entry_point='mimoEnv.envs:MIMoDummyEnv',
         max_episode_steps=6000,
         )

register(id='MIMoShowroom-v0',
         entry_point='mimoEnv.envs:MIMoShowroomEnv',
         max_episode_steps=500,
         )

register(id='MIMoReach-v0',
         entry_point='mimoEnv.envs:MIMoReachEnv',
         max_episode_steps=1000,
         )

register(id='MIMoStandup-v0',
         entry_point='mimoEnv.envs:MIMoStandupEnv',
         max_episode_steps=500, 
         )

register(id='MIMoSaccades-v0',
         entry_point='mimoEnv.envs:MIMoSaccadesEnv',
         max_episode_steps=500, 
         )

register(id='MIMoSelfBody-v0',
         entry_point='mimoEnv.envs:MIMoSelfBodyEnv',
         max_episode_steps=500, 
         )

register(id='MIMoEyeHand-v0',
         entry_point='mimoEnv.envs:MIMoEyeHandEnv',
         max_episode_steps=500, 
         )

register(id='MIMoBinocular-v0',
         entry_point='mimoEnv.envs:MIMoBinocularEnv',
         max_episode_steps=10, 
         )

register(id='MIMoBinoc_Saccade-v0',
         entry_point='mimoEnv.envs:MIMoBinoc_SaccadeEnv',
         max_episode_steps=10, 
         )