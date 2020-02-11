from gym.envs.registration import register

register(
    id='mergablerobots-v0',
    entry_point='gym_mergablerobots.envs:URPickAndPlaceEnv',
)
