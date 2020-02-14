from gym.envs.registration import register

register(
    id='mergablerobots-v0',
    entry_point='gym_mergablerobots.envs:URPickAndPlaceEnv',
)

register(
    id='UrReach-v0',
    entry_point='gym_mergablerobots.envs:UrReachEnv',
)
