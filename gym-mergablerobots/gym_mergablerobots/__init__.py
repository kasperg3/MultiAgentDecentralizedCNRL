from gym.envs.registration import register

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='UrPickAndPlace{}-v0'.format(suffix),
        entry_point='gym_mergablerobots.envs:URPickAndPlaceEnv',
    )

    register(
        id='UrReach{}-v0'.format(suffix),
        entry_point='gym_mergablerobots.envs:UrReachEnv',
        kwargs=kwargs,
        max_episode_steps=70,
    )
