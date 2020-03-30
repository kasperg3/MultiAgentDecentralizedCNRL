from gym.envs.registration import register

for reward_type in ['sparse', 'dense', 'place', 'orient', 'lift', 'reach']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    episode_steps = 70  # The default episode step
    if reward_type == 'place':
        episode_steps = 150 # Extra time for placing the object
        suffix = 'Place'
    elif reward_type == 'lift':
        suffix = 'Lift'
    elif reward_type == 'orient':
        suffix = 'Orient'
    elif reward_type == 'reach':
        suffix = 'Reach'


    kwargs = {
        'reward_type': reward_type,
    }
    if reward_type == 'sparse' or reward_type == 'dense':
        register(
            id='UrPickAndPlace{}-v0'.format(suffix),
            entry_point='gym_mergablerobots.envs:URPickAndPlaceEnv',
            kwargs=kwargs,
            max_episode_steps=150,
        )

        register(
            id='UrReach{}-v0'.format(suffix),
            entry_point='gym_mergablerobots.envs:UrReachEnv',
            kwargs=kwargs,
            max_episode_steps=70,
        )

    register(
        id='UrBinPicking{}-v0'.format(suffix),
        entry_point='gym_mergablerobots.envs:UrBinPicking',
        kwargs=kwargs,
        max_episode_steps=episode_steps,
    )
