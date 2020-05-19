from gym.envs.registration import register


register(
    id='Concept-v0',
    entry_point='gym_mergablerobots.envs:Concept',
    max_episode_steps=10,
)

register(
    id='Concept-dual-asynchronous-v0',
    entry_point='gym_mergablerobots.envs:Concept',
    max_episode_steps=20,
)

register(
    id='Concept-dual-synchronous-v0',
    entry_point='gym_mergablerobots.envs:Concept',
    max_episode_steps=20,
)



for reward_type in ['sparse', 'dense', 'place', 'orient', 'lift', 'reach', 'composite']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    episode_steps = 70  # The default episode step
    if reward_type == 'place':
        episode_steps = 150 # Extra time for placing the object
        suffix = 'Place'
    elif reward_type == 'lift':
        episode_steps = 50
        suffix = 'Lift'
    elif reward_type == 'orient':
        episode_steps = 150
        suffix = 'Orient'
    elif reward_type == 'reach':
        episode_steps = 70
        suffix = 'Reach'
    elif reward_type == 'composite':
        episode_steps = 200
        suffix = 'Composite'
    elif reward_type == 'dense':
        episode_steps = 70
    elif reward_type == 'sparse':
        episode_steps = 150

    kwargs = {
        'reward_type': reward_type,
    }
    if reward_type == 'sparse' or reward_type == 'dense' or reward_type == 'composite_reward':
        register(
            id='UrPickAndPlace{}-v0'.format(suffix),
            entry_point='gym_mergablerobots.envs:URPickAndPlaceEnv',
            kwargs=kwargs,
            max_episode_steps=episode_steps,
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
