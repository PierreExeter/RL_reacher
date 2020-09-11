from gym.envs.registration import register

register(
	id='Reacher1Dof-v0',
	entry_point='gym_envs.envs.reacher_env:ReacherBulletEnv1',
	max_episode_steps=150,
	reward_threshold=18.0,
	)

register(
	id='Reacher2Dof-v0',
	entry_point='gym_envs.envs.reacher_env:ReacherBulletEnv2',
	max_episode_steps=150,
	reward_threshold=18.0,
	)

register(
	id='Reacher3Dof-v0',
	entry_point='gym_envs.envs.reacher_env:ReacherBulletEnv3',
	max_episode_steps=150,
	reward_threshold=18.0,
	)

register(
	id='Reacher4Dof-v0',
	entry_point='gym_envs.envs.reacher_env:ReacherBulletEnv4',
	max_episode_steps=150,
	reward_threshold=18.0,
	)

register(
	id='Reacher5Dof-v0',
	entry_point='gym_envs.envs.reacher_env:ReacherBulletEnv5',
	max_episode_steps=150,
	reward_threshold=18.0,
	)

register(
	id='Reacher6Dof-v0',
	entry_point='gym_envs.envs.reacher_env:ReacherBulletEnv6',
	max_episode_steps=150,
	reward_threshold=18.0,
	)
