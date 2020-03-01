import numpy as np

from gym.envs.robotics import rotations, utils, robot_env
from scipy.spatial.transform import Rotation


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class UrEnv(robot_env.RobotEnv):
    """Superclass for UR environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        super(UrEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=8,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        reward = 0
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            reward = -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense':
            reward = -d
        elif self.reward_type == 'dense_reward_shaping':

            R = -d
            start = self.initial_gripper_xpos[:3]
            exploit_factor = 1
            #max_step_length = 0.01 * 20

            if self.action_counter == 0:
                self.phi_old = (goal_distance(start, goal)) / exploit_factor
                self.action_counter = 1

            phi_new = d / exploit_factor
            F = self.phi_old - phi_new
            self.phi_old = phi_new
            reward = R + F

        elif self.reward_type == 'sparse_reward_shaping':
            R = -(d > self.distance_threshold).astype(np.float32)
            start = self.initial_gripper_xpos[:3]
            init_object_pos = self.initial_object_xpos

            bestRoute = np.abs(np.linalg.norm(start - init_object_pos)) + np.abs(
                np.linalg.norm(init_object_pos - goal))  # Euclidean distance

            exploit_factor = 0.5
            t = bestRoute * exploit_factor  # Heuristic * exploit/(exploit+explore)
            N = 3  # Number of subgoals
            ns = 0  # Number of subgoals reached
            if not self.action_counter:
                self.phi_old = -((N - ns - 0.5) / N) * t
            self.action_counter += 1
            phi_new = -((N - ns - 0.5) / N) * t  # Remaining distance based on subgoals
            # (0.5 is because you on average are halfway between each subgoal)
            F = phi_new - self.phi_old  # add gamma
            self.phi_old = phi_new
            reward = R + F
        return reward

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:joint7_l', 0.)
            self.sim.data.set_joint_qpos('robot0:joint7_r', 0.)
            self.sim.forward()

    def _set_action(self, action):
        #Change action space if number of is changed
        assert action.shape == (8,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7]
        pos_ctrl *= 0.01  # limit maximum change in position
        rot_ctrl *= 0.1
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        grip_rot_cart = Rotation.from_matrix(self.sim.data.get_site_xmat('robot0:grip'))# The rotation of gripper(Cartesian)
        grip_rot = Rotation.as_quat(grip_rot_cart)
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            grip_pos, grip_rot, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:robotiq_base_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.action_counter = 0
        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < self.distance_threshold:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.initial_object_xpos = object_qpos[:3]
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

            # # Open the gripper fingers
            # self.sim.data.set_joint_qpos('robot0:joint7_l', 0.1)
            # self.sim.data.set_joint_qpos('robot0:joint7_r', 0.1)
            # for _ in range(10):
            #     self.sim.step()

            # Place the end effector at the object every other episode
            if bool(np.random.binomial(1, 1)) and self.has_object:
                initial_grip = np.add(self.initial_object_xpos, [0, 0, 0.045])
                self.sim.data.set_mocap_pos('robot0:mocap', initial_grip)
                self.sim.data.set_mocap_quat('robot0:mocap', [0, 0, 1, 0])
                for _ in range(10):
                    self.sim.step()

                # initial_grip = np.add(self.initial_object_xpos, [0, 0, 0.01])
                # self.sim.data.set_mocap_pos('robot0:mocap', initial_grip)
                # self.sim.data.set_mocap_quat('robot0:mocap', [0, 0, 1, 0])
                # for _ in range(10):
                #     self.sim.step()

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.sim.data.get_site_xpos('object0')
            while goal_distance(self.sim.data.get_site_xpos('object0'), goal) < self.distance_threshold:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                              size=3)
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = rotations.mat2quat(self.sim.data.get_site_xmat('robot0:grip'))

        # Place the end effector at the object every other episode
        if bool(np.random.binomial(1, 0.5)) and self.has_object:
            gripper_target = self.sim.data.get_site_xpos('object0')

        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(UrEnv, self).render(mode)

