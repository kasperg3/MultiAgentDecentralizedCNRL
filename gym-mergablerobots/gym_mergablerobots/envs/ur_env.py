import numpy as np

from gym.envs.robotics import rotations, utils
from gym_mergablerobots.envs import  robot_env
from scipy.spatial.transform import Rotation
import mujoco_py


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

    def is_grasped(self, b_z):
        return (b_z > 0.025).astype(np.bool)

    def is_stacked(self,  object0, object1):
        # In the paper the threshold used for stacking is half the size of reaching, this will also be used here
        # The paper distinguish between the orientation, but in out case we use a symmetric object
        cube_threshold = [0.01, 0.01, 0.01]
        if abs(object0[0] - object1[0]) < cube_threshold[0] and \
           abs(object0[1] - object1[1]) < cube_threshold[1] and \
           abs(object0[2] - object1[2]) < cube_threshold[2]:
            return True
        else:
            return False


    def is_reached(self, object, gripper):
        cube_threshold = [0.020, 0.020, 0.020]
        if abs(object[0] - gripper[0]) < cube_threshold[0] and \
           abs(object[1] - gripper[1]) < cube_threshold[1] and \
           abs(object[2] - gripper[2]) < cube_threshold[2]:
            return True
        else:
            return False

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        reward = 0
        exploit_factor = 0.8
        slack = 0.7
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            reward = -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense':
            reward = -d
        elif self.reward_type == 'composite_reward':
            # Set the goal to follow object0
            self.goal = self.sim.data.get_site_xpos('object0') + [0, 0, 0.030]

            # Calculate the different metrics
            # b_z   : height of brick 1 above table
            # s_B1  : XYZ pos of the site located in the center of brick1
            # s_B2  : XYZ pos of the site located just over brick0
            # s_P   : XYZ pos of the pinch site in the gripper, the position where the fingers meet
            # w1    : weights used in the reaching reward function
            # w2    : weights used in the stack reward function
            s_B1 = self.sim.data.get_site_xpos('object1')  # The object to grasp
            s_B2 = self.goal  # The position where object1 will lay once stacked
            b_z = s_B1[2] - 0.414  # 0.4 is the height of the table(0.014 extra for inaccuracies in the sim)
            s_P = self.sim.data.get_site_xpos('robot0:grip')
            w1 = 2
            w2 = 2

            # if the blocks are stacked return the maximum reward
            if self.is_stacked(s_B1, s_B2):
                print("STACKED MOTHERFUCKER")
                return 1
            # If the blocks are not stacked but the block is grasped
            elif not self.is_stacked(s_B1, s_B2) and self.is_grasped(b_z):
                value = 1 - np.square(np.tanh(np.linalg.norm(s_B1*w2 - s_B2*2)))
                return 0.25 + 0.25*value
            # if the blocks are not stacked or grasped but is reached
            elif not(self.is_stacked(s_B1, s_B2) or self.is_grasped(b_z)) and self.is_reached(s_B1, s_P):
                return 0.125
            else:  # otherwise
                value = 1 - np.square(np.tanh(np.linalg.norm(w1 * s_B1 - 2*s_P)))
                return 0.125*value

        elif self.reward_type == 'dense_reward_shaping':
            R = -d
            start = self.initial_gripper_xpos[:3]
            # max_step_length = 0.01 * 20
            if self.action_counter == 0:
                self.phi_old = slack*((goal_distance(start, goal)) / exploit_factor)
                self.action_counter = 1

            phi_new = d / exploit_factor
            F = self.phi_old - phi_new
            self.phi_old = phi_new
            reward = R + F
        elif self.reward_type == 'sparse_reward_shaping':
            R = -(d > self.distance_threshold).astype(np.float32)
            start = self.initial_gripper_xpos[:3]
            init_object_pos = self.initial_object_xpos

            bestRoute = np.linalg.norm(start - init_object_pos) + np.linalg.norm(init_object_pos - goal)  # Euclidean distance
            t = bestRoute * exploit_factor  # Heuristic * exploit/(exploit+explore)
            N = 3  # Number of subgoals
            ns = 0  # Number of subgoals reached
            if np.linalg.norm(self.sim.data.get_site_xpos('robot0:grip') - self.sim.data.get_site_xpos('object0')) < self.distance_threshold:
                ns = 1
                if self.get_contact_points('object0', 'robot0:robotiq_finger_mesh_l') > 1 and \
                        self.get_contact_points('object0', 'robot0:robotiq_finger_mesh_r') > 1:       #contact between fingers and object
                    ns = 2
                    if np.abs(np.linalg.norm(self.sim.data.get_site_xpos('object0') - self.goal)) < self.distance_threshold:
                        ns = 3
            if not self.action_counter:
                self.phi_old = slack*(-((N - ns - 0.5) / N) * t)
                self.action_counter += 1
            phi_new = slack*(-((N - ns - 0.5) / N) * t)  # Remaining distance based on subgoals
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
        # Change action space if number of is changed
        assert action.shape == (8,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7]
        pos_ctrl *= 0.01  # limit maximum change in position
        rot_ctrl *= 0.01
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
            object0_pos = self.sim.data.get_site_xpos('object0')
            object1_pos = self.sim.data.get_site_xpos('object1')
            # rotations
            object0_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            object1_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object1'))
            # velocities
            object0_velp = self.sim.data.get_site_xvelp('object0') * dt
            object1_velp = self.sim.data.get_site_xvelp('object1') * dt
            # gripper state
            object0_rel_pos = object0_pos - grip_pos
            object0_velp -= grip_velp
            object1_rel_pos = object1_pos - grip_pos
            object1_velp -= grip_velp

            # relative potition from object1 to goal
            object1_rel_to_goal = object1_pos - self.goal
        else:
            object0_pos = object0_rot = object0_velp = object0_velr = object0_rel_pos = np.zeros(0)
            object1_pos = object1_rot = object1_velp = object1_velr = object1_rel_pos = np.zeros(0)
            object1_rel_to_goal = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object0_pos.copy())

        obs = np.concatenate([
            grip_pos,
            grip_rot,
            gripper_state,
            object0_pos.ravel(),
            object0_rel_pos.ravel(),
            object1_pos.ravel(),
            object1_rel_pos.ravel(),
            object1_rel_to_goal.ravel(),
            grip_velp,
            gripper_vel,
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

    def print_contact_points(self, object1=None, object2=None):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]

            # Print contact points between two specific geoms
            if object1 is not None and object2 is not None:
                if self.sim.model.geom_id2name(contact.geom1) == object1 or self.sim.model.geom_id2name(contact.geom1) == object2:
                    if self.sim.model.geom_id2name(contact.geom2) == object1 or self.sim.model.geom_id2name(contact.geom2) == object2:
                        print('contact', i)
                        print('dist', contact.dist)
                        print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
                        print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
            else:
                print('contact', i)
                print('dist', contact.dist)
                print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
                print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))

    def get_contact_points(self, object1=None, object2=None):
        contact_points = 0
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]

            # Print contact points between two specific geoms
            if object1 is not None and object2 is not None:
                if self.sim.model.geom_id2name(contact.geom1) == object1 or self.sim.model.geom_id2name(contact.geom1) == object2:
                    if self.sim.model.geom_id2name(contact.geom2) == object1 or self.sim.model.geom_id2name(contact.geom2) == object2:
                        contact_points = contact_points + 1

        return contact_points

    def _is_failed(self):
        return False

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.action_counter = 0
        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < self.distance_threshold:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                base_to_object_distance = np.linalg.norm(object_xpos[:2] - self.sim.data.get_site_xpos('robot0:base')[:2])
                # Make sure the object did not spawn in the robot base
                if base_to_object_distance < 0.20:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

            # Set object 0 initial position
            object0_qpos = self.sim.data.get_joint_qpos('object0:joint')
            object0_qpos[:2] = object_xpos

            # Place object 1 within object range of object0
            object1_qpos = self.sim.data.get_joint_qpos('object1:joint')
            object1_xpos = object_xpos + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object1_qpos[:2] = object1_xpos

            # Set object positions
            self.initial_object_xpos = object0_qpos[:3]
            self.sim.data.set_joint_qpos('object0:joint', object0_qpos)
            self.initial_object_xpos = object1_qpos[:3]
            self.sim.data.set_joint_qpos('object1:joint', object1_qpos)

            # open the grippers
            action = np.concatenate([[0, 0, 0], [0, 0, 0, 0], [-1, -1]])
            utils.ctrl_set_action(self.sim, action)
            utils.mocap_set_action(self.sim, action)

            # Place the end effector at the object every other episode
            if bool(np.random.binomial(1, 1)) and self.has_object:
                initial_grip = np.add(self.initial_object_xpos, [0, 0, 0.043])
                self.sim.data.set_mocap_quat('robot0:mocap', [0, 0, 1, 0])
                self.sim.data.set_mocap_pos('robot0:mocap', initial_grip)
                for _ in range(20):
                    self.sim.step()
                self.sim.data.set_mocap_quat('robot0:mocap', [0, 0, 1, 0])
                self.sim.data.set_mocap_pos('robot0:mocap', np.subtract(initial_grip, [0, 0, 0.03]))
                for _ in range(20):
                    self.sim.step()

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object and self.reward_type is not 'composite_reward':
            goal = self.sim.data.get_site_xpos('object0')
            # sample a point which is not close to the base
            while goal_distance(self.sim.data.get_site_xpos('object0'), goal) < self.distance_threshold:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)
        elif self.has_object and self.reward_type == 'composite_reward':
            goal = self.sim.data.get_site_xpos('object0') + [0, 0, 0.030]
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        result = (d < self.distance_threshold).astype(np.float32)
        if self.reward_type is 'composite_reward':
            result = np.all(((self.sim.data.get_site_xpos('object0') - 0.414) < 0.04))
        return result

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = rotations.mat2quat(self.sim.data.get_site_xmat('robot0:grip'))

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
