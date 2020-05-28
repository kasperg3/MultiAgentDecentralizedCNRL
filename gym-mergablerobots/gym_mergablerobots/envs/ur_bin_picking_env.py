import random

import numpy as np
import math
from gym.envs.robotics import rotations, utils
from gym_mergablerobots.envs import robot_env
from scipy.spatial.transform import Rotation
import mujoco_py


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


class UrBinPickingEnv(robot_env.RobotEnv):
    """Superclass for UR environments.
    """

    def __init__(self,
                 model_path,
                 n_substeps,
                 initial_qpos,
                 reward_type,
                 box_range,
                 success_threshold,
                 lift_threshold):

        """Initializes a new Ur environment.

        Args: model_path (string): path to the environments XML file n_substeps (int): number of substeps the
        simulation runs on every call to step initial_qpos (dict): a dictionary of joint names and values that define
        the initial configuration reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        success_threshold: used for reach, which describes how close to the goal it is to return success
        lift_threshold: used for lift, and determines how high the object should be lifted from the box in order to
            return success
        """
        self.episode_steps = 0
        self.reward_type = reward_type
        self.box_range = box_range
        self.success_threshold = success_threshold
        self.lift_threshold = lift_threshold

        if self.reward_type == 'place':
            n_actions = 4
        elif self.reward_type == 'composite':
            n_actions = 8
        else:
            n_actions = 7

        super(UrBinPickingEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def is_grasped(self, b_z):
        return (b_z > 0.025).astype(np.bool)


    def is_stacked(self,  object0, object1):
        # In the paper the threshold used for stacking is half the size of reaching, this will also be used here
        # The paper distinguish between the orientation, but in out case we use a symmetric object
        cube_threshold = [0.01, 0.01, 0.02]
        if abs(object0[0] - object1[0]) < cube_threshold[0] and \
           abs(object0[1] - object1[1]) < cube_threshold[1] and \
           abs(object0[2] - object1[2]) < cube_threshold[2]:
            return True
        else:
            return False


    def is_reached(self, object, gripper):
        cube_threshold = [0.01, 0.01, 0.01]
        if abs(object[0] - gripper[0]) < cube_threshold[0] and \
           abs(object[1] - gripper[1]) < cube_threshold[1] and \
           abs(object[2] - gripper[2]) < cube_threshold[2]:
            return True
        else:
            return False

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        reward = 0
        if self.reward_type == 'reach':
            #d = goal_distance(achieved_goal, goal)
            #penalty_weight = 1
            #box_move_penalty = np.abs(np.linalg.norm(self.sim.data.get_site_xvelp('box')) * penalty_weight)
            reward = float(-(goal_distance(achieved_goal, goal)))

        elif self.reward_type == 'orient':
            # #angular component
            w_d = 0.5
            w_theta = 0.5
            alpha = 0.4

            body_id1 = self.sim.model.body_name2id('robot0:left_finger')
            body_id2 = self.sim.model.body_name2id('robot0:right_finger')

            finger_left = self.sim.data.body_xpos[body_id1]
            finger_right = self.sim.data.body_xpos[body_id2]

            orient_line = finger_left - finger_right
            rot_object0 = self.sim.data.get_site_xmat('object0')
            theta_x = min(angle_between(orient_line, np.matmul(rot_object0, (1, 0, 0))), angle_between(orient_line, np.matmul(rot_object0, (-1, 0, 0))))
            theta_y = min(angle_between(orient_line, np.matmul(rot_object0, (0, 1, 0))), angle_between(orient_line, np.matmul(rot_object0, (0, -1, 0))))
            theta_z = angle_between(orient_line, np.matmul(rot_object0, (0, 0, 1)))
            angle_45 = math.radians(45)
            angle_90 = math.radians(90)

            theta_x = angle_45 - np.abs(theta_x - angle_45)
            theta_y = angle_45 - np.abs(theta_y - angle_45)
            theta_z = angle_90 - np.abs(theta_z - angle_90)

            if theta_x < theta_y:
                r_theta = -(1 - (np.clip((0.5 * ((1 - (theta_x / angle_45)) + theta_z / angle_90)), 0, 1)) ** alpha)
            else:
                r_theta = -(1 - (np.clip((0.5 * ((1 - (theta_y / angle_45)) + theta_z / angle_90)), 0, 1)) ** alpha)

            if self._is_success(achieved_goal, goal):
                reward = 1
            else:
                r_d = -(np.clip((goal_distance(achieved_goal, goal) / self.initial_goal_distance), 0, 1) ** alpha)
                reward = (w_theta * r_theta + w_d * r_d)

        elif self.reward_type == 'lift':
            h = np.abs(goal - achieved_goal)
            h_max = self.lift_threshold
            alpha = 4
            r_h = -(np.clip((h / h_max), 0, 1) ** alpha)[0]
            bonus_reward = 1

            if self._is_success(achieved_goal, goal):    # Is within 1 cm of the goal height
                reward = bonus_reward
            else:
                reward = r_h

        elif self.reward_type == 'place':
            # Positional difference
            dist = goal_distance(achieved_goal[:3], goal[:3])
            dist_ref = goal_distance(self.initial_box_xpos, goal[:3])
            position_score = -np.clip((dist / dist_ref), 0, 1)
            w_d = 2
            w_theta = 0.5
            bonus = 1
            alpha = 0.4
            goal_rot = rotations.quat2mat(goal[3:])
            goal_y = np.matmul(goal_rot, (0, 1, 0))
            goal_x = np.matmul(goal_rot, (1, 0, 0))

            rot_object0 = self.sim.data.get_site_xmat('object0')
            theta_x = min(angle_between(goal_x, np.matmul(rot_object0, (1, 0, 0))),
                          angle_between(goal_x, np.matmul(rot_object0, (-1, 0, 0))))
            theta_y = min(angle_between(goal_y, np.matmul(rot_object0, (0, 1, 0))),
                          angle_between(goal_y, np.matmul(rot_object0, (0, -1, 0))))
            angle_45 = math.radians(45)
            theta_x = angle_45 - np.abs(theta_x - angle_45)
            theta_y = angle_45 - np.abs(theta_y - angle_45)

            if theta_x < theta_y:
                r_theta = -(1 - (np.clip((1 - (theta_x / angle_45)), 0, 1)) ** alpha)
            else:
                r_theta = -(1 - (np.clip((1 - (theta_y / angle_45)), 0, 1)) ** alpha)

            # Calculate move box penalty
            dt = self.sim.nsubsteps * self.sim.model.opt.timestep
            box_velp = self.sim.data.get_site_xvelp('box') * dt
            penalty_weight = 0
            velp_pen = -(np.linalg.norm(box_velp)*penalty_weight)

            if self._is_success(achieved_goal, goal):
                reward = bonus
            else:
                reward = r_theta * w_theta + position_score * w_d + velp_pen

        elif self.reward_type == 'composite':
            # Set the goal to follow object0
            self.goal = self.sim.data.get_site_xpos('object0') + [0, 0, 0.030]

            # Calculate the different metrics
            # b_z   : height of brick 1 above table
            # s_B1  : XYZ pos of the site located in the center of brick1
            # s_B2  : XYZ pos of the site located just over brick0
            # s_P   : XYZ pos of the pinch site in the gripper, the position where the fingers meet
            # w1    : weights used in the reaching reward function
            # w2    : weights used in the stack reward function
            s_B1 = self.sim.data.get_site_xpos('object1').copy()  # The object to grasp
            s_B2 = self.goal.copy()  # The position where object1 will lay once stacked
            b_z = s_B1[2] - 0.414  # 0.4 is the height of the table(0.014 extra for inaccuracies in the sim)
            s_P = self.sim.data.get_site_xpos('robot0:grip').copy()

            # if the blocks are stacked return the maximum reward
            if self.is_stacked(s_B1, s_B2):
                return 1
            # If the blocks are not stacked but the block is grasped
            elif not self.is_stacked(s_B1, s_B2) and self.is_grasped(b_z):
                value = 1 - np.square(np.tanh(15*np.linalg.norm(s_B1 - s_B2)))
                return 0.25 + 0.25 * value
            # if the blocks are not stacked or grasped but is reached
            elif not (self.is_stacked(s_B1, s_B2) or self.is_grasped(b_z)) and self.is_reached(s_B1, s_P):
                return 0.125
            elif self.gripper_open():  # otherwise
                value = 1 - np.square(np.tanh(15*np.linalg.norm(s_B1 - s_P)))
                return 0.125 * value
            else:
                return 0
        return reward

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        # Lock gripper open in reach
        if self.reward_type == 'reach' or self.reward_type == 'orient':  # Keep the gripper open
            self.sim.data.set_joint_qpos('robot0:joint7_l', -0.008)
            self.sim.data.set_joint_qpos('robot0:joint7_r', -0.008)
            self.sim.forward()
        self.episode_steps += 1

    def gripper_open(self):
        return np.bool(self.sim.data.ctrl[0] <= 0)

    def _set_action(self, action):
        # Change action space if number of is changed
        action = action.copy()  # ensure that we don't change the action outside of this scope

        if action.shape == (7,):
            pos_ctrl, rot_ctrl = action[:3], action[3:7]
            gripper_ctrl = np.array([0, 0])
        elif action.shape == (4,):
            pos_ctrl, rot_ctrl = action[:3], action[3]
            gripper_ctrl = np.array([0, 0])
        elif action.shape == (8,):
            pos_ctrl, rot_ctrl, g_ctrl = action[:3], action[3:7], action[7]
            gripper_ctrl = np.array([g_ctrl, g_ctrl]) * 0.3 # Limit gripper ctrl to 0.3
        else:
            raise Exception('Invalid action space for ' + self.reward_type)

        assert gripper_ctrl.shape == (2,)
        if self.reward_type == 'reach' or self.reward_type == 'orient':
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        elif self.reward_type == 'lift':
            gripper_ctrl = np.ones_like(gripper_ctrl)*0.3
        elif self.reward_type == 'place':
            gripper_ctrl = np.ones_like(gripper_ctrl) * 0.3
            rot_ctrl = (rotations.euler2quat([0, np.pi, rot_ctrl * 2 * np.pi]) * rotations.quat_conjugate(rotations.mat2quat(self.sim.data.get_site_xmat('robot0:grip'))))*2

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl *= 0.01
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        grip_rot_cart = Rotation.from_matrix(
            self.sim.data.get_site_xmat('robot0:grip'))  # The rotation of gripper(Cartesian)
        grip_rot = Rotation.as_quat(grip_rot_cart)
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        grip_velr = self.sim.data.get_site_xvelr('robot0:grip') * dt
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        box_pos = self.sim.data.get_site_xpos('box')
        box_rel_pos = box_pos - grip_pos
        box_rot = Rotation.from_matrix(self.sim.data.get_site_xmat('box')).as_quat()

        object_pos = self.sim.data.get_site_xpos('object0')
        object_rel_pos = object_pos - grip_pos
        object_rot = Rotation.from_matrix(self.sim.data.get_site_xmat('object0')).as_quat()

        achieved_goal = grip_pos.copy()

        obs = None
        if self.reward_type == 'reach':
            # The relative position to the goal
            goal_rel_pos = self.goal - achieved_goal
            # Update the goal to follow the box
            self.goal = self.sim.data.get_site_xpos('box')[:3] + self.box_offset
            obs = np.concatenate([
                grip_pos,
                grip_rot,
                grip_velp,
                grip_velr,
            #   box_pos.ravel(),
                box_rel_pos.ravel(),
            #   box_rot.ravel(),
                goal_rel_pos,
            ])
        elif self.reward_type == 'orient':
            # The relative position to the goal
            goal_rel_pos = self.goal - achieved_goal
            # Update the goal to follow the box
            self.goal = self.sim.data.get_site_xpos('object0')[:3]
            obs = np.concatenate([
                grip_pos,
                grip_rot,
                grip_velp,
                grip_velr,
                #object_pos.ravel(),
                #object_rel_pos.ravel(),
                #object_rot.ravel(),
                box_pos.ravel(),
                box_rel_pos.ravel(),
                box_rot.ravel(),
                goal_rel_pos,
            ])
        elif self.reward_type == 'lift':
            # The goal should be static and is x height from the box
            # The relative position is between object and goal position
            object_height = self.sim.data.get_site_xpos('object0')[2]
            goal_rel_height = self.goal - object_height
            # Override achieved_goal to fit with the one dimentional goal of lift(the height of the object)
            achieved_goal = object_height

            obs = np.concatenate([
                grip_pos,
                #grip_rot,
                grip_velp,
                grip_velr,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                #box_pos.ravel(),
                box_rel_pos.ravel(),
                #box_rot.ravel(),
                goal_rel_height,
            ])
        elif self.reward_type == 'place':
            # The relative position to the goal
            #achieved_goal = np.concatenate((grip_pos, grip_rot))
            achieved_goal = np.concatenate((object_pos, object_rot.ravel()))
            goal_rel_pos = self.goal[:3] - achieved_goal[:3]
            grip_q = rotations.mat2quat(self.sim.data.get_site_xmat('robot0:grip'))
            goal_q = self.goal[3:]

            goal_rel_rot = np.array(2 * np.arccos(np.abs(np.inner(grip_q, rotations.quat_conjugate(goal_q)))))

            obs = np.concatenate([
                #grip_pos,
                #grip_rot,
                object_pos,
                object_rot.ravel(),
                object_velp,
                object_velr,
                #object_rel_pos,
                #box_pos.ravel(),
                box_rel_pos.ravel(),
                goal_rel_pos,
                goal_rel_rot.ravel(),
            ])
        elif self.reward_type == 'composite':
            # The relative position to the goal
            # achieved_goal = np.concatenate((grip_pos, grip_rot))
            achieved_goal = np.concatenate((object_pos, object_rot.ravel()))
            goal_rel_pos = self.goal[:3] - achieved_goal[:3]
            grip_q = rotations.mat2quat(self.sim.data.get_site_xmat('robot0:grip'))
            goal_q = self.goal[3:]
            # Set the goal position over the other object
            self.goal = self.sim.data.get_site_xpos('object0') + [0, 0, 0.030]
            obs = np.concatenate([
                [self.gripper_open()],
                grip_pos,
                grip_rot,
                object_pos,
                object_rot.ravel(),
                object_velp,
                object_velr,
                object_rel_pos,
                goal_rel_pos,
            ])
        else:
            raise Exception('Invalid reward type:' + self.reward_type + ' \n use either: reach, orient, lift')

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
        if self.reward_type == 'lift':
            # Lift only has a height goal, so we set the visualization target to the box xy pos and the goal height
            visualized_goal = self.sim.data.get_site_xpos('box').copy()
            visualized_goal[2] = self.goal
            self.sim.model.site_pos[site_id] = visualized_goal - sites_offset[0]
        elif self.reward_type == 'place':
            # Place also has a orientation as goal
            self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
            self.sim.model.site_quat[site_id] = self.goal[3:]
        elif self.reward_type == 'composite':
            self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        else:
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def print_contact_points(self, object1=None, object2=None):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]

            # Print contact points between two specific geoms
            if object1 is not None and object2 is not None:
                if self.sim.model.geom_id2name(contact.geom1) == object1 or self.sim.model.geom_id2name(
                        contact.geom1) == object2:
                    if self.sim.model.geom_id2name(contact.geom2) == object1 or self.sim.model.geom_id2name(
                            contact.geom2) == object2:
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
                if self.sim.model.geom_id2name(contact.geom1) == object1 or self.sim.model.geom_id2name(
                        contact.geom1) == object2:
                    if self.sim.model.geom_id2name(contact.geom2) == object1 or self.sim.model.geom_id2name(
                            contact.geom2) == object2:
                        contact_points = contact_points + 1

        return contact_points

    def sample_box_position(self):
        box_xpos = self.initial_box_xpos[:2] + self.np_random.uniform(-self.box_range, self.box_range, size=2)
        self.box_qpos = self.sim.data.get_joint_qpos('box:joint')
        assert self.box_qpos.shape == (7,)
        self.box_qpos[:2] = box_xpos
        # Set box position
        self.initial_box_xpos = self.box_qpos[:3]
        self.sim.data.set_joint_qpos('box:joint', self.box_qpos)

    def sample_point(self, x_range, y_range, z_range):
        return [
            float(self.np_random.uniform(-x_range, x_range)),
            float(self.np_random.uniform(-y_range, y_range)),
            float(self.np_random.uniform(-z_range, z_range))]

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.episode_steps = 0
        # Randomize start position of object.
        if self.reward_type == 'reach':
            self.sample_box_position()
        elif self.reward_type == 'orient':
            # Start the simulation just over the box(same space as reach goal)
            target_y_range = 0.05  # The length of the box
            target_x_range = 0.1  # The width of the box
            target_height = 0.05  # The height of the box

            # sample a positional init within the box
            gripper_offset = [
                float(self.np_random.uniform(-target_x_range, target_x_range)),
                float(self.np_random.uniform(-target_y_range, target_y_range)),
                target_height]

            # Randomize box
            self.sample_box_position()

            # random object spawn (z-rot and xyz)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            object_offset = [
                float(self.np_random.uniform(-target_x_range, target_x_range)),
                float(self.np_random.uniform(-target_y_range, target_y_range)),
                target_height]
            object_z_angle = float(self.np_random.uniform(-np.pi, np.pi))
            object_qpos_euler = rotations.quat2euler(object_qpos[3:])
            object_qpos_euler[2] = object_z_angle
            object_qpos[3:] = rotations.euler2quat(object_qpos_euler)
            object_qpos[:3] = self.box_qpos[:3] + object_offset

            # for test or perfect alignment
            quat_test_grip = np.array([0, 0, 1, 0])
            euler_test_grip = rotations.quat2euler(quat_test_grip)
            euler_test_grip[2] = -object_z_angle  # Negative because the z-axis is opposite
            quat_test_grip = rotations.euler2quat(euler_test_grip)

            # sample the initial gripper rot
            rot_grip = Rotation.random().as_matrix()
            rot_box = self.sim.data.get_site_xmat('box')
            rot_gb = np.matmul(rot_grip.transpose(), rot_box)
            p_box = np.array([0, 0, -1])
            p_g = np.matmul(rot_gb, p_box)
            alpha_z = np.arccos(p_g[2] / np.sqrt(p_g[0] ** 2 + p_g[1] ** 2 + p_g[2] ** 2))

            while math.degrees(alpha_z) > 10:
                rot_grip = Rotation.random().as_matrix()
                rot_gb = np.matmul(rot_grip.transpose(), rot_box)
                p_g = np.matmul(rot_gb, p_box)
                alpha_z = np.arccos(p_g[2] / np.sqrt(p_g[0] ** 2 + p_g[1] ** 2 + p_g[2] ** 2))

            # rotate gripper into position
            # move overbox
            overbox_pos = self.box_qpos[:3] + gripper_offset
            overbox_pos[2] = overbox_pos[2] + 0.1
            self.sim.data.set_mocap_pos('robot0:mocap', overbox_pos)  # This is the real one
            self.sim.data.set_mocap_quat('robot0:mocap', rotations.mat2quat(rot_grip))
            grip_test_pos = object_qpos[:3]
            # self.sim.data.set_mocap_pos('robot0:mocap', grip_test_pos)                #this is test or perfect alignment
            # self.sim.data.set_mocap_quat('robot0:mocap', quat_test_grip)
            for _ in range(10):
                self.sim.step()
            # go down to box 1
            self.sim.data.set_mocap_pos('robot0:mocap', self.box_qpos[:3] + gripper_offset)  # real
            # self.sim.data.set_mocap_pos('robot0:mocap', object_qpos[:3])              #test
            for _ in range(10):
                self.sim.step()
            self.initial_goal_distance = goal_distance(overbox_pos, object_qpos[:3])

        elif self.reward_type == 'lift':
            self.sample_box_position()
            self.sim.data.get_site_xpos('object0')
            # Start the simulation with the object between the fingers
            # sample a position within the box
            object_offset = self.sample_point(0.1, 0.05, 0)
            box_qpos = self.sim.data.get_joint_qpos('box:joint')
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            object_qpos[:3] = box_qpos[:3] + object_offset
            self.initial_object_pos = object_qpos[:3].copy()
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

            # Start the simulation just over the box(same space as reach goal)
            target_y_range = 0.05  # The length of the box
            target_x_range = 0.1  # The width of the box
            target_height = 0.05  # The height of the box

            # sample a positional init within the box
            gripper_offset = [
                float(self.np_random.uniform(-target_x_range, target_x_range)),
                float(self.np_random.uniform(-target_y_range, target_y_range)),
                target_height]

            # random object spawn (z-rot and xyz)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            object_z_angle = float(self.np_random.uniform(-np.pi, np.pi))
            object_qpos_euler = rotations.quat2euler(object_qpos[3:])
            object_qpos_euler[2] = object_z_angle
            object_qpos[3:] = rotations.euler2quat(object_qpos_euler)
            object_qpos[:3] = self.box_qpos[:3] + object_offset

            # for test or perfect alignment
            quat_test_grip = np.array([0, 0, 1, 0])
            euler_test_grip = rotations.quat2euler(quat_test_grip)
            euler_test_grip[2] = -object_z_angle  # Negative because the z-axis is opposite
            quat_test_grip = rotations.euler2quat(euler_test_grip)

            # rotate gripper into position
            # move overbox
            overbox_pos = self.box_qpos[:3] + gripper_offset
            overbox_pos[2] = overbox_pos[2] + 0.1
            grip_test_pos = object_qpos[:3] + [0, 0, 0.1]
            self.sim.data.set_mocap_pos('robot0:mocap', grip_test_pos)   #this is test or perfect alignment
            self.sim.data.set_mocap_quat('robot0:mocap', quat_test_grip)
            for _ in range(10):
                self.sim.data.set_joint_qpos('robot0:joint7_l', -0.0)
                self.sim.data.set_joint_qpos('robot0:joint7_r', -0.0)
                self.sim.forward()
                #self.render()
                self.sim.step()
            self.sim.data.set_mocap_pos('robot0:mocap', object_qpos[:3])
            for _ in range(10):
                self.sim.data.set_joint_qpos('robot0:joint7_l', -0.0)
                self.sim.data.set_joint_qpos('robot0:joint7_r', -0.0)
                self.sim.forward()
                #self.render()
                self.sim.step()
        elif self.reward_type == 'place':
            self.sample_box_position()
            self.sim.data.get_site_xpos('object0')
            # Start the simulation with the object between the fingers
            # sample a position within the box
            object_offset = self.sample_point(0.1, 0.05, 0)
            box_qpos = self.sim.data.get_joint_qpos('box:joint')
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            object_qpos[:3] = box_qpos[:3] + object_offset
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

            # Start the simulation just over the box(same space as reach goal)
            target_y_range = 0.05  # The length of the box
            target_x_range = 0.1  # The width of the box
            target_height = 0.05  # The height of the box

            # sample a positional init within the box
            gripper_offset = [
                float(self.np_random.uniform(-target_x_range, target_x_range)),
                float(self.np_random.uniform(-target_y_range, target_y_range)),
                target_height]

            # random object spawn (z-rot and xyz)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            object_z_angle = float(self.np_random.uniform(-np.pi, np.pi))
            object_qpos_euler = rotations.quat2euler(object_qpos[3:])
            object_qpos_euler[2] = object_z_angle
            object_qpos[3:] = rotations.euler2quat(object_qpos_euler)
            object_qpos[:3] = self.box_qpos[:3] + object_offset

            # for test or perfect alignment
            quat_test_grip = np.array([0, 0, 1, 0])
            euler_test_grip = rotations.quat2euler(quat_test_grip)
            euler_test_grip[2] = -object_z_angle  # Negative because the z-axis is opposite
            quat_test_grip = rotations.euler2quat(euler_test_grip)

            # rotate gripper into position
            # move overbox
            overbox_pos = self.box_qpos[:3] + gripper_offset
            overbox_pos[2] = overbox_pos[2] + 0.1
            grip_test_pos = object_qpos[:3] + [0, 0, 0.1]
            self.sim.data.set_mocap_pos('robot0:mocap', grip_test_pos)  # this is test or perfect alignment
            self.sim.data.set_mocap_quat('robot0:mocap', quat_test_grip)

            action = np.concatenate([[0, 0, 0], [0, 0, 0, 0], [0, 0]])
            utils.ctrl_set_action(self.sim, action)
            for _ in range(10):
                self.sim.data.set_joint_qpos('robot0:joint7_l', -0.0)
                self.sim.data.set_joint_qpos('robot0:joint7_r', -0.0)
                self.sim.forward()
                self.sim.step()
            utils.ctrl_set_action(self.sim, action)
            self.sim.data.set_mocap_pos('robot0:mocap', object_qpos[:3])
            for _ in range(10):
                self.sim.data.set_joint_qpos('robot0:joint7_l', -0.0)
                self.sim.data.set_joint_qpos('robot0:joint7_r', -0.0)
                self.sim.forward()
                self.sim.step()

            # Grasp and lift the gripper
            action = np.concatenate([[0, 0, 0], [0, 0, 0, 0], [0.3, 0.3]])
            # Apply action to simulation.
            utils.ctrl_set_action(self.sim, action)
            utils.mocap_set_action(self.sim, action)
            for _ in range(10):
                self.sim.step()
            action = np.concatenate([[0, 0, self.lift_threshold], [0, 0, 0, 0], [0.3, 0.3]])
            # Apply action to simulation.
            utils.ctrl_set_action(self.sim, action)
            utils.mocap_set_action(self.sim, action)
            for _ in range(10):
                self.sim.step()
            self.initial_goal_distance = goal_distance(overbox_pos, object_qpos[:3])
        elif self.reward_type == 'composite':
            # Place the box far away from the workspace
            box_xpos = self.initial_box_xpos[:2] + self.np_random.uniform(-self.box_range, self.box_range, size=2) + [1,2]
            self.box_qpos = self.sim.data.get_joint_qpos('box:joint')
            assert self.box_qpos.shape == (7,)
            self.box_qpos[:2] = box_xpos
            # Set box position
            self.initial_box_xpos = self.box_qpos[:3]
            self.sim.data.set_joint_qpos('box:joint', self.box_qpos)

            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-0.10, 0.10, size=2) - [0.1, 0]
            base_to_object_distance = np.linalg.norm(object_xpos[:2] - self.sim.data.get_site_xpos('robot0:base')[:2])

            # Set object 0 initial position
            object0_qpos = self.sim.data.get_joint_qpos('object0:joint')
            object0_qpos[:2] = object_xpos

            # Place object 1 within object range of object0
            object1_qpos = self.sim.data.get_joint_qpos('object1:joint')
            object1_xpos = object_xpos + self.np_random.uniform(-0.15, 0.15, size=2)

            # place the objects at least 5 cm apart
            while np.linalg.norm(object1_xpos - object_xpos) > 0.1:
                object1_xpos = object_xpos + self.np_random.uniform(-0.15, 0.15, size=2)

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
            if bool(np.random.binomial(1, 0.5)):
                initial_grip = np.add(self.initial_object_xpos, [0, 0, 0.043])
                self.sim.data.set_mocap_quat('robot0:mocap', [0, 0, 1, 0])
                self.sim.data.set_mocap_pos('robot0:mocap', initial_grip)
                for _ in range(20):
                    self.sim.step()
                self.sim.data.set_mocap_quat('robot0:mocap', [0, 0, 1, 0])
                self.sim.data.set_mocap_pos('robot0:mocap', np.subtract(initial_grip, [0, 0, 0.035]))
                for _ in range(20):
                    self.sim.step()
                # Grasp
                action = np.concatenate([[0, 0, 0], [0, 0, 0, 0], [0.3, 0.3]])
                # Apply action to simulation.
                utils.ctrl_set_action(self.sim, action)
                for _ in range(20):
                    self.sim.forward()

        else:
            raise Exception('Invalid reward type:' + self.reward_type + ' \n use either: reach, orient, lift, place, composite')
        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.reward_type == 'reach':
            target_y_range = 0.08  # The length of the box
            target_x_range = 0.125  # The width of the box
            target_height = 0.05  # The height of the box
            self.box_offset = [
                float(self.np_random.uniform(-target_x_range, target_x_range)),
                float(self.np_random.uniform(-target_y_range, target_y_range)),
                target_height]
            goal = self.sim.data.get_site_xpos('box')[:3] + self.box_offset

        elif self.reward_type == 'orient':
            goal = self.sim.data.get_site_xpos('object0')[:3]
        elif self.reward_type == 'lift':
            # The goal is the z height lift_threshold over the box
            goal = self.sim.data.get_site_xpos('object0')[2] + self.lift_threshold
        elif self.reward_type == 'place':
            # goal zone size 10x10cm
            goal_offset = self.sample_point(0.1, 0.1, 0.1)
            goal_height = 0.53
            # add + 1.025 to y to get the right pos for the second robot
            goal_pos = np.array([0.63, 0.5, goal_height]) + goal_offset  # A goal just beside the robot

            theta = np.random.uniform(0, 2 * np.pi)
            goal_quat = [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
            goal = np.concatenate((goal_pos, goal_quat))

            # # goal zone size 10x10cm
            # goal_offset = self.sample_point(0.02, 0.02, 0.02)
            # goal_height = 0.43
            # grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            # object_pos = self.sim.data.get_site_xpos('object0')
            # goal_pos = np.array(object_pos) + goal_offset  # A goal just beside the robot
            # grip_rot = self.sim.data.get_site_xmat('robot0:grip')
            # object_rot = self.sim.data.get_site_xmat('object0')
            # grip_rot_euler = rotations.mat2euler(grip_rot)
            # object_rot_euler = rotations.mat2euler(object_rot)
            # object_rot_euler[2] = float(self.np_random.uniform(-np.pi, np.pi))

            #Sample a random rotation
            # rot_grip = Rotation.random().as_matrix()
            # rot_box = self.sim.data.get_site_xmat('box')
            # rot_gb = np.matmul(rot_grip.transpose(), rot_box)
            # p_box = np.array([0, 0, -1])
            # p_g = np.matmul(rot_gb, p_box)
            # alpha_z = np.arccos(p_g[2] / np.sqrt(p_g[0] ** 2 + p_g[1] ** 2 + p_g[2] ** 2))
            #
            # while math.degrees(alpha_z) > 15:
            #     rot_grip = Rotation.random().as_matrix()
            #     rot_gb = np.matmul(rot_grip.transpose(), rot_box)
            #     p_g = np.matmul(rot_gb, p_box)
            #     alpha_z = np.arccos(p_g[2] / np.sqrt(p_g[0] ** 2 + p_g[1] ** 2 + p_g[2] ** 2))
            # TODO: Test if this works properly
            #goal = np.concatenate((goal_pos, rotations.euler2quat(object_rot_euler)))

        elif self.reward_type == 'composite':
            goal = self.sim.data.get_site_xpos('object0') + [0, 0, 0.030] # Set the goal position over the other object
        else:
            raise Exception('Invalid reward type:' + self.reward_type + ' \n use either: reach, orient, lift, place')
        return goal.ravel().copy()

    def _is_success(self, achieved_goal, desired_goal):
        result = False
        if self.reward_type == 'reach':
            d = goal_distance(self.sim.data.get_site_xpos('robot0:grip'), desired_goal)
            result = (d < self.success_threshold).astype(np.bool)
        elif self.reward_type == 'orient':
            body_id1 = self.sim.model.body_name2id('robot0:left_finger')
            body_id2 = self.sim.model.body_name2id('robot0:right_finger')

            finger_left = self.sim.data.body_xpos[body_id1]
            finger_right = self.sim.data.body_xpos[body_id2]

            orient_line = finger_left - finger_right
            rot_object0 = self.sim.data.get_site_xmat('object0')
            theta_x = min(angle_between(orient_line, np.matmul(rot_object0, (1, 0, 0))),
                          angle_between(orient_line, np.matmul(rot_object0, (-1, 0, 0))))
            theta_y = min(angle_between(orient_line, np.matmul(rot_object0, (0, 1, 0))),
                          angle_between(orient_line, np.matmul(rot_object0, (0, -1, 0))))
            theta_z = angle_between(orient_line, np.matmul(rot_object0, (0, 0, 1)))
            angle_45 = math.radians(45)
            angle_90 = math.radians(90)

            theta_x = angle_45 - np.abs(theta_x - angle_45)
            theta_y = angle_45 - np.abs(theta_y - angle_45)
            theta_z = angle_90 - np.abs(theta_z - angle_90)

            if goal_distance(achieved_goal, desired_goal) < self.success_threshold and (theta_x < math.radians(10) or theta_y > math.radians(10)) and theta_z > math.radians(45):
                result = True
        elif self.reward_type == 'lift':
            # A lift is successful if the object has been lifted lift_threshold over the box
            object_height = self.sim.data.get_site_xpos('object0')[2]
            table_height = 0.414  # 0.4 is the height of the table(0.014 extra for inaccuracies in the sim)
            lift_cylinder_radius = 0.05
            dist_vec = np.abs(self.sim.data.get_site_xpos('object0')[:2]-self.initial_object_pos[:2])
            radial_dist = np.sqrt(np.square(dist_vec[0])+np.square(dist_vec[1]))
            if np.abs(object_height - table_height) > self.lift_threshold and radial_dist < lift_cylinder_radius:
                result = True
        elif self.reward_type == 'place':
            goal_rot = rotations.quat2mat(desired_goal[3:])
            goal_x = np.matmul(goal_rot, (1, 0, 0))
            goal_y = np.matmul(goal_rot, (0, 1, 0))
            rot_object0 = self.sim.data.get_site_xmat('object0')
            theta_x = min(angle_between(goal_x, np.matmul(rot_object0, (1, 0, 0))),
                          angle_between(goal_x, np.matmul(rot_object0, (-1, 0, 0))))
            theta_y = min(angle_between(goal_y, np.matmul(rot_object0, (0, 1, 0))),
                          angle_between(goal_y, np.matmul(rot_object0, (0, -1, 0))))
            angle_45 = math.radians(45)

            theta_x = angle_45 - np.abs(theta_x - angle_45)
            theta_y = angle_45 - np.abs(theta_y - angle_45)

            if goal_distance(self.sim.data.get_site_xpos('object0'), desired_goal[:3]) < self.success_threshold and (theta_x < math.radians(10) or theta_y > math.radians(10)):
                result = True
        elif self.reward_type == 'composite':
            if goal_distance(achieved_goal[:3], desired_goal[:3]) < 0.01:
                result = True
        return result

    def _is_failed(self):
        result = False
        if self.reward_type == 'reach':
            pass
        elif self.reward_type == 'orient':
            # calculate the tilt of object
            rot_object0 = self.sim.data.get_site_xmat('object0')
            object_tilt = angle_between(np.array((0, 0, 1)), np.matmul(rot_object0, (0, 0, 1)))
            if object_tilt > math.radians(15):
                result = True
            if goal_distance(self.sim.data.get_site_xpos('robot0:grip'), self.sim.data.get_site_xpos('object0')) > 0.4:
                result = True

        elif self.reward_type == 'lift':
            # A lift is successful if the object has been lifted lift_threshold over the box
            lift_cylinder_radius = 0.05
            dist_vec = np.abs(self.sim.data.get_site_xpos('object0')[:2]-self.initial_object_pos[:2])
            radial_dist = np.sqrt(np.square(dist_vec[0])+np.square(dist_vec[1]))

            # If the object is outside the cylinder zone
            if radial_dist > lift_cylinder_radius:
                result = True
            # If the object is further than 5cm of the gripper
            if np.linalg.norm(self.sim.data.get_site_xpos('object0') - self.sim.data.get_site_xpos('robot0:grip'), axis=-1) > 0.05:
                result = True

        elif self.reward_type == 'place':
            # If the object is further than 5cm of the gripper
            if np.linalg.norm(self.sim.data.get_site_xpos('object0') - self.sim.data.get_site_xpos('robot0:grip'), axis=-1) > 0.05:
                result = True
        elif self.reward_type == 'composite':
            result = False
        return result

    def _is_collision(self):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if 'robot0' in self.sim.model.geom_id2name(contact.geom1) or 'robot0' in self.sim.model.geom_id2name(contact.geom2):
                if 'object' not in self.sim.model.geom_id2name(contact.geom1) or 'object' not in self.sim.model.geom_id2name(contact.geom2):
                    return True
        return False

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
        self.initial_box_xpos = self.sim.data.get_site_xpos('box').copy()

    def render(self, mode='human', width=500, height=500):
        return super(UrBinPickingEnv, self).render(mode)
