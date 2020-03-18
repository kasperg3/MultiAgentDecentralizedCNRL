import numpy as np

from gym.envs.robotics import rotations, utils
from gym_mergablerobots.envs import robot_env
from scipy.spatial.transform import Rotation
import mujoco_py


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class UrBinPickingEnv(robot_env.RobotEnv):
    """Superclass for UR environments.
    """

    def __init__(self,
                 model_path,
                 n_substeps,
                 initial_qpos,
                 reward_type,
                 box_range,
                 success_threshold):

        """Initializes a new Ur environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.reward_type = reward_type
        self.box_range = box_range
        self.success_threshold = success_threshold

        super(UrBinPickingEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=8,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)

        if self.reward_type == 'reach':
            return float(-d)
        elif self.reward_type == 'orient':
            pass
        elif self.reward_type == 'lift':
            pass
    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        # Lock gripper open in reach
        if self.reward_type == 'reach' or self.reward_type == 'orient':
            self.sim.data.set_joint_qpos('robot0:joint7_l', -0.008)
            self.sim.data.set_joint_qpos('robot0:joint7_r', -0.008)
            self.sim.forward()

    def _set_action(self, action):
        # Change action space if number of is changed
        assert action.shape == (8,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7]
        pos_ctrl *= 0.01  # limit maximum change in position
        rot_ctrl *= 0.1
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.reward_type == 'reach':
            gripper_ctrl = np.zeros_like(gripper_ctrl)
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
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        box_pos = self.sim.data.get_site_xpos('box')
        box_rel_pos = box_pos - grip_pos
        box_rot = rotations.mat2euler(self.sim.data.get_site_xmat('box'))

        achieved_goal = grip_pos.copy()

        # The relative position to the goal
        goal_rel_pos = self.goal - achieved_goal

        obs = None
        if self.reward_type == 'reach':
            # Update the goal to follow the box
            self.goal = self.sim.data.get_site_xpos('box')[:3] + self.box_offset
            obs = np.concatenate([
                grip_pos,
                grip_rot,
                grip_velp,
                grip_velr,
                box_pos.ravel(),
                box_rel_pos.ravel(),
                box_rot.ravel(),
                goal_rel_pos,
            ])
        elif self.reward_type == 'orient':
            # Set gripper state and velocity for orient where it's possible to move the gripper
            gripper_state = robot_qpos[-2:]
            gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
            # Object position
            object0_pos = self.sim.data.get_site_xpos('object0')
            object1_pos = self.sim.data.get_site_xpos('object1')
            object2_pos = self.sim.data.get_site_xpos('object2')

            # Object rotation
            object0_rot_cart = self.sim.data.get_site_xmat('object0')
            object1_rot_cart = self.sim.data.get_site_xmat('object1')
            object2_rot_cart = self.sim.data.get_site_xmat('object2')
            object0_rot = Rotation.as_quat(object0_rot_cart)
            object1_rot = Rotation.as_quat(object1_rot_cart)
            object2_rot = Rotation.as_quat(object2_rot_cart)

            # relative object pos

            obs = np.concatenate([
                grip_pos,
                grip_rot,
                grip_velp,
                grip_velr,
                box_pos.ravel(),
                box_rel_pos.ravel(),
                box_rot.ravel(),
                object0_pos,
                object0_rot,
                object1_pos,
                object1_rot,
                object2_pos,
                object2_rot,
            ])
        elif self.reward_type == 'lift':
            pass
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

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.action_counter = 0
        # Randomize start position of object.
        if self.reward_type == 'reach':
            box_xpos = self.initial_box_xpos[:2] + self.np_random.uniform(-self.box_range, self.box_range, size=2)
            box_qpos = self.sim.data.get_joint_qpos('box:joint')
            assert box_qpos.shape == (7,)
            box_qpos[:2] = box_xpos
            # Set object position
            self.initial_box_xpos = box_qpos[:3]
            self.sim.data.set_joint_qpos('box:joint', box_qpos)
        elif self.reward_type == 'orient':
            # Start the simulation just over the box(same space as reach goal)

            pass
        elif self.reward_type == 'lift':
            box_xpos = self.initial_box_xpos[:2] + self.np_random.uniform(-self.box_range, self.box_range, size=2)
            box_qpos = self.sim.data.get_joint_qpos('box:joint')
            assert box_qpos.shape == (7,)
            box_qpos[:2] = box_xpos
            # Set object position
            self.initial_box_xpos = box_qpos[:3]
            self.sim.data.set_joint_qpos('box:joint', box_qpos)

            self.sim.data.set_joint_qpos('object0')

            # Start the simulation with the object between the fingers
            pass
        elif self.reward_type == 'place':
            # Start the simulation over the box with the
            pass
        else:
            raise Exception('Invalid reward type:' + self.reward_type + ' \n use either: reach, orient, lift, place')

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
            pass
        elif self.reward_type == 'lift':
            pass
        elif self.reward_type == 'place':
            pass
        else:
            target_range = 0.2
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-target_range, target_range, size=3)
        return goal.ravel().copy()

    def _is_success(self, achieved_goal, desired_goal):
        # Todo: Add different success criteria depending on the goal
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.success_threshold).astype(np.float32)

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
