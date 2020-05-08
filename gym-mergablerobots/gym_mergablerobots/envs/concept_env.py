import random
import os
import copy
from gym.envs.robotics import rotations, utils
from gym_mergablerobots.envs import robot_env
from scipy.spatial.transform import Rotation
import mujoco_py
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from TD3 import TD3


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


class ConceptEnv(gym.Env):
    """
    """
    def task_success(self, agent):
        object_position = self.sim.data.get_site_xpos('object'+agent)
        if goal_distance(self.goal_place[int(agent)], object_position) < 0.10:
            # print('Success -- for debug')
            return True
        return False

    def sample_point(self, x_range, y_range, z_range):
        return [
            float(self.np_random.uniform(-x_range, x_range)),
            float(self.np_random.uniform(-y_range, y_range)),
            float(self.np_random.uniform(-z_range, z_range))]

    def rad_between_obj_and_grip(self, agent):
        body_id1 = self.sim.model.body_name2id('robot'+agent+':left_finger')
        body_id2 = self.sim.model.body_name2id('robot'+agent+':right_finger')

        finger_left = self.sim.data.body_xpos[body_id1]
        finger_right = self.sim.data.body_xpos[body_id2]

        orient_line = finger_left - finger_right
        rot_object0 = self.sim.data.get_site_xmat('object'+agent)
        theta_x = min(angle_between(orient_line, np.matmul(rot_object0, (1, 0, 0))),
                      angle_between(orient_line, np.matmul(rot_object0, (-1, 0, 0))))
        theta_y = min(angle_between(orient_line, np.matmul(rot_object0, (0, 1, 0))),
                      angle_between(orient_line, np.matmul(rot_object0, (0, -1, 0))))
        theta_z = angle_between(orient_line, np.matmul(rot_object0, (0, 0, 1)))
        angle_45 = math.radians(45)
        angle_90 = math.radians(90)

        theta_array = np.array([0.785, 0.785, 0.001]) # float of worst-case
        theta_array[0] = angle_45 - np.abs(theta_x - angle_45)
        theta_array[1] = angle_45 - np.abs(theta_y - angle_45)
        theta_array[2] = angle_90 - np.abs(theta_z - angle_90)
        return theta_array

    def orientation_is_success(self, agent):
        xy_success_threshold = math.radians(10)
        z_success_threshold = math.radians(45)
        theta_xyz = self.rad_between_obj_and_grip(agent)
        if min(theta_xyz[0], theta_xyz[1]) < xy_success_threshold and theta_xyz[2] > z_success_threshold:
            return True
        return False

    def gripper_is_closed(self, agent, threshold):
        body_id1 = self.sim.model.body_name2id('robot' + agent + ':left_finger')
        body_id2 = self.sim.model.body_name2id('robot' + agent + ':right_finger')

        finger_left = self.sim.data.body_xpos[body_id1]
        finger_right = self.sim.data.body_xpos[body_id2]

        fingers_closed = bool(goal_distance(finger_left, finger_right) < threshold)
        return fingers_closed

    def grasped_object(self, agent):
        contact_points = 0
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if 'robot' + str(agent) in self.sim.model.geom_id2name(contact.geom1) or 'robot' + str(agent) in self.sim.model.geom_id2name(contact.geom2):
                if 'object' + str(agent) not in self.sim.model.geom_id2name(contact.geom1) or 'object' + str(agent) not in self.sim.model.geom_id2name(contact.geom2):
                    contact_points += 1

        return contact_points

    def _is_collision(self):
        pass

    def __init__(self,
                 model_path,
                 n_substeps,
                 initial_qpos,
                 n_actions,
                 n_agents,
                 max_action_steps):

        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.initial_gripper_xpos = [0, 0]
        self.initial_gripper_quat = [0, 0]

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.n_actions = n_actions
        self.current_action = [6, 6]
        self.goal_visuliser_array = [('0', [0]), ('1', [0])]
        self.goal = []
        self.gripper_ctrl = np.array((-0.3, -0.3))
        obs = self._get_obs('0')
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_agents, obs.shape[0]), dtype='float32')
        self.goal_place = np.array([[0.63, 0.5, 0.65], [0.63, 1.525, 0.65]])

        self.move_home = [False, False]
        # Multi-agent variables
        self.actions_available = {
                                    "REACH": 0,
                                    "ORIENT": 1,
                                    "LIFT": 2,
                                    "PLACE": 3,
                                    "CLOSE_GRIPPER": 4,
                                    "OPEN_GRIPPER": 5,
                                    "NOOP": 6,
                                    "HOME": 7,
        }

        self.n_agents = n_agents
        self.current_action_steps = [0, 0]
        self.max_action_steps = max_action_steps

        # Policies and TD3 variables
        self.policies = {}
        for action in self.actions_available:
            action_dim = 7
            max_action = 1.0
            if action == "REACH":
                name = "Reach"
                state_dim = 29
            elif action == "LIFT":
                name = "Lift"
                state_dim = 31
            elif action == "ORIENT":
                name = "Orient"
                state_dim = 39
            elif action == "PLACE":
                name = "Place"
                action_dim = 4
                state_dim = 40
            else:  # If not a trained policy, then skip and dont load a policy
                continue
            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
            }
            file_name = f"TD3_UrBinPicking{name}-v0_2345"
            self.policies[self.actions_available[action]] = [TD3(**kwargs), TD3(**kwargs)]
            self.policies[self.actions_available[action]][0].load(f"./models/concepts_29_04/agent0/{file_name}")
            self.policies[self.actions_available[action]][1].load(f"./models/concepts_29_04/agent1/{file_name}")

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def compute_agent_reward(self, agent):
        reward = 0
        lift_threshold = 0.10
        dist_success_threshold = 0.05
        dist_success_threshold_place = 0.10
        object_position = self.sim.data.get_site_xpos('object'+agent)
        gripper_position = self.sim.data.get_site_xpos('robot'+agent+':grip')
        object_height = object_position[2] - 0.414
        is_lifted = bool(object_height > lift_threshold)
        grip_to_object_rel_distance = goal_distance(object_position, gripper_position)

        #hardcoded
        theta = math.radians(35)
        place_goal_pos = self.goal_place[int(agent)]

         # rewards
        if grip_to_object_rel_distance < dist_success_threshold:  # Reach
            reward = 1
        if self.orientation_is_success(agent):                    # Orient
            reward = 1.5
        if self.gripper_is_closed(agent, 0.03) and grip_to_object_rel_distance < 0.010:     # Grasped
            reward = 2.5
        if is_lifted:                                             # Lift
           reward = 4
        if goal_distance(object_position, place_goal_pos) < dist_success_threshold_place:  # Place
           reward = 15  # 15 if not sparse
        # if goal_distance(object_position, place_goal_pos) < dist_success_threshold_place:  # Place
        #     reward = 1
        return reward

    def _step_callback(self):
        pass

    """
    get_concept_state: Returns the state space of the agent
    concept: integer corresponding to a specific concept
    agent: number of the agent used to get the right state space
    """

    def get_concept_state(self, concept, agent):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot' + str(agent) + ':grip')
        grip_rot_cart = Rotation.from_matrix(
            self.sim.data.get_site_xmat('robot' + str(agent) + ':grip'))  # The rotation of gripper(Cartesian)
        grip_rot = Rotation.as_quat(grip_rot_cart)
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot' + str(agent) + ':grip') * dt
        grip_velr = self.sim.data.get_site_xvelr('robot' + str(agent) + ':grip') * dt

        object_velp = self.sim.data.get_site_xvelp('object' + str(agent)) * dt
        object_velr = self.sim.data.get_site_xvelr('object' + str(agent)) * dt

        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        box_pos = self.sim.data.get_site_xpos('box')
        box_rel_pos = box_pos - grip_pos
        box_rot = Rotation.from_matrix(self.sim.data.get_site_xmat('box')).as_quat()

        object_pos = self.sim.data.get_site_xpos('object' + str(agent))
        object_rel_pos = object_pos - grip_pos
        object_rot = Rotation.from_matrix(self.sim.data.get_site_xmat('object' + str(agent))).as_quat()

        achieved_goal = grip_pos.copy()

        # Define a goal and concat at the front of the observation
        if concept == self.actions_available["REACH"]:
            target_height = 0.05  # The height of the box
            box_offset = [0, 0, target_height]
            goal = self.sim.data.get_site_xpos('box')[:3].copy() + box_offset
            # The relative position to the goal
            goal_rel_pos = goal - achieved_goal
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
        elif concept == self.actions_available["LIFT"]:
            # The relative position is between object and goal position
            object_height = self.sim.data.get_site_xpos('object' + str(agent))[2]
            goal = (self.sim.data.get_site_xpos('box')[2] + 0.15).ravel()
            goal_rel_height = goal - object_height
            obs = np.concatenate([
                grip_pos,
                grip_rot,
                grip_velp,
                grip_velr,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                box_pos.ravel(),
                box_rel_pos.ravel(),
                box_rot.ravel(),
                goal_rel_height,
            ])
        elif concept == self.actions_available["ORIENT"]:
            # Update the goal to follow the box
            goal = self.sim.data.get_site_xpos('object' + str(agent))
            # The relative position to the goal
            goal_rel_pos = goal - achieved_goal
            obs = np.concatenate([
                grip_pos,
                grip_rot,
                grip_velp,
                grip_velr,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                object_rot.ravel(),
                box_pos.ravel(),
                box_rel_pos.ravel(),
                box_rot.ravel(),
                goal_rel_pos,
            ])
        elif concept == self.actions_available["PLACE"]:
            # TODO: make the goal dynamic, so one can change it if one wants to move the position of place
            goal_offset = self.sample_point(0.1, 0.1, 0.1)
            base_goal_pos = self.goal_place[int(agent)]
            goal_pos = base_goal_pos #+ goal_offset  # A goal just beside the robot

            #theta = np.random.uniform(0, 2 * np.pi)  # TODO: make random in later implementation
            theta = math.radians(35)
            goal_quat = [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
            goal = np.concatenate((goal_pos, goal_quat))
            achieved_goal = np.concatenate((object_pos, object_rot.ravel()))
            goal_rel_pos = goal[:3] - achieved_goal[:3]
            grip_q = rotations.mat2quat(self.sim.data.get_site_xmat('robot'+agent+':grip'))
            goal_q = goal[3:]
            goal_rel_rot = np.array(2 * np.arccos(np.abs(np.inner(grip_q, rotations.quat_conjugate(goal_q)))))

            obs = np.concatenate([
                grip_pos,
                grip_rot,
                object_pos,
                object_rot.ravel(),
                object_velp,
                object_velr,
                object_rel_pos,
                box_pos.ravel(),
                box_rel_pos.ravel(),
                goal_rel_pos,
                goal_rel_rot.ravel(),
            ])

        self.goal_visuliser_array[int(agent)] = [(concept, goal)]
        self.goal = goal
        return np.concatenate((goal, obs))

    """
        choose_action: Takes a action for a single agent
        action: Discrete action from the DQN
        agent: used to determine what states to use for the different policies
        returns: agent_done , agent_movement
    """

    def choose_action(self, action, agent):
        # get a movement from a policy dependent on the agent and the action chosen
        # Set agent_action_done to True if
        Negative_termination_active = True   # SET TRUE TO ACTIVATE NEGATIVE TERMINATION
        self.move_allowed[agent] = True
        self.move_home[agent] = False
        agent_movement = np.zeros(7)
        agent_done = False

        has_object = False
        object_pos = self.sim.data.get_site_xpos('object'+str(agent))
        gripper_pos = self.sim.data.get_site_xpos('robot'+str(agent)+':grip')
        gripper_open = np.bool(self.gripper_ctrl[int(agent)] == -1)
        if goal_distance(object_pos, gripper_pos) < 0.02 and not gripper_open:
            has_object = True


        # increment timeout counter
        self.current_action_steps[agent] += 1  # increment the amount of
        if self.current_action_steps[agent] >= self.max_action_steps:
            agent_done = True
            self.current_action_steps[agent] = 0

        elif action == self.actions_available["NOOP"]:
            self.move_allowed[agent] = False

        elif action == self.actions_available["REACH"]:
            state = self.get_concept_state(action, str(agent))
            agent_movement = self.policies[action][agent].select_action(state)
            d = np.linalg.norm(self.sim.data.get_site_xpos('robot' + str(agent) + ':grip') - (self.sim.data.get_site_xpos('box') + [0, 0, 0.05]), axis=-1)
            agent_done = (d < 0.05).astype(np.bool)
            if has_object and Negative_termination_active:
                agent_done = True  # Neg Termination

        elif action == self.actions_available["LIFT"]:
            state = self.get_concept_state(action, str(agent))
            agent_movement = self.policies[action][agent].select_action(state)
            table_height = 0.414
            object_height = self.sim.data.get_site_xpos('object' + str(agent))[2]
            if np.abs(object_height - table_height) >= 0.14:
                agent_done = True
            if not has_object and Negative_termination_active:
                agent_done = True  # Neg Termination

        elif action == self.actions_available["ORIENT"]:
            state = self.get_concept_state(action, str(agent))
            agent_movement = self.policies[action][agent].select_action(state)
            d = np.linalg.norm(self.sim.data.get_site_xpos('robot' + str(agent) + ':grip') - self.sim.data.get_site_xpos('object' + str(agent)), axis=-1)
            if d < 0.015:
                agent_done = True
            if (d > 0.4 or not gripper_open) and Negative_termination_active:
                agent_done = True  # Neg termination

        elif action == self.actions_available["CLOSE_GRIPPER"]:
            self.gripper_ctrl[agent] = 0.3
            if self.gripper_is_closed(str(agent), 0.04) or self.grasped_object(agent) > 2:
                agent_done = True

        elif action == self.actions_available["OPEN_GRIPPER"]:
            self.gripper_ctrl[agent] = -1
            if not self.gripper_is_closed(str(agent), 0.04):
                agent_done = True

        elif action == self.actions_available["PLACE"]:
            state = self.get_concept_state(action, str(agent))
            policy_output = self.policies[action][agent].select_action(state)
            rot_ctrl = (rotations.euler2quat([0, np.pi, policy_output[3] * 2 * np.pi]) * rotations.quat_conjugate(rotations.mat2quat(self.sim.data.get_site_xmat('robot0:grip')))) * 2
            pos_crtl = policy_output[:3] * 0.5
            agent_movement = np.concatenate((pos_crtl, rot_ctrl))
            if goal_distance(self.goal_place[int(agent)], self.sim.data.get_site_xpos('robot'+str(agent)+':grip')) < 0.05:
                agent_done = True
            if not has_object and Negative_termination_active:
                agent_done = True  # Neg Termination

        elif action == self.actions_available["HOME"]:
            self.move_home[agent] = True

        if agent_done:
            self.current_action_steps[agent] = 0
        return agent_done, agent_movement

    """ step:
            This method should only return when the agent has completed an action or timeouted
        action: Contains an array of n discrete agent actions ranging from 0 to available action of the agent
    """

    def step(self, action):
        # This should be the same concept until the concept is done
        # Choose a new concept independently depending on the robot
        agent_movement = np.empty((2, 7))
        info = {"agent_done": [-1, -1]}

        while True:
            # Choose action
            for agent in range(self.n_agents):
                agent_done, agent_movement[agent] = self.choose_action(action[agent], agent)
                if agent_done:
                    self.move_allowed[agent] = False
                    info["agent_done"][agent] = True

            # Act in the environment
            self._set_action(agent_movement)
            self.sim.step()
            self._step_callback()
            done = False
            #self.render()

            # If any agents are done, then break the while
            observation_arr = np.empty(self.observation_space.shape)
            reward_arr = np.empty((2,))
            if info["agent_done"][0] == 1 and info["agent_done"][1] == 1:
                for agent in range(len((info["agent_done"]))):
                    # Extract the observation and reward
                    observation_arr[agent] = self._get_obs(str(agent))
                    reward_arr[agent] = self.compute_agent_reward(str(agent))
                return observation_arr, reward_arr, done, info

    def sample_action(self):
        return np.array((
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1))))

    def _set_action(self, action):
        # Change action space if number of actions is changed
        assert action.shape == (2, 7)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        mocap_action = np.zeros((2, 7))
        actuator_action = np.zeros((2, 2))
        for i in range(action.shape[0]):
            pos_ctrl, rot_ctrl, gripper_ctrl = action[i][:3], action[i][3:7], self.gripper_ctrl[i]
            pos_ctrl *= 0.05  # limit maximum change in position
            rot_ctrl *= 0.01
            actuator_action[i] = np.array((gripper_ctrl, gripper_ctrl))

            # Disallow movement when closing or opening gripper
            if not self.move_allowed[i]:
                pos_ctrl = np.zeros(3)
                pos_ctrl[2] = 0.0002  # To avoid drifting in z
                rot_ctrl = np.zeros(4)

            if self.move_home[i]:
                pos_ctrl = (self.initial_gripper_xpos[i] - self.sim.data.mocap_pos[i])*0.2
                rot_ctrl = rotations.euler2quat([0, np.pi, 1.5*np.pi])
            mocap_action[i] = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, np.concatenate((np.zeros(14), actuator_action.ravel())))
        utils.mocap_set_action(self.sim, mocap_action)

    def _get_obs(self, agent):
        dist_success_threshold = 0.02
        lift_threshold = 0.10
        grasp_ready = False
        has_object = False
        object_lifted = False
        pinch_point = self.sim.data.get_site_xpos('robot'+agent+':grip')
        opposite_pinch_point = []
        if agent == '0':
            opposite_pinch_point = self.sim.data.get_site_xpos('robot1:grip')
        if agent == '1':
            opposite_pinch_point = self.sim.data.get_site_xpos('robot0:grip')

        object_position = self.sim.data.get_site_xpos('object' + agent)
        grip_to_object_rel_distance = goal_distance(pinch_point, object_position)

        gripper_open = np.bool(self.gripper_ctrl[int(agent)] == -1)
        if grip_to_object_rel_distance < dist_success_threshold and self.orientation_is_success(agent):
            grasp_ready = True
        if not gripper_open and grip_to_object_rel_distance < 0.02:
            has_object = True
        if (object_position[2]-0.414) > lift_threshold:
            object_lifted = True

        obs = np.concatenate([
            pinch_point.ravel(),
            object_position.ravel(),
            grip_to_object_rel_distance.ravel(),
            np.array([grasp_ready]),
            np.array([has_object]),
            np.array([object_lifted]),
            np.array([gripper_open]), # Gripper open bool
            opposite_pinch_point.ravel(),
        ])

        return obs.copy()

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
        site_id = [self.sim.model.site_name2id('target0'), self.sim.model.site_name2id('target1')]
        for agent in range(len(self.goal_visuliser_array)):
            if self.goal_visuliser_array[agent][0][0] == self.actions_available["REACH"]:
                visualized_goal = self.goal_visuliser_array[agent][0][1]
                self.sim.model.site_pos[site_id[agent]] = visualized_goal - sites_offset[0]

            elif self.goal_visuliser_array[agent][0][0] == self.actions_available["ORIENT"]:
                visualized_goal = self.goal_visuliser_array[agent][0][1]
                self.sim.model.site_pos[site_id[agent]] = visualized_goal[:3] - sites_offset[0]
                self.sim.model.site_quat[site_id[agent]] = rotations.mat2quat(self.sim.data.get_site_xmat('object'+str(agent)))

            elif self.goal_visuliser_array[agent][0][0] == self.actions_available["LIFT"]:
                # Lift only has a height goal, so we set the visualization target to the box xy pos and the goal height
                visualized_goal = self.sim.data.get_site_xpos('box').copy()
                visualized_goal[2] = self.goal_visuliser_array[agent][0][1]  # z-axis of the goal is the only thing contained in goal in lift
                self.sim.model.site_pos[site_id[agent]] = visualized_goal - sites_offset[0]

            elif self.goal_visuliser_array[agent][0][0] == self.actions_available["PLACE"]:
                # Place also has a orientation as goal
                goal_pos = self.goal_visuliser_array[agent][0][1][:3]
                goal_quat = self.goal_visuliser_array[agent][0][1][3:]
                self.sim.model.site_pos[site_id[agent]] = goal_pos - sites_offset[0]
                self.sim.model.site_quat[site_id[agent]] = goal_quat

            elif self.goal_visuliser_array[agent][0][0] == self.actions_available["HOME"]:
                self.sim.model.site_pos[site_id[agent]] = self.initial_gripper_xpos[agent] - sites_offset[0]
            self.sim.forward()

    def sample_box_position(self):
        box_xpos = self.initial_box_xpos[:2] + self.np_random.uniform(-0.1, 0.1, size=2)
        self.box_qpos = self.sim.data.get_joint_qpos('box:joint')
        assert self.box_qpos.shape == (7,)
        self.box_qpos[:2] = box_xpos
        # Set box position
        self.initial_box_xpos = self.box_qpos[:3]
        self.sim.data.set_joint_qpos('box:joint', self.box_qpos)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.episode_steps = 0

        self.sample_box_position()

        #Start with the gripper open:
        self.gripper_ctrl = np.array((-0.3, -0.3))
        target_y_range = 0.05  # The length of the box
        target_x_range = 0.1  # The width of the box
        target_height = 0.05  # The height of the box

        # Allow movement for all other actions than close and open gripper
        self.move_allowed = np.array((True, True), np.bool)

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
        object_qpos[:3] = self.sim.data.get_joint_qpos('box:joint')[:3] + object_offset

        # random object1 spawn (z-rot and xyz)
        object_qpos = self.sim.data.get_joint_qpos('object1:joint')
        object_offset = [
            float(self.np_random.uniform(-target_x_range, target_x_range)),
            float(self.np_random.uniform(-target_y_range, target_y_range)),
            target_height]
        object_z_angle = float(self.np_random.uniform(-np.pi, np.pi))
        object_qpos_euler = rotations.quat2euler(object_qpos[3:])
        object_qpos_euler[2] = object_z_angle
        object_qpos[3:] = rotations.euler2quat(object_qpos_euler)
        object_qpos[:3] = self.sim.data.get_joint_qpos('box:joint')[:3] + object_offset

        self.sim.forward()
        if np.linalg.norm(self.sim.data.get_joint_qpos('object1:joint')[:3] - self.sim.data.get_joint_qpos('object0:joint')[:3]) < 0.10:
            self._reset_sim()
        return True

    def _sample_goal(self):
        goal = self.sim.data.get_site_xpos('box')[:3]
        return goal.ravel().copy()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper0_target = self.sim.data.get_site_xpos('robot0:grip')
        gripper0_rotation = [0, -1, 1, 0]
        gripper1_target = self.sim.data.get_site_xpos('robot1:grip')
        gripper1_rotation = [0, -1, 1, 0]

        self.sim.data.set_mocap_pos('robot0:mocap', gripper0_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper0_rotation)
        self.sim.data.set_mocap_pos('robot1:mocap', gripper1_target)
        self.sim.data.set_mocap_quat('robot1:mocap', gripper1_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_box_xpos = self.sim.data.get_site_xpos('box').copy()
        self.initial_gripper_xpos[0] = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.initial_gripper_xpos[1] = self.sim.data.get_site_xpos('robot1:grip').copy()
        self.initial_gripper_quat[0] = rotations.mat2quat(self.sim.data.get_site_xmat('robot0:grip').copy())
        self.initial_gripper_quat[1] = rotations.mat2quat(self.sim.data.get_site_xmat('robot1:grip').copy())

    def render(self, mode='human', width=500, height=500):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = np.array((self._get_obs('0'), self._get_obs('1')))
        return obs

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
