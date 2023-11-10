# scene_objects.py
"""
Definition of objects that can build in the scene

Created on 2023/3/14
@author: Pin-Yun Hung
"""
import copy
from dataclasses import dataclass
from typing import List
import numpy as np
import math

############################
# Obstacle Definition
############################


@dataclass
class Lidar:
    radius: float = 1.5    # lidar scanning radius


@dataclass
class CirObstacles:
    obstacle_num: int
    pos: List[np.ndarray] = None    # center of circle
    radius: List[float] = None


@dataclass
class RectObstacles:
    obstacle_num: int
    pos: List[np.ndarray] = None  # vertex


############################
# Agent Definition
############################


@dataclass
class SingleAgent(object):
    def __init__(self, initial_state, target_state, safety_dis, Q, R, Dis_weight, Back_weight):
        """
        Build a Single agent, which is shape as a circle.


        Assume the agent is a unicycle with circle shape. State includes
        2D position (x, y), heading angle (theta) and velocity (v), and
        input includes acceleration (a) and angular velocity (w). State,
        input and the dynamic fxn can simply represent by following,

        state = [x, y, theta, v]
        input = [a, w]

        Dynamic fxn :
            x_dot = v * cos(theta)
            y_dot = v * sin(theta)
            theta_dot = w
            v_dot = a

        ------------------------------------------------------------------
        PreConditions : None
        PostConditions : A single agent with given properties

        @param initial_state : [x, y, theta, v]
        @param  target_state : [x, y, theta, v]
        @param  safety_dis : The safety distance that agent want to keep with other agents. Diameter of circle.
        @param           Q : penalty weight of state difference
        @param           R : penalty weight of control input
        @param  Dis_weight : penalty weight for keeping the safety distance
        @param Back_weight : penalty weight of backing up
        """
        # agent size
        self.radius = 0.4

        # state & input properties
        # state = [x, y, theta, v]
        # input = [a, w]
        self.state_dim = 4
        self.input_dim = 2
        self.position = initial_state   # current state
        self.next_step_prediction = self.position   # next state
        self.control_input = np.array([0, 0])   # current input
        self.control_opt_series = None  # optimized input series used on next time step iLQR initial guess

        # To store real input series and trajectory (state)
        self.control_series = [self.control_input]
        self.trajectory = []

        # for non-cooperated case, store input series and trajectory
        self.pred_non_cooperated_traj: np.ndarray
        self.pred_non_cooperated_input: np.ndarray

        # velocity limit (m/sec)
        self.vmin = -10
        self.vmax = 10

        # input limit [a, w] (m/sec, rad/sec)
        self.umin = [-50, -np.pi / 2]
        self.umax = [50, np.pi / 2]

        # sampling time (All agents must have same sampling time)
        self.Ts = 0.1

        # initial / goal state & input
        self.initial_state = initial_state
        self.target_state = target_state
        self.input_ref = np.array([0, 0])

        # safety_distance
        self.safety_dis = safety_dis

        # cost parameters, the penalty weight
        self.Q_cost = Q
        self.R_cost = R
        self.D_cost = Dis_weight
        self.B_cost = Back_weight

        # To store whole solution of iLQR
        self.sol_series = tuple()

        # predict next state
        self.predict_next_state()

        # solver feasibility
        self.feasibility = False

        # parameter for checking stuck
        self.stuck_time_threshold = 8
        self.last_stuck_time_step = 0
        self.org_safety_dis = safety_dis
        self.stuck_agent_id = -1
        self.safety_dis_is_updated = False

    def update(self, sol, agent_id):

        u_opt = sol[1]
        n2 = agent_id * self.input_dim
        self.control_opt_series = u_opt

        # clip acceleration & angular velocity to satisfy the limitation
        a_opt = u_opt[0, n2]
        w_opt = u_opt[0, n2 + 1]
        a_opt = np.clip(a_opt, self.umin[0], self.umax[0])
        w_opt = np.clip(w_opt, self.umin[1], self.umax[1])

        # store input, state, iLQR solution
        self.control_input = np.array([a_opt, w_opt])
        self.trajectory.append(self.position)
        self.control_series.append(self.control_input)
        sol_xu = tuple()
        sol_xu = sol_xu + (sol[0], )
        sol_xu = sol_xu + (sol[1], )
        self.sol_series = self.sol_series + (sol_xu,)

        # get next state
        self.predict_next_state()

    def update_non_cooperated_traj(self, sol):
        self.pred_non_cooperated_traj = sol[0]
        self.pred_non_cooperated_input = sol[1]

    def rebuild_control_opt_series(self, agent_id):
        n1 = agent_id * self.input_dim
        if np.size(self.control_opt_series, 1) > self.input_dim:
            self.control_opt_series = self.control_opt_series[:, n1:n1 + self.input_dim]

    def rebuild_control_opt_series_ref(self, u_ref, agent_id):
        n1 = agent_id * self.input_dim
        self.control_opt_series = self.control_opt_series.at[:, n1:n1 + self.input_dim].set(u_ref)

    def rebuild_sense_control_opt_series(self, sense_list, N):
        agent_num = len(sense_list)
        count = 0
        U = copy.deepcopy(self.control_opt_series)
        self.control_opt_series = np.zeros((N, agent_num * self.input_dim))

        for i in sense_list:
            n0 = count * self.input_dim
            n1 = i * self.input_dim
            self.control_opt_series[:, n0:n0 + self.input_dim] = U[:, n1:n1 + self.input_dim]
            count += 1

    def predict_next_state(self):
        position = self.position

        # dynamic fxn
        x = position[0] + self.Ts * position[3] * np.cos(position[2])
        y = position[1] + self.Ts * position[3] * np.sin(position[2])
        theta = position[2] + self.Ts * self.control_input[1]
        v = position[3] + self.Ts * self.control_input[0]

        # Clip velocity to satisfy limitation
        v = np.clip(v, self.vmin, self.vmax)

        # update next state
        self.next_step_prediction = np.array([x, y, theta, v])

    def check_getting_stuck(self, time_step, sense_list, new_agents_tmp):
        if (time_step - self.last_stuck_time_step) >= self.stuck_time_threshold:
            vel_list = np.zeros(len(sense_list))
            angular_vel_list = np.zeros(len(sense_list))
            updated_safety_dis = 100000
            traj = np.asarray(self.trajectory)

            current_p = [self.position[0], self.position[1]]
            start_p = [traj[time_step - self.stuck_time_threshold, 0], traj[time_step - self.stuck_time_threshold, 1]]
            dis_diff = math.dist(start_p, current_p)

            for i in range(1, len(sense_list)):
                n1 = i * self.state_dim
                others_current_pos = new_agents_tmp.agent[i].position[:2]
                others_previous_pos = self.sol_series[time_step - self.stuck_time_threshold][0][0][n1:n1 + 2]

                dis_diff = dis_diff + math.dist(others_current_pos, others_previous_pos)
                new_safety_dis = math.dist(current_p, others_current_pos) + self.radius * 2

                if new_safety_dis < updated_safety_dis:
                    self.stuck_agent_id = sense_list[i]
                    updated_safety_dis = new_safety_dis

                angular_vel_list[i] = abs(new_agents_tmp.agent[i].control_input[1])
                vel_list[i] = abs(new_agents_tmp.agent[i].position[3])

            if len(sense_list) > 1 and dis_diff < 0.1 and np.all(angular_vel_list < 0.1) and np.all(vel_list < 0.1):
                self.safety_dis = updated_safety_dis
                self.safety_dis_is_updated = True
            else:
                if self.safety_dis != self.org_safety_dis and self.stuck_agent_id not in sense_list:
                    self.safety_dis = self.org_safety_dis
                    self.stuck_agent_id = -1
                    self.safety_dis_is_updated = True

            self.last_stuck_time_step = time_step


@dataclass
class MultiAgents(object):
    def __init__(self, agents, agent_num):
        """
        Build multi-agents.


        Multi-agents can store all agents and the agent
        number decides how many agents will exist in the
        scene. Note that the agent always picks from the
        beginning. For example,

        agent = [a1, a2, a3, a4]
        agent_num = 2
        existed agent = a1, a2

        agent = [a3, a2, a1, a4]
        agent_num = 2
        existed agent = a3, a2

        -------------------------------------------------
        PreConditions : at least one agent exist
        PostConditions : Multi-agent

        @param    agents : [SingleAgent_1, SingleAgent_2, ...]
        @param agent_num : existed agent number
        """
        self.agent: List[SingleAgent] = agents
        self.agent_num: int = agent_num

        # All agents have same state & input definition, and same sampling time
        self.state_dim: int = 4
        self.input_dim: int = 2
        self.Ts: float = self.agent[0].Ts

    def update_limit(self, v_limit, a_limit, w_limit):
        for agent in self.agent:
            agent.vmin = v_limit[0]
            agent.vmax = v_limit[1]
            agent.umin[0] = a_limit[0]
            agent.umax[0] = a_limit[1]
            agent.umin[1] = w_limit[0]
            agent.umax[1] = w_limit[1]
