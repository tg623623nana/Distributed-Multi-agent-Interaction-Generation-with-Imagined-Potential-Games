# draw_functions.py
"""
Functions for plotting

Created on 2023/10/12
@author: Pin-Yun Hung
"""
import os
from scene_objects import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.legend_handler import HandlerTuple
from moviepy.editor import VideoFileClip
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D

pos_type = 'o'
goal_type = 'x'
traj_type = '.-'
pasttraj_type = '-'
predtraj_type = '--'
max_predict_agent_num = 3


class DrawClass:
    def load_info(self, folder_name):

        info_file = folder_name + "info.npz"
        info = np.load(info_file, mmap_mode='r', allow_pickle=True)

        # # show all keys
        # for k in info.files:
        #     print(k)

        agent_num = info['agent_num']
        state_dim = info['state_dim']
        input_dim = info['input_dim']
        Ts = info['Ts']
        agent_radius = info['agent_radius']
        initial_state = info['initial_state']
        target_state = info['target_state']
        vmin = info['vmin']
        vmax = info['vmax']
        umin = info['umin']
        umax = info['umax']
        safety_dis = info['safety_dis']
        Q_cost = info['Q_cost']
        R_cost = info['R_cost']
        D_cost = info['D_cost']
        B_cost = info['B_cost']
        boundary = info['boundary']
        boundary_range = info['boundary_range']  # random setting range
        circle_obs_num = info['circle_obs_num']
        circle_obs_pos = info['circle_obs_pos']
        circle_obs_radius = info['circle_obs_radius']
        horizon = info['horizon']
        solver_type = info['solver_type']
        # constraints_threshold = info['constraints_threshold']
        adaptive_backup_weight = None
        agents_with_games = None
        if 'adaptive_backup_weight' in info.files:
            adaptive_backup_weight = info['adaptive_backup_weight']
        if 'agents_with_games' in info.files:
            agents_with_games = info['agents_with_games']

        # rebuild multi-agents
        agent_list = []
        if agent_num > 1:
            agent = SingleAgent(
                initial_state=initial_state[0],
                target_state=target_state[0],
                Q=Q_cost[0],
                R=R_cost[0],
                Dis_weight=D_cost[0][0],
                Back_weight=B_cost[0][0],
                safety_dis=safety_dis[0][0])

            agent.state_dim = state_dim
            agent.input_dim = input_dim
            agent.Ts = Ts
            agent.radius = agent_radius[0][0]
            agent.vmin = vmin[0][0]
            agent.vmax = vmax[0][0]
            agent.umin = umin[0]
            agent.umax = umax[0]

        else:
            agent = SingleAgent(
                initial_state=initial_state,
                target_state=target_state,
                Q=Q_cost,
                R=R_cost,
                Dis_weight=D_cost,
                Back_weight=B_cost,
                safety_dis=safety_dis)

            agent.state_dim = state_dim
            agent.input_dim = input_dim
            agent.Ts = Ts
            agent.radius = agent_radius
            agent.vmin = vmin
            agent.vmax = vmax
            agent.umin = umin
            agent.umax = umax

        agent_list.append(agent)

        for i in range(1, agent_num):
            agent = SingleAgent(
                initial_state=initial_state[i],
                target_state=target_state[i],
                Q=Q_cost[i],
                R=R_cost[i],
                Dis_weight=D_cost[i][0],
                Back_weight=B_cost[i][0],
                safety_dis=safety_dis[i][0])

            agent.state_dim = state_dim
            agent.input_dim = input_dim
            agent.Ts = Ts
            agent.radius = agent_radius[i][0]
            agent.vmin = vmin[i][0]
            agent.vmax = vmax[i][0]
            agent.umin = umin[i]
            agent.umax = umax[i]

            agent_list.append(agent)

        circle_obs = CirObstacles(obstacle_num=circle_obs_num,
                                  pos=circle_obs_pos,
                                  radius=circle_obs_radius)

        agents = MultiAgents(agent_list, agent_num)

        if len(np.shape(circle_obs.obstacle_num)) > 0:
            circle_obs.obstacle_num = circle_obs.obstacle_num[0]
            circle_obs.pos = circle_obs.pos[0]
        else:
            circle_obs = None

        hetero_solvers = False
        if agents_with_games is not None:
            hetero_solvers = dict()
            for i in range(np.size(agents_with_games, 0)):
                key = int(agents_with_games[i, 0])
                hetero_solvers[key] = agents_with_games[i, 1]

        return agents, circle_obs, boundary, horizon, boundary_range, solver_type, adaptive_backup_weight, hetero_solvers

    def read_data(self, file_name, agent_num, state_dim, input_dim):
        x_raw = []
        y_raw = []
        theta_raw = []
        v_raw = []
        a_raw = []
        w_raw = []

        with open(file_name, 'r') as f:
            for line in f.readlines():
                l = line.strip().split(' ')
                count = 0
                for i in range(agent_num):
                    n1 = i * 4
                    x_raw.append(l[n1])
                    y_raw.append(l[n1 + 1])
                    theta_raw.append(l[n1 + 2])
                    v_raw.append(l[n1 + 3])
                    count += 4
                for i in range(agent_num):
                    n1 = count + i * 2
                    a_raw.append(l[n1])
                    w_raw.append(l[n1 + 1])
        N = round(len(x_raw) / agent_num)
        x_opt = np.zeros((N, state_dim * agent_num))
        u_opt = np.zeros((N, input_dim * agent_num))

        for i in range(N):
            for j in range(agent_num):

                x_opt[i, j * state_dim] = x_raw[i * agent_num + j]
                x_opt[i, j * state_dim + 1] = y_raw[i * agent_num + j]
                x_opt[i, j * state_dim + 2] = theta_raw[i * agent_num + j]
                x_opt[i, j * state_dim + 3] = v_raw[i * agent_num + j]

                u_opt[i, j * input_dim] = a_raw[i * agent_num + j]
                u_opt[i, j * input_dim + 1] = w_raw[i * agent_num + j]

        f.close()

        return x_opt, u_opt

    def read_data_npz(self, file_name):

        info = np.load(file_name, mmap_mode='r', allow_pickle=True)

        x_opt = info['X']
        u_opt = info['U']

        return x_opt, u_opt

    def load_reactive_safety_distance_info(self, folder_name):

        info_file = folder_name + "safety_dis.npz"
        info = np.load(info_file, mmap_mode='r', allow_pickle=True)

        reactive_safety_dis_info = info['time_step_and_safety_dis']

        time_step_and_safety_dis = {}
        for i in range(np.size(reactive_safety_dis_info, 0)):
            time_step_and_safety_dis[int(reactive_safety_dis_info[i, 0])] = np.array([reactive_safety_dis_info[i, 1], reactive_safety_dis_info[i, 2]])

        return time_step_and_safety_dis

    def load_solver_cost_time_info(self, folder_name):

        solved_time_series = None

        for fold_path in os.listdir(folder_name):
            # check if current path is a file
            if os.path.isfile(os.path.join(folder_name, fold_path)):
                file_string = fold_path.split('.')
                file_name = file_string[0]

                if file_name == "solver_cost_time":
                    info_file = folder_name + "solver_cost_time.npz"
                    info = np.load(info_file, mmap_mode='r', allow_pickle=True)

                    solved_time_series = info['solver_cost_time']

        return solved_time_series

    def rebuild_agents(self, folder_name, total_num, agents, boundary, circle_obs):

        ########################################################
        # rebuild agents (traj, prediction, other prediction)
        ########################################################

        # get horizon
        folder = "figs/" + folder_name
        agent_file_name = folder + "agent1_traj0.npz"
        n = agents.state_dim
        m = agents.input_dim
        x_opt, u_opt = self.read_data_npz(agent_file_name)
        N = np.size(x_opt, 0) - 1
        plot_boundary = [1000000, 1000000, -1000000, -1000000]  # xmin, ymin, xmax, ymax

        for k in range(total_num):
            for i in range(agents.agent_num):
                agent_file_name = folder + "agent" + str(i + 1) + "_traj" + str(k) + ".npz"
                x_opt, u_opt = self.read_data_npz(agent_file_name)

                sol = tuple()
                sol = sol + (x_opt, )
                sol = sol + (u_opt, )

                agents.agent[i].update(sol, i)
                agents.agent[i].position = agents.agent[i].next_step_prediction

                plot_boundary = self.update_plot_boundary(plot_boundary, agents, sol[0])

        plot_boundary = self.update_boundary_with_obstacle(plot_boundary, circle_obs)

        safety_dis = agents.agent[0].safety_dis / 2
        target_xy = agents.agent[0].target_state
        xf_min = target_xy[0] - safety_dis
        xf_max = target_xy[0] + safety_dis
        yf_min = target_xy[1] - safety_dis
        yf_max = target_xy[1] + safety_dis
        for i in range(1, agents.agent_num):
            safety_dis = agents.agent[i].safety_dis / 2
            target_xy = agents.agent[i].target_state
            xf_min = min(xf_min, target_xy[0] - safety_dis)
            xf_max = max(xf_max, target_xy[0] + safety_dis)
            yf_min = min(yf_min, target_xy[1] - safety_dis)
            yf_max = max(yf_max, target_xy[1] + safety_dis)

        if plot_boundary[0] > xf_min:
            plot_boundary[0] = xf_min
        if plot_boundary[1] > yf_min:
            plot_boundary[1] = yf_min
        if plot_boundary[2] < xf_max:
            plot_boundary[2] = xf_max
        if plot_boundary[3] < yf_max:
            plot_boundary[3] = yf_max

        return agents, plot_boundary

    def rebuild_agents_txt(self, folder_name, total_num, agents, boundary, circle_obs):

        ########################################################
        # rebuild agents (traj, prediction, other prediction)
        ########################################################

        # get horizon
        folder = "figs/" + folder_name
        agent_file_name = folder + "agent1_traj0.txt"
        n = agents.state_dim
        m = agents.input_dim
        x_opt, u_opt = self.read_data(agent_file_name, agents.agent_num, n, m)
        N = np.size(x_opt, 0) - 1
        plot_boundary = [1000000, 1000000, -1000000, -1000000]  # xmin, ymin, xmax, ymax

        for k in range(total_num):
            for i in range(agents.agent_num):
                agent_file_name = folder + "agent" + str(i + 1) + "_traj" + str(k) + ".txt"
                x_opt, u_opt = self.read_data(agent_file_name, agents.agent_num, n, m)

                sol = tuple()
                sol = sol + (x_opt, )
                sol = sol + (u_opt, )

                agents.agent[i].update(sol, i)
                agents.agent[i].position = agents.agent[i].next_step_prediction

                plot_boundary = self.update_plot_boundary(plot_boundary, agents, sol[0])

        plot_boundary = self.update_boundary_with_obstacle(plot_boundary, circle_obs)

        safety_dis = agents.agent[0].safety_dis / 2
        target_xy = agents.agent[0].target_state
        xf_min = target_xy[0] - safety_dis
        xf_max = target_xy[0] + safety_dis
        yf_min = target_xy[1] - safety_dis
        yf_max = target_xy[1] + safety_dis
        for i in range(1, agents.agent_num):
            safety_dis = agents.agent[i].safety_dis / 2
            target_xy = agents.agent[i].target_state
            xf_min = min(xf_min, target_xy[0] - safety_dis)
            xf_max = max(xf_max, target_xy[0] + safety_dis)
            yf_min = min(yf_min, target_xy[1] - safety_dis)
            yf_max = max(yf_max, target_xy[1] + safety_dis)

        if plot_boundary[0] > xf_min:
            plot_boundary[0] = xf_min
        if plot_boundary[1] > yf_min:
            plot_boundary[1] = yf_min
        if plot_boundary[2] < xf_max:
            plot_boundary[2] = xf_max
        if plot_boundary[3] < yf_max:
            plot_boundary[3] = yf_max

        return agents, plot_boundary

    ###############################
    # Draw a plot
    ###############################

    def draw_full_trajectory(self, subplot, agents, plot_boundary, boundary=None, circle_obs=None):
        """
        Draw the full closed-loop trajectory


        The full trajectory will not include the size and the safety distance
        of the agent, which makes the plot more simple than the one draw by fxn
        "draw_timestep_trajectory". The reason is to draw a clear image to show
        the full path of the multi-agents.

        -----------------------------------------------------------------
        PreConditions : Need to create a figure, add subplot to the figure, and assign the subplot as input
        PostConditions : draw the scene and the full trajectory of all the agents on the figure

        @param       subplot:
        @param        agents: all agents in the scene
        @param plot_boundary: the boundary of the plot
        @param      boundary: the real 2D x-y boundary of the scene
        @param    circle_obs: circle obstacle
        """

        ax = subplot

        # create color list
        self_color, other_color = self.generate_color_list(agents.agent_num)

        pos_type = 'o'
        goal_type = 'x'
        traj_type = '.-'
        pasttraj_type = '-'
        predtraj_type = '--'

        # draw full trajectory
        for i in range(agents.agent_num):
            x0 = agents.agent[i].initial_state[0]
            y0 = agents.agent[i].initial_state[1]
            xf = agents.agent[i].target_state[0]
            yf = agents.agent[i].target_state[1]

            # plot the initial state & target state
            plt.plot(x0, y0, pos_type, color=self_color[i], markerfacecolor='None')
            plt.plot(xf, yf, goal_type, color=self_color[i])

            # draw trajectory
            traj = np.asarray(agents.agent[i].trajectory)
            plt.plot(traj[:, 0], traj[:, 1], traj_type, color=self_color[i])

        # draw circle obstacle
        if circle_obs is not None:
            for i in range(circle_obs.obstacle_num):
                ax.add_patch(
                    Circle(xy=(circle_obs.pos[i][0], circle_obs.pos[i][1]), radius=circle_obs.radius[i], alpha=0.5,
                           color='k'))

        # draw boundary
        if boundary is not None:
            plt.plot([boundary[0], boundary[2]], [boundary[1], boundary[1]], 'k')
            plt.plot([boundary[0], boundary[2]], [boundary[3], boundary[3]], 'k')
            plt.plot([boundary[0], boundary[0]], [boundary[1], boundary[3]], 'k')
            plt.plot([boundary[2], boundary[2]], [boundary[1], boundary[3]], 'k')

            if plot_boundary[0] - 0.2 < boundary[0]:
                plot_boundary[0] = boundary[0] + 0.2
            if plot_boundary[1] - 0.2 < boundary[1]:
                plot_boundary[1] = boundary[1] + 0.2
            if plot_boundary[2] + 0.2 > boundary[2]:
                plot_boundary[2] = boundary[2] - 0.2
            if plot_boundary[3] + 0.2 > boundary[3]:
                plot_boundary[3] = boundary[3] - 0.2

        # limit the plotting boundary & adjust the height-weight ratio of the plot
        plt.xlim([plot_boundary[0] - 0.2, plot_boundary[2] + 0.2])
        plt.ylim([plot_boundary[1] - 0.2, plot_boundary[3] + 0.2])
        ax.set_aspect('equal', adjustable='box')
        # plt.show()

    def draw_part_trajectory(self, subplot, agents, plot_boundary, sim_time_range, agent_1_color_list, agent_2_color_list, boundary=None, circle_obs=None, predict_other=True):
        """
        Draw part of the closed-loop trajectory

        -----------------------------------------------------------------
        PreConditions : Need to create a figure, add subplot to the figure, and assign the subplot as input
        PostConditions :

        @param       subplot:
        @param        agents: all agents in the scene
        @param plot_boundary: the boundary of the plot
        @param      boundary: the real 2D x-y boundary of the scene
        @param    circle_obs: circle obstacle
        """

        ax = subplot
        end_time = sim_time_range[-1]
        # agent_1_color_list = sns.color_palette("Blues", len(sim_time_range))
        # agent_2_color_list = sns.color_palette("Greens", len(sim_time_range))

        # create color list
        self_color, other_color = self.generate_color_list(agents.agent_num)

        pos_type = 'o'
        goal_type = 'x'
        traj_type = '.-'
        pasttraj_type = '-'
        predtraj_type = '--'

        # find agent that have the smallest safety distance
        agent_id = 0
        for i in range(1, agents.agent_num):
            if agents.agent[i].safety_dis < agents.agent[agent_id].safety_dis:
                agent_id = i

        # draw part of the trajectory
        count = 0
        for k in sim_time_range:

            for i in range(agents.agent_num):
                n1 = i * agents.state_dim

                # draw current position (include size and safety distance)
                traj = np.asarray(agents.agent[i].trajectory)
                x = traj[k, 0]
                y = traj[k, 1]
                ax.plot(x, y, traj_type, color=self_color[i], markersize='0.5')
                # ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].safety_dis / 2, alpha=0.1, color=self_color[i]))

                if i == 0:
                    circle_color = agent_1_color_list[count]
                else:
                    circle_color = agent_2_color_list[count]

                ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].radius, alpha=0.5, color=circle_color))

                # plot the car direction
                # agent_radius = agents.agent[i].radius
                # if agent_radius == 0:
                #     agent_radius = agents.agent[agent_id].safety_dis / 2
                # agent_front_x = agent_radius * np.cos(traj[k, 2])
                # agent_front_y = agent_radius * np.sin(traj[k, 2])
                # ax.plot([x, x + agent_front_x], [y, y + agent_front_y], '-k')

            count += 1

        # draw full trajectory
        for i in range(agents.agent_num):
            n1 = i * agents.state_dim

            x0 = agents.agent[i].initial_state[0]
            y0 = agents.agent[i].initial_state[1]
            xf = agents.agent[i].target_state[0]
            yf = agents.agent[i].target_state[1]

            # plot the initial state & target state
            ax.plot(x0, y0, pos_type, color=self_color[i], markerfacecolor='None')
            ax.plot(xf, yf, goal_type, color=self_color[i])

            # draw trajectory
            traj = np.asarray(agents.agent[i].trajectory)
            x = traj[end_time, 0]
            y = traj[end_time, 1]
            ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].safety_dis / 2, alpha=0.1, color=self_color[i]))
            ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].radius, alpha=0.3, color=self_color[i]))
            ax.plot(traj[:end_time, 0], traj[:end_time, 1], pasttraj_type, color=self_color[i], zorder=0)

            # plot the car direction
            agent_radius = agents.agent[i].radius
            if agent_radius == 0:
                agent_radius = agents.agent[agent_id].safety_dis / 2
            agent_front_x = agent_radius * np.cos(traj[end_time, 2])
            agent_front_y = agent_radius * np.sin(traj[end_time, 2])
            ax.plot([x, x + agent_front_x], [y, y + agent_front_y], '-k', zorder=15)


            # plot the prediction of self-trajectory
            ax.plot(agents.agent[i].sol_series[end_time][0][:, n1], agents.agent[i].sol_series[end_time][0][:, n1 + 1],
                    predtraj_type, color=self_color[i], zorder=10)

            # plot the prediction of other-trajectory if agent number less than three
            if predict_other == True:
                for j in range(agents.agent_num):
                    if j != i and agents.agent_num < max_predict_agent_num:
                        nj = j * agents.state_dim
                        ax.plot(agents.agent[i].sol_series[end_time][0][:, nj],
                                agents.agent[i].sol_series[end_time][0][:, nj + 1],
                                predtraj_type, color=other_color[i], zorder=5)



        # draw circle obstacle
        if circle_obs is not None:
            for i in range(circle_obs.obstacle_num):
                ax.add_patch(
                    Circle(xy=(circle_obs.pos[i][0], circle_obs.pos[i][1]), radius=circle_obs.radius[i], alpha=0.5,
                           color='k'))
                plot_boundary = self.update_boundary_with_obstacle(plot_boundary, circle_obs)

        # draw boundary
        if boundary is not None:
            plt.plot([boundary[0], boundary[2]], [boundary[1], boundary[1]], 'k')
            plt.plot([boundary[0], boundary[2]], [boundary[3], boundary[3]], 'k')
            plt.plot([boundary[0], boundary[0]], [boundary[1], boundary[3]], 'k')
            plt.plot([boundary[2], boundary[2]], [boundary[1], boundary[3]], 'k')

            if plot_boundary[0] - 0.2 < boundary[0]:
                plot_boundary[0] = boundary[0] + 0.2
            if plot_boundary[1] - 0.2 < boundary[1]:
                plot_boundary[1] = boundary[1] + 0.2
            if plot_boundary[2] + 0.2 > boundary[2]:
                plot_boundary[2] = boundary[2] - 0.2
            if plot_boundary[3] + 0.2 > boundary[3]:
                plot_boundary[3] = boundary[3] - 0.2

        # limit the plotting boundary & adjust the height-weight ratio of the plot
        # plt.xlim([plot_boundary[0] - 0.2, plot_boundary[2] + 0.2])
        # plt.ylim([plot_boundary[1] - 0.2, plot_boundary[3] + 0.2])
        plt.xlim([0.5, 7.5])
        plt.ylim([-2.0, 2.0])
        ax.set_aspect('equal', adjustable='box')
        # plt.show()

    def draw_timestep_trajectory(self, subplot, agents, plot_boundary, sim_num, boundary=None, circle_obs=None,
                                 rect_obs=None,
                                 predict_other=True):
        """
        Draw the closed-loop trajectory at the given time step

        -----------------------------------------------------------------
        PreConditions : Need to create a figure, add subplot to the figure, and assign the subplot as input
        PostConditions : draw the scene and the trajectory of all the agents at given time step on the figure

        @param       subplot:
        @param        agents: all agents in the scene
        @param plot_boundary: the boundary of the plot
        @param       sim_num: the time step that want to draw,
                              for example, the trajectory includes 10 time steps, then you can choose 0-9
        @param      boundary: the real 2D x-y boundary of the scene
        @param    circle_obs: circle obstacle
        @param predict_other: bool, if true, then the prediction of other-trajectory
                              will be drawn if agent number is less than three
        """
        ax = subplot
        k = sim_num

        # create color list
        self_color, other_color = self.generate_color_list(agents.agent_num)

        pos_type = 'o'
        goal_type = 'x'
        traj_type = '.-'
        pasttraj_type = '-'
        predtraj_type = '--'

        # find agent that have the smallest safety distance
        agent_id = 0
        for i in range(1, agents.agent_num):
            if agents.agent[i].safety_dis < agents.agent[agent_id].safety_dis:
                agent_id = i

        # draw trajectory
        for i in range(agents.agent_num):
            n1 = i * agents.state_dim

            x0 = agents.agent[i].initial_state[0]
            y0 = agents.agent[i].initial_state[1]
            xf = agents.agent[i].target_state[0]
            yf = agents.agent[i].target_state[1]

            # plot the initial state & target state
            ax.plot(x0, y0, pos_type, color=self_color[i], markerfacecolor='None')
            ax.plot(xf, yf, goal_type, color=self_color[i])

            # draw current position (include size and safety distance)
            traj = np.asarray(agents.agent[i].trajectory)
            x = traj[k, 0]
            y = traj[k, 1]
            ax.plot(x, y, traj_type, color=self_color[i], markersize='0.5')
            ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].safety_dis / 2, alpha=0.1, color=self_color[i]))
            ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].radius, alpha=0.3, color=self_color[i]))

            # plot the car direction
            agent_radius = agents.agent[i].radius
            if agent_radius == 0:
                agent_radius = agents.agent[agent_id].safety_dis / 2
            agent_front_x = agent_radius * np.cos(traj[k, 2])
            agent_front_y = agent_radius * np.sin(traj[k, 2])
            ax.plot([x, x + agent_front_x], [y, y + agent_front_y], '-k')

            # plot the past trajectory
            ax.plot(traj[:k, 0], traj[:k, 1], pasttraj_type, color=self_color[i], zorder=0)

            # plot the prediction of self-trajectory
            ax.plot(agents.agent[i].sol_series[k][0][:, n1], agents.agent[i].sol_series[k][0][:, n1 + 1],
                    predtraj_type, color=self_color[i], zorder=10)

            # plot the prediction of other-trajectory if agent number less than three
            if predict_other == True:
                for j in range(agents.agent_num):
                    if j != i and agents.agent_num < max_predict_agent_num:
                        nj = j * agents.state_dim
                        ax.plot(agents.agent[i].sol_series[k][0][:, nj], agents.agent[i].sol_series[k][0][:, nj + 1],
                                predtraj_type, color=other_color[i], zorder=5)

        # plot the circle obstacle
        if circle_obs is not None:
            for i in range(circle_obs.obstacle_num):
                ax.add_patch(
                    Circle(xy=(circle_obs.pos[i][0], circle_obs.pos[i][1]), radius=circle_obs.radius[i], alpha=0.5,
                           color='k'))
                plot_boundary = self.update_boundary_with_obstacle(plot_boundary, circle_obs)

        if rect_obs is not None:
            for i in range(rect_obs.obstacle_num):
                xy = rect_obs.pos[i][0]
                w = rect_obs.pos[i][2][0] - rect_obs.pos[i][0][0]
                h = rect_obs.pos[i][2][1] - rect_obs.pos[i][0][1]
                ax.add_patch(Rectangle(xy=xy, width=w, height=h, alpha=0.5, color='k'))
                plot_boundary = self.update_boundary_with_rect_obstacle(plot_boundary, rect_obs)

        # draw real boundary
        if boundary is not None:
            ax.plot([boundary[0], boundary[2]], [boundary[1], boundary[1]], 'k')
            ax.plot([boundary[0], boundary[2]], [boundary[3], boundary[3]], 'k')
            ax.plot([boundary[0], boundary[0]], [boundary[1], boundary[3]], 'k')
            ax.plot([boundary[2], boundary[2]], [boundary[1], boundary[3]], 'k')

            if plot_boundary[0] - 0.2 < boundary[0]:
                plot_boundary[0] = boundary[0] + 0.2
            if plot_boundary[1] - 0.2 < boundary[1]:
                plot_boundary[1] = boundary[1] + 0.2
            if plot_boundary[2] + 0.2 > boundary[2]:
                plot_boundary[2] = boundary[2] - 0.2
            if plot_boundary[3] + 0.2 > boundary[3]:
                plot_boundary[3] = boundary[3] - 0.2

        # For T-intersection Case
        # ax.plot([-1.5, -1.5], [-2.0, 2.2], 'k')
        # ax.plot([8.5, 8.5], [-2.0, 2.2], 'k')
        # ax.add_patch(Rectangle((-1.5, -2.0), 10.0, 0.5, linewidth=1, alpha=0.5, edgecolor='k', facecolor='k'))

        # limit the plotting boundary & adjust the height-weight ratio of the plot
        ax.set_xlim([plot_boundary[0] - 0.2, plot_boundary[2] + 0.2])
        ax.set_ylim([plot_boundary[1] - 0.2, plot_boundary[3] + 0.2])
        # ax.set_xlim([-1.5, 8.5])
        # ax.set_ylim([-2.0, 2.2])
        # ax.set_xlim([0.5, 7.5])
        # ax.set_ylim([-2.0, 2.0])

        ax.set_aspect('equal', adjustable='box')
        # plt.show()

    def draw_timestep_trajectory2(self, subplot, agents, plot_boundary, sim_num, boundary=None, circle_obs=None,
                                  rect_obs=None,
                                 predict_other=True):
        """
        Draw the closed-loop trajectory at the given time step

        -----------------------------------------------------------------
        PreConditions : Need to create a figure, add subplot to the figure, and assign the subplot as input
        PostConditions : draw the scene and the trajectory of all the agents at given time step on the figure

        @param       subplot:
        @param        agents: all agents in the scene
        @param plot_boundary: the boundary of the plot
        @param       sim_num: the time step that want to draw,
                              for example, the trajectory includes 10 time steps, then you can choose 0-9
        @param      boundary: the real 2D x-y boundary of the scene
        @param    circle_obs: circle obstacle
        @param predict_other: bool, if true, then the prediction of other-trajectory
                              will be drawn if agent number is less than three
        """
        ax = subplot
        k = sim_num
        nl = 7

        # create color list
        self_color, other_color = self.generate_color_list(agents.agent_num)

        pos_type = 'o'
        goal_type = 'x'
        traj_type = '.-'
        pasttraj_type = '-'
        predtraj_type = '--'

        # find agent that have the smallest safety distance
        agent_id = 0
        for i in range(1, agents.agent_num):
            if agents.agent[i].safety_dis < agents.agent[agent_id].safety_dis:
                agent_id = i

        # draw trajectory
        for i in range(agents.agent_num):
            n1 = i * agents.state_dim
            n2 = k + 1

            x0 = agents.agent[i].initial_state[0]
            y0 = agents.agent[i].initial_state[1]
            xf = agents.agent[i].target_state[0]
            yf = agents.agent[i].target_state[1]

            # plot the initial state & target state
            ax.plot(x0, y0, pos_type, color=self_color[i], markerfacecolor='None', label='start')  # label 1
            ax.plot(xf, yf, goal_type, color=self_color[i], label='goal')  # label 2

            # draw current position (include size and safety distance)
            traj = np.asarray(agents.agent[i].trajectory)
            x = traj[k, 0]
            y = traj[k, 1]
            ax.plot(x, y, traj_type, color=self_color[i], markersize='0.5')
            ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].safety_dis / 2, alpha=0.1, color=self_color[i],
                                label='safety circle'))  # label 3
            if agents.agent[i].radius > 0:
                ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].radius, alpha=0.3, color=self_color[i],
                                    label='current position & agent size'))  # label 4
            else:
                ax.add_patch(Circle(xy=(x, y), radius=0.1, alpha=0.3, color=self_color[i],
                                    label='current position'))  # label 4

            # plot the car direction
            agent_radius = agents.agent[i].radius
            if agent_radius == 0:
                agent_radius = agents.agent[agent_id].safety_dis / 2
            agent_front_x = agent_radius * np.cos(traj[k, 2])
            agent_front_y = agent_radius * np.sin(traj[k, 2])
            ax.plot([x, x + agent_front_x], [y, y + agent_front_y], '-k', label='direction')  # label 5

            # plot the past trajectory
            ax.plot(traj[:k, 0], traj[:k, 1], pasttraj_type, color=self_color[i], label='trajectory')  # label 6  # , linewidth= 1

            # plot the prediction of self-trajectory
            ax.plot(agents.agent[i].sol_series[k][0][:, n1], agents.agent[i].sol_series[k][0][:, n1 + 1],
                     predtraj_type, color=self_color[i], zorder=10, label='plan')  # label 7  # , linewidth=1

            # plot the prediction of other-trajectory if agent number less than three
            if predict_other:
                for j in range(agents.agent_num):
                    if j != i and agents.agent_num < 3:
                        nj = j * agents.state_dim
                        ax.plot(agents.agent[i].sol_series[k][0][:, nj],
                                 agents.agent[i].sol_series[k][0][:, nj + 1], predtraj_type,
                                color=other_color[i], zorder=0,
                                label='cooperative prediction')  # label 8, 9  # , linewidth=1
                        nl += 1

        # plot the circle obstacle
        if circle_obs is not None:
            ax.add_patch(
                Circle(xy=(circle_obs.pos[0][0], circle_obs.pos[0][1]), radius=circle_obs.radius[0], alpha=0.5,
                       color='k', label='wall'))  # label 10
            plot_boundary = self.update_boundary_with_obstacle(plot_boundary, circle_obs)

            for i in range(1, circle_obs.obstacle_num):
                ax.add_patch(
                    Circle(xy=(circle_obs.pos[i][0], circle_obs.pos[i][1]), radius=circle_obs.radius[i], alpha=0.5,
                           color='k'))
                plot_boundary = self.update_boundary_with_obstacle(plot_boundary, circle_obs)

        if rect_obs is not None:
            for i in range(rect_obs.obstacle_num):
                xy = rect_obs.pos[i][0]
                w = rect_obs.pos[i][2][0] - rect_obs.pos[i][0][0]
                h = rect_obs.pos[i][2][1] - rect_obs.pos[i][0][1]
                ax.add_patch(Rectangle(xy=xy, width=w, height=h, alpha=0.5, color='k'))
                plot_boundary = self.update_boundary_with_rect_obstacle(plot_boundary, rect_obs)

        # draw real boundary
        if boundary is not None:
            ax.plot([boundary[0], boundary[2]], [boundary[1], boundary[1]], 'k')  # bottom
            ax.plot([boundary[0], boundary[2]], [boundary[3], boundary[3]], 'k')  # Top
            ax.plot([boundary[0], boundary[0]], [boundary[1], boundary[3]], 'k')  # left
            ax.plot([boundary[2], boundary[2]], [boundary[1], boundary[3]], 'k')  # right

            if plot_boundary[0] - 0.2 < boundary[0]:
                plot_boundary[0] = boundary[0] + 0.2
            if plot_boundary[1] - 0.2 < boundary[1]:
                plot_boundary[1] = boundary[1] + 0.2
            if plot_boundary[2] + 0.2 > boundary[2]:
                plot_boundary[2] = boundary[2] - 0.2
            if plot_boundary[3] + 0.2 > boundary[3]:
                plot_boundary[3] = boundary[3] - 0.2

        # For T-intersection Case
        # ax.plot([-1.5, -1.5], [-2.0, 2.2], 'k')
        # ax.plot([8.5, 8.5], [-2.0, 2.2], 'k')
        # ax.add_patch(Rectangle((-1.5, -2.0), 10.0, 0.5, linewidth=1, alpha=0.5, edgecolor='k', facecolor='k'))

        # limit the plotting boundary & adjust the height-weight ratio of the plot
        ax.set_xlim([plot_boundary[0] - 0.2, plot_boundary[2] + 0.2])
        ax.set_ylim([plot_boundary[1] - 0.2, plot_boundary[3] + 0.2])
        # ax.set_xlim([-1.5, 8.5])
        # ax.set_ylim([-2.0, 2.2])
        # ax.set_xlim([0.5, 7.5])
        # ax.set_ylim([-2.0, 2.0])
        ax.set_aspect('equal', adjustable='box')

        # Get & plot the legend
        handles, labels = ax.get_legend_handles_labels()
        if nl >= 8:
            nl = 8

        handle_list = []
        label_list = []
        for nl_index in range(nl):

            if nl_index == 4:  # < 6 or nl_index > 7
                handle_list.append((handles[nl_index]))
                label_list.append(labels[nl_index])
                continue

            handle_row = tuple()
            for i in range(agents.agent_num):
                handle_row = handle_row + (handles[nl * i + nl_index],)
            handle_list.append(handle_row)
            label_list.append(labels[nl_index])

        if circle_obs is not None:
            handle_list.append((handles[-1]))
            label_list.append(labels[-1])

        return handle_list, label_list

    def draw_prediction_animation(self, ax, agents, plot_boundary, k, sim_num, boundary=None, circle_obs=None, rect_obs=None,
                                  predict_other=True, show_full_predict=True):
        """
        Draw and save the animation of the trajectory and the prediction trajectory of others at the given time step


        While drawing the animation, if predict_other is "True", the prediction trajectory
        of others will include the predicted current position of other agents. The size
        and the safety distance will be plotted, which is different with the fxn "save_animate".

        Therefore, this animation can use to check the feasibility of the prediction (you
        can check whether the collision will happen or not).

        -------------------------------------------------------------------
        PreConditions :
        PostConditions :
            save the animation of the trajectory planning at given time step, the video
            type is "mp4".

        @param        agents: all agents in the scene
        @param plot_boundary: the boundary of the plot
        @param      boundary: the real 2D x-y boundary of the scene
        @param    circle_obs: circle obstacle
        @param predict_other: bool, if true, then the prediction of other-trajectory
                              will be drawn if agent number is less than three
        """

        # create color list
        self_color, other_color = self.generate_color_list(agents.agent_num)

        pos_type = 'o'
        goal_type = 'x'
        traj_type = '.-'
        pasttraj_type = '-'
        predtraj_type = '--'

        for i in range(agents.agent_num):
            n1 = i * agents.state_dim

            x0 = agents.agent[i].initial_state[0]
            y0 = agents.agent[i].initial_state[1]
            xf = agents.agent[i].target_state[0]
            yf = agents.agent[i].target_state[1]

            # plot the initial state & target state
            ax.plot(x0, y0, pos_type, color=self_color[i], markerfacecolor='None')
            ax.plot(xf, yf, goal_type, color=self_color[i])

            # draw current position (include size and safety distance)
            traj = agents.agent[i].sol_series[sim_num][0]

            x = traj[k, n1]
            y = traj[k, n1 + 1]
            # ax.plot(x, y, pos_type, color=self_color[i])
            ax.plot(x, y, traj_type, color=self_color[i], markersize='0.5')
            ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].safety_dis / 2, alpha=0.1, color=self_color[i]))
            ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].radius, alpha=0.3, color=self_color[i]))

            # plot the car direction
            agent_radius = agents.agent[i].radius
            if agent_radius == 0:
                agent_radius = agents.agent[i].safety_dis / 2
            agent_front_x = agent_radius * np.cos(traj[k, n1 + 2])
            agent_front_y = agent_radius * np.sin(traj[k, n1 + 2])
            ax.plot([x, x + agent_front_x], [y, y + agent_front_y], '-k')

            # plot past trajectory
            ax.plot(traj[k:, n1], traj[k:, n1 + 1], predtraj_type, color=self_color[i], zorder=10)
            ax.plot(traj[:k, n1], traj[:k, n1 + 1], pasttraj_type, color=self_color[i], zorder=0)
            past_traj = np.asarray(agents.agent[i].trajectory)
            ax.plot(past_traj[:sim_num + 1, 0], past_traj[:sim_num + 1, 1], pasttraj_type, color=self_color[i], zorder=0)

            # plot prediction trajectory of other
            if predict_other is True:
                for j in range(agents.agent_num):
                    if j != i and agents.agent_num < max_predict_agent_num:
                        nj = j * agents.state_dim
                        ax.plot(traj[k:, nj], traj[k:, nj + 1], predtraj_type, color=other_color[i], markersize=0.5, zorder=5)

                        if show_full_predict:
                            x_j = traj[k, nj]
                            y_j = traj[k, nj + 1]
                            ax.plot(x_j, y_j, pos_type, color=other_color[i], markersize=0.5)
                            ax.add_patch(
                                Circle(xy=(x_j, y_j), radius=agents.agent[i].safety_dis / 2, alpha=0.1,
                                       color=other_color[i]))
                            ax.add_patch(
                                Circle(xy=(x_j, y_j), radius=agents.agent[i].radius, alpha=0.3, color=other_color[i]))

        # draw circle obstacle
        if circle_obs is not None:
            for i in range(circle_obs.obstacle_num):
                ax.add_patch(
                    Circle(xy=(circle_obs.pos[i][0], circle_obs.pos[i][1]), radius=circle_obs.radius[i], alpha=0.5,
                           color='k'))

        if rect_obs is not None:
            for i in range(rect_obs.obstacle_num):
                xy = rect_obs.pos[i][0]
                w = rect_obs.pos[i][2][0] - rect_obs.pos[i][0][0]
                h = rect_obs.pos[i][2][1] - rect_obs.pos[i][0][1]
                ax.add_patch(Rectangle(xy=xy, width=w, height=h, alpha=0.5, color='k'))

        # draw boundary
        if boundary is not None:
            ax.plot([boundary[0], boundary[2]], [boundary[1], boundary[1]], 'k')
            ax.plot([boundary[0], boundary[2]], [boundary[3], boundary[3]], 'k')
            ax.plot([boundary[0], boundary[0]], [boundary[1], boundary[3]], 'k')
            ax.plot([boundary[2], boundary[2]], [boundary[1], boundary[3]], 'k')

            if plot_boundary[0] - 0.2 < boundary[0]:
                plot_boundary[0] = boundary[0] + 0.2
            if plot_boundary[1] - 0.2 < boundary[1]:
                plot_boundary[1] = boundary[1] + 0.2
            if plot_boundary[2] + 0.2 > boundary[2]:
                plot_boundary[2] = boundary[2] - 0.2
            if plot_boundary[3] + 0.2 > boundary[3]:
                plot_boundary[3] = boundary[3] - 0.2


        # For T-intersection Case
        # ax.plot([-1.5, -1.5], [-2.0, 2.2], 'k')
        # ax.plot([8.5, 8.5], [-2.0, 2.2], 'k')
        # ax.add_patch(Rectangle((-1.5, -2.0), 10.0, 0.5, linewidth=1, alpha=0.5, edgecolor='k', facecolor='k'))


        ax.set_xlim([plot_boundary[0] - 0.2, plot_boundary[2] + 0.2])
        ax.set_ylim([plot_boundary[1] - 0.2, plot_boundary[3] + 0.2])
        # ax.set_xlim([-1.5, 8.5])
        # ax.set_ylim([-2.0, 2.2])
        # ax.set_xlim([0.5, 7.5])
        # ax.set_ylim([-2.0, 2.0])
        ax.set_aspect('equal', adjustable='box')

    def draw_prediction_animation2(self, ax, agents, plot_boundary, k, sim_num, boundary=None, circle_obs=None, rect_obs=None, predict_other=True):
        """
        Draw and save the animation of the trajectory and the prediction trajectory of others at the given time step


        While drawing the animation, if predict_other is "True", the prediction trajectory
        of others will include the predicted current position of other agents. The size
        and the safety distance will be plotted, which is different with the fxn "save_animate".

        Therefore, this animation can use to check the feasibility of the prediction (you
        can check whether the collision will happen or not).

        -------------------------------------------------------------------
        PreConditions :
        PostConditions :
            save the animation of the trajectory planning at given time step, the video
            type is "mp4".

        @param        agents: all agents in the scene
        @param plot_boundary: the boundary of the plot
        @param      boundary: the real 2D x-y boundary of the scene
        @param    circle_obs: circle obstacle
        @param predict_other: bool, if true, then the prediction of other-trajectory
                              will be drawn if agent number is less than three
        """

        nl = 7

        # create color list
        self_color, other_color = self.generate_color_list(agents.agent_num)

        pos_type = 'o'
        goal_type = 'x'
        traj_type = '.-'
        pasttraj_type = '-'
        predtraj_type = '--'

        for i in range(agents.agent_num):
            n1 = i * agents.state_dim

            x0 = agents.agent[i].initial_state[0]
            y0 = agents.agent[i].initial_state[1]
            xf = agents.agent[i].target_state[0]
            yf = agents.agent[i].target_state[1]

            # plot the initial state & target state
            ax.plot(x0, y0, pos_type, color=self_color[i], markerfacecolor='None', label='start')  # label 1
            ax.plot(xf, yf, goal_type, color=self_color[i], label='goal')  # label 2

            # draw current position (include size and safety distance)
            traj = agents.agent[i].sol_series[sim_num][0]

            x = traj[k, n1]
            y = traj[k, n1 + 1]
            # ax.plot(x, y, pos_type, color=self_color[i])
            ax.plot(x, y, traj_type, color=self_color[i], markersize='0.5')
            ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].safety_dis / 2, alpha=0.1, color=self_color[i],
                                label='safety circle'))  # label 3
            if agents.agent[i].radius > 0:
                ax.add_patch(Circle(xy=(x, y), radius=agents.agent[i].radius, alpha=0.3, color=self_color[i],
                                    label='current position & agent size'))  # label 4
            else:
                ax.add_patch(Circle(xy=(x, y), radius=0.1, alpha=0.3, color=self_color[i],
                                    label='current position'))  # label 4

            # plot the car direction
            agent_radius = agents.agent[i].radius
            if agent_radius == 0:
                agent_radius = agents.agent[i].safety_dis / 2
            agent_front_x = agent_radius * np.cos(traj[k, n1 + 2])
            agent_front_y = agent_radius * np.sin(traj[k, n1 + 2])
            ax.plot([x, x + agent_front_x], [y, y + agent_front_y], '-k', label='direction')  # label 5

            # plot past trajectory
            ax.plot(traj[:k, n1], traj[:k, n1 + 1], pasttraj_type, color=self_color[i], linewidth=1,
                    label='trajectory')  # label 6
            past_traj = np.asarray(agents.agent[i].trajectory)
            ax.plot(past_traj[:, 0], past_traj[:, 1], pasttraj_type, color=self_color[i], linewidth=1)

            # plot the prediction of self-trajectory
            ax.plot(traj[k:, n1], traj[k:, n1 + 1], predtraj_type,
                    color=self_color[i], linewidth=1, zorder=10, label='prediction')  # label 7

            # plot prediction trajectory of other
            if predict_other is True:
                for j in range(agents.agent_num):
                    if j != i and agents.agent_num < 3:
                        nj = j * agents.state_dim
                        ax.plot(traj[k:, nj], traj[k:, nj + 1], predtraj_type,
                                color=other_color[i], linewidth=1, zorder=0,
                                label='cooperated prediction')  # label 8, 9

                        nl += 1

        # draw circle obstacle
        if circle_obs is not None:
            ax.add_patch(
                Circle(xy=(circle_obs.pos[0][0], circle_obs.pos[0][1]), radius=circle_obs.radius[0], alpha=0.5,
                       color='k', label='wall'))  # label 10
            plot_boundary = self.update_boundary_with_obstacle(plot_boundary, circle_obs)

            for i in range(1, circle_obs.obstacle_num):
                ax.add_patch(
                    Circle(xy=(circle_obs.pos[i][0], circle_obs.pos[i][1]), radius=circle_obs.radius[i], alpha=0.5,
                           color='k'))
                plot_boundary = self.update_boundary_with_obstacle(plot_boundary, circle_obs)

        if rect_obs is not None:
            if circle_obs is not None:
                xy = rect_obs.pos[0][0]
                w = rect_obs.pos[0][2][0] - rect_obs.pos[0][0][0]
                h = rect_obs.pos[0][2][1] - rect_obs.pos[0][0][1]
                ax.add_patch(Rectangle(xy=xy, width=w, height=h, alpha=0.5, color='k'))

                for i in range(1, rect_obs.obstacle_num):
                    xy = rect_obs.pos[i][0]
                    w = rect_obs.pos[i][2][0] - rect_obs.pos[i][0][0]
                    h = rect_obs.pos[i][2][1] - rect_obs.pos[i][0][1]
                    ax.add_patch(Rectangle(xy=xy, width=w, height=h, alpha=0.5, color='k'))

                plot_boundary = self.update_boundary_with_rect_obstacle(plot_boundary, rect_obs)
            else:
                for i in range(rect_obs.obstacle_num):
                    xy = rect_obs.pos[i][0]
                    w = rect_obs.pos[i][2][0] - rect_obs.pos[i][0][0]
                    h = rect_obs.pos[i][2][1] - rect_obs.pos[i][0][1]
                    ax.add_patch(Rectangle(xy=xy, width=w, height=h, alpha=0.5, color='k'))
                plot_boundary = self.update_boundary_with_rect_obstacle(plot_boundary, rect_obs)

        # draw real boundary
        if boundary is not None:
            ax.plot([boundary[0], boundary[2]], [boundary[1], boundary[1]], 'k')
            ax.plot([boundary[0], boundary[2]], [boundary[3], boundary[3]], 'k')
            ax.plot([boundary[0], boundary[0]], [boundary[1], boundary[3]], 'k')
            ax.plot([boundary[2], boundary[2]], [boundary[1], boundary[3]], 'k')

            if plot_boundary[0] - 0.2 < boundary[0]:
                plot_boundary[0] = boundary[0] + 0.2
            if plot_boundary[1] - 0.2 < boundary[1]:
                plot_boundary[1] = boundary[1] + 0.2
            if plot_boundary[2] + 0.2 > boundary[2]:
                plot_boundary[2] = boundary[2] - 0.2
            if plot_boundary[3] + 0.2 > boundary[3]:
                plot_boundary[3] = boundary[3] - 0.2


        # For T-intersection Case
        # ax.plot([-1.5, -1.5], [-2.0, 2.2], 'k')
        # ax.plot([8.5, 8.5], [-2.0, 2.2], 'k')
        # ax.add_patch(Rectangle((-1.5, -2.0), 10.0, 0.5, linewidth=1, alpha=0.5, edgecolor='k', facecolor='k'))


        # limit the plotting boundary & adjust the height-weight ratio of the plot
        ax.set_xlim([plot_boundary[0] - 0.2, plot_boundary[2] + 0.2])
        ax.set_ylim([plot_boundary[1] - 0.2, plot_boundary[3] + 0.2])
        # ax.set_xlim([-1.5, 8.5])
        # ax.set_ylim([-2.0, 2.2])
        # ax.set_xlim([0.5, 7.5])
        # ax.set_ylim([-2.0, 2.0])
        ax.set_aspect('equal', adjustable='box')

        # Get & plot the legend
        handles, labels = ax.get_legend_handles_labels()
        if nl >= 8:
            nl = 8

        handle_list = []
        label_list = []
        for nl_index in range(nl):

            if nl_index == 4:
                handle_list.append((handles[nl_index]))
                label_list.append(labels[nl_index])
                continue

            handle_row = tuple()
            for i in range(agents.agent_num):
                handle_row = handle_row + (handles[nl * i + nl_index],)
            handle_list.append(handle_row)
            label_list.append(labels[nl_index])

        if circle_obs is not None:
            handle_list.append((handles[-1]))
            label_list.append(labels[-1])

        return handle_list, label_list

    def draw_closed_loop_process(self, img_list, agents, plot_boundary, save_name, boundary=None, circle_obs=None, predict_other=True, time_step_and_safety_dis=None):
        """

        @param img_list : need to include five number
        """
        img_num = len(img_list)
        sim_num = np.size(agents.agent[0].trajectory, 0) - 1

        row = len(img_list)
        col = img_num // row
        if img_num % row != 0:
            col += 1

        # create figure
        fig_width = 6.00
        heigh_width_ratio = (plot_boundary[3] - plot_boundary[1]) / (plot_boundary[2] - plot_boundary[0])
        if heigh_width_ratio > 1.4:
            fig_width = 10.00
        fig, ax = plt.subplots(col, row, constrained_layout=True, figsize=(22.20, fig_width))  # T-intersection : 4.00

        count = 0
        for cl in range(col):
            for rl in range(row):

                if cl > 1:
                    axes = ax[cl, rl]
                else:
                    axes = ax[rl]

                img_id = img_list[count]


                if time_step_and_safety_dis is not None and len(time_step_and_safety_dis) > 1:
                    for reactive_time in time_step_and_safety_dis:
                        if reactive_time <= img_id:
                            for j in range(agents.agent_num):
                                agents.agent[j].safety_dis = time_step_and_safety_dis[reactive_time][j]
                        else:
                            break

                if img_id < sim_num:
                    handle_list, label_list = self.draw_timestep_trajectory2(
                        axes,
                        agents,
                        plot_boundary,
                        img_id,
                        boundary=boundary,
                        circle_obs=circle_obs,
                        predict_other=predict_other)
                else:
                    handle_list, label_list = self.draw_prediction_animation2(
                        axes,
                        agents,
                        plot_boundary,
                        img_id - sim_num,
                        sim_num,
                        boundary=boundary,
                        circle_obs=circle_obs,
                        predict_other=predict_other)

                axes.axis('off')
                axes.set_title('T = %.1f s' % (img_id * 0.1), fontsize=16, weight='bold') # fontname='monospace', 'cursive'
                count += 1

                if rl == 0:
                    fig.legend(handle_list,
                               label_list,
                               handler_map={tuple: HandlerTuple(ndivide=None)},
                               loc='lower center',
                               ncol=len(label_list),
                               prop={"size": 16})
                                # loc='lower center', bbox_to_anchor=(0.5, 0.28), handletextpad=0.3, handlelength=5,

                if rl > 0:
                    trans = blended_transform_factory(axes.transAxes, axes.transAxes)
                    line = Line2D([0, 0], [0, 1], color='k', transform=trans, linewidth=1)
                    fig.lines.append(line)

        fig_name = save_name + "closedLoop"
        # self.save_plot(fig, fig_name)
        fig.savefig(
            f"figs/{fig_name}.png",
            transparent=False,
            facecolor="white", dpi=300, bbox_inches='tight')
        plt.close(fig)
        # plt.show()

    def draw_velocity_plot(self, agents):

        # create color list
        color_list = sns.color_palette("Paired", agents.agent_num * 2)
        self_color = []
        other_color = []
        for i in range(agents.agent_num):
            other_color.append(color_list[i * 2])
            self_color.append(color_list[i * 2 + 1])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(agents.agent_num):
            traj = np.asarray(agents.agent[i].trajectory)
            ax.plot(range(np.size(traj, 0)), traj[:, 3], color=self_color[i])

        ax.spines[['right', 'top']].set_visible(False)

        plt.show()

    ###############################
    # Save plots
    ###############################

    def save_env_setting(self, agents, plot_boundary, save_name, boundary=None, circle_obs=None,
                         rect_obs=None, axis_off=False):
        """
        Draw the problem scene and save as image.

        PreConditions :
        PostConditions : Save the image of the problem scene at /figs/...

        @param agents: all agents in the problem scene
        @param plot_boundary: boundary used on plotting, [xmin, ymin, xmax, ymax]
        @param save_name: a string that will save as the image's name
        @param boundary: boundary of the problem scene, [xmin, ymin, xmax, ymax]
        @param circle_obs: circle obstacle
        @param polygon_obs: polygon obstacle
        """

        rol = 2
        col = 1
        fig_width = (plot_boundary[2] - plot_boundary[0]) * 1.1 * 2
        fig_height = plot_boundary[3] - plot_boundary[1]

        if boundary is not None:
            fig_width = (boundary[2] - boundary[0]) * 1.1 * 2
            fig_height = boundary[3] - boundary[1]

        if fig_width * 0.9 > 15.0:
            ratio = 15.0 / fig_width
            fig_width = 15.0
            fig_height = fig_height * ratio

        fig, ax = plt.subplots(col, rol, constrained_layout=True, figsize=(fig_width, fig_height))  # (8, 4)

        # create color list
        self_color, other_color = self.generate_color_list(agents.agent_num)

        pos_type = 'o'
        goal_type = 'x'
        traj_type = '.-'

        # find agent that have the smallest safety distance
        agent_id = 0
        for i in range(1, agents.agent_num):
            if agents.agent[i].safety_dis < agents.agent[agent_id].safety_dis:
                agent_id = i

        # plot the initial state
        axes = ax[0]
        axes.set_title('Initial')
        for i in range(agents.agent_num):
            x = agents.agent[i].initial_state[0]
            y = agents.agent[i].initial_state[1]
            xf = agents.agent[i].target_state[0]
            yf = agents.agent[i].target_state[1]

            # plot the initial state & target state
            axes.plot(x, y, pos_type, color=self_color[i], markerfacecolor='None', label='start')  # legend 1
            axes.plot(xf, yf, goal_type, color=self_color[i], label='goal')  # legend 2

            # plot the agent size & safety distance at the initial state
            theta = agents.agent[i].initial_state[2]
            axes.plot(x, y, traj_type, color=self_color[i], markersize='0.5')
            axes.add_patch(Circle(xy=(x, y), radius=agents.agent[i].safety_dis / 2, alpha=0.1, color=self_color[i], label='safety circle'))  # legend 3
            axes.add_patch(Circle(xy=(x, y), radius=agents.agent[i].radius, alpha=0.3, color=self_color[i], label='current position with agent size'))  # legend 4

            # plot the car direction at the initial state
            agent_radius = agents.agent[i].radius
            if agent_radius == 0:
                agent_radius = agents.agent[agent_id].safety_dis / 2
            agent_front_x = agent_radius * np.cos(theta)
            agent_front_y = agent_radius * np.sin(theta)
            axes.plot([x, x + agent_front_x], [y, y + agent_front_y], '-k', label='direction')  # legend 5

        # Get & plot the legend
        handles, labels = axes.get_legend_handles_labels()
        nl = 5
        handle_list = []
        for nl_index in range(nl - 1):
            handle_row = tuple()
            for i in range(agents.agent_num):
                handle_row = handle_row + (handles[nl * i + nl_index], )
            handle_list.append(handle_row)
        handle_list.append((handles[4]))

        fig.legend(handle_list,
                   [labels[0], labels[1], labels[2], labels[3], labels[4]],
                   handler_map={tuple: HandlerTuple(ndivide=None)}, loc='lower center', ncol=3)

        if axis_off:
            axes.xaxis.set_visible(False)
            axes.yaxis.set_visible(False)

        # For T-intersection Case
        # axes.add_patch(Rectangle((-2.2, -2.0), 12.7, 0.5, linewidth=1, alpha=0.5, edgecolor='k', facecolor='k'))

        # plot the target state
        axes = ax[1]
        axes.set_title('Target')
        for i in range(agents.agent_num):
            x = agents.agent[i].initial_state[0]
            y = agents.agent[i].initial_state[1]
            xf = agents.agent[i].target_state[0]
            yf = agents.agent[i].target_state[1]

            # plot the initial state & target state
            axes.plot(x, y, pos_type, color=self_color[i], markerfacecolor='None')
            axes.plot(xf, yf, goal_type, color=self_color[i])

            # plot the agent size & safety distance at the target state
            axes.plot(xf, yf, traj_type, color=self_color[i], markersize='0.5')
            theta_f = agents.agent[i].target_state[2]
            axes.add_patch(Circle(xy=(xf, yf), radius=agents.agent[i].safety_dis / 2, alpha=0.1, color=self_color[i]))
            axes.add_patch(Circle(xy=(xf, yf), radius=agents.agent[i].radius, alpha=0.3, color=self_color[i]))

            # plot the car direction at the target state
            agent_radius = agents.agent[i].radius
            if agent_radius == 0:
                agent_radius = agents.agent[agent_id].safety_dis / 2

            agent_front_x = agent_radius * np.cos(theta_f)
            agent_front_y = agent_radius * np.sin(theta_f)
            axes.plot([xf, xf + agent_front_x], [yf, yf + agent_front_y], '-k')

        # plot obstacles & boundary
        for ax_index in range(2):
            axes = ax[ax_index]
            # draw circle obstacle
            if circle_obs is not None:
                for i in range(circle_obs.obstacle_num):
                    axes.add_patch(
                        Circle(xy=(circle_obs.pos[i][0], circle_obs.pos[i][1]), radius=circle_obs.radius[i], alpha=0.5,
                               color='k'))
                plot_boundary = self.update_boundary_with_obstacle(plot_boundary, circle_obs)

            plot_boundary = self.update_boundary_with_agents(plot_boundary, agents)

            if rect_obs is not None:
                for i in range(rect_obs.obstacle_num):
                    xy = rect_obs.pos[i][0]
                    w = rect_obs.pos[i][2][0] - rect_obs.pos[i][0][0]
                    h = rect_obs.pos[i][2][1] - rect_obs.pos[i][0][1]
                    axes.add_patch(Rectangle(xy=xy, width=w, height=h, alpha=0.5, color='k'))
                    plot_boundary = self.update_boundary_with_rect_obstacle(plot_boundary, rect_obs)

        # draw boundary
        if boundary is not None:
            plot_boundary = boundary
        for ax_index in range(2):
            axes = ax[ax_index]
            if boundary is not None:
                axes.plot([boundary[0], boundary[2]], [boundary[1], boundary[1]], 'k')
                axes.plot([boundary[0], boundary[2]], [boundary[3], boundary[3]], 'k')
                axes.plot([boundary[0], boundary[0]], [boundary[1], boundary[3]], 'k')
                axes.plot([boundary[2], boundary[2]], [boundary[1], boundary[3]], 'k')

            axes.set_xlim([plot_boundary[0] - 0.2, plot_boundary[2] + 0.2])
            axes.set_ylim([plot_boundary[1] - 0.2, plot_boundary[3] + 0.2])
            axes.set_aspect('equal', adjustable='box')

        if axis_off:
            axes.xaxis.set_visible(False)
            axes.yaxis.set_visible(False)

        # For T-intersection Case
        # axes.add_patch(Rectangle((-2.2, -2.0), 12.7, 0.5, linewidth=1, alpha=0.5, edgecolor='k', facecolor='k'))

        # plt.show()

        fig_name = save_name
        self.save_plot(fig, fig_name)

    def save_timestep_plot(self, agents, plot_boundary, save_name, sim_num, boundary=None, circle_obs=None,
                           rect_obs=None,
                           predict_other=True, draw_animation=False):
        """
        Draw and save the trajectory plot at the given time-step

        -------------------------------------------------------------------
        PreConditions :
        PostConditions : draw trajectory plot and save as an image

        @param        agents: all agents in the scene
        @param plot_boundary: the boundary of the plot
        @param     save_name: string, name of the image
        @param       sim_num: the current time step of the trajectory
        @param      boundary: the real 2D x-y boundary of the scene
        @param    circle_obs: circle obstacle
        @param predict_other: bool, if true, then the prediction of other-trajectory
                              will be drawn if agent number is less than three
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.draw_timestep_trajectory(
            ax,
            agents,
            plot_boundary,
            sim_num,
            boundary=boundary,
            circle_obs=circle_obs,
            rect_obs=rect_obs,
            predict_other=predict_other)

        fig_name = save_name + "img_" + str(sim_num)
        self.save_plot(fig, fig_name)

        # save animation
        if draw_animation:
            animate_len = np.shape(agents.agent[0].sol_series[-1][0])[0]

            fig = plt.figure()
            ax = fig.add_subplot(111)

            def animate_traj(k):
                ax.clear()

                self.draw_prediction_animation(
                    ax,
                    agents,
                    plot_boundary,
                    k,
                    sim_num,
                    boundary=boundary,
                    circle_obs=circle_obs,
                    rect_obs=rect_obs,
                    predict_other=True)

            # save animate
            save_name = f"figs/{fig_name}.mp4"

            ani = FuncAnimation(fig, animate_traj, frames=animate_len, interval=200, repeat=False)
            ani.save(save_name, writer='ffmpeg', fps=10)
            # plt.show()

    def save_part_trajectory_plot(self, agents, plot_boundary, save_name, sim_time_range, boundary=None, circle_obs=None,
                        predict_other=True, axis_off=False):

        for time_range in sim_time_range:

            color_num = 6
            if len(time_range) > color_num:
                color_num = len(time_range)
            color_list_blue = sns.color_palette("Blues", color_num)
            color_list_green = sns.color_palette("Greens", color_num)

            fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(5.00, 4.00))

            self.draw_part_trajectory(ax, agents, plot_boundary, time_range, color_list_blue, color_list_green, boundary=boundary, circle_obs=circle_obs, predict_other=predict_other)

            if axis_off:
                ax.axis('off')

            time_range_string = str(time_range[0])
            for num in time_range:
                time_range_string = time_range_string + "_" + str(num)

            fig_name = save_name + "_trajectory_" + time_range_string
            self.save_plot(fig, fig_name)

    def save_multi_plot(self, agents, plot_boundary, save_name, sim_num, boundary=None, circle_obs=None,
                        rect_obs=None, predict_other=True, axis_off=False, time_step_and_safety_dis=None):
        """
        Draw and save the full trajectory plot and the trajectory plot at each time-step

        -------------------------------------------------------------------
        PreConditions :
        PostConditions : save the full path image and the trajectory image at each time step

        @param        agents: all agents in the scene
        @param plot_boundary: the boundary of the plot
        @param     save_name: string, name of the image
        @param       sim_num: the current time step of the trajectory
        @param      boundary: the real 2D x-y boundary of the scene
        @param    circle_obs: circle obstacle
        @param predict_other: bool, if true, then the prediction of other-trajectory
                              will be drawn if agent number is less than three
        """

        # save full path
        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.draw_full_trajectory(ax, agents, plot_boundary, boundary=boundary, circle_obs=circle_obs)

        fig_name = save_name + "full_path"
        self.save_plot(fig, fig_name)

        # save trajectory at each time step
        for k in range(sim_num):

            if time_step_and_safety_dis is not None and k in time_step_and_safety_dis:
                for i in range(agents.agent_num):
                    agents.agent[i].safety_dis = time_step_and_safety_dis[k][i]

            # fig = plt.figure()
            # ax = fig.add_subplot(111)

            fig_width = plot_boundary[2] - plot_boundary[0] # 8
            fig_height = plot_boundary[3] - plot_boundary[1] # 5
            if fig_width * 0.9 > 15.0:
                ratio = 15.0 / fig_width
                fig_width = 15.0
                fig_height = fig_height * ratio

            fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(fig_width * 0.9, fig_height))  # T-intersection : 4.00

            self.draw_timestep_trajectory(
                ax,
                agents,
                plot_boundary,
                k,
                boundary=boundary,
                circle_obs=circle_obs,
                rect_obs=rect_obs,
                predict_other=predict_other)

            if axis_off:
                ax.axis('off')

            # ax.set_title('T = %.1f s' % (k * 0.1), fontsize=16, weight='bold')

            fig_name = save_name + "img_" + str(k)
            self.save_plot(fig, fig_name)
            print("save img...", k + 1, "/", sim_num)

    def save_animate(self,
                     agents, plot_boundary, save_name, sim_num,
                     boundary=None,
                     circle_obs=None,
                     rect_obs=None,
                     predict_other=True,
                     costum_figsize=None,
                     running_time_visible=False,
                     labels_visible=False,
                     complete_trajectory_with_prediction=False,
                     axis_off=False,
                     time_step_and_safety_dis=None):
        """
        Draw and save the animation of the closed-loop trajectory planning

        -------------------------------------------------------------------
        PreConditions :
        PostConditions : save the animation of the closed-loop trajectory planning, the video type is "mp4".

        @param        agents: all agents in the scene
        @param plot_boundary: the boundary of the plot
        @param     save_name: string, name of the image
        @param       sim_num: the current time step of the trajectory
        @param      boundary: the real 2D x-y boundary of the scene
        @param    circle_obs: circle obstacle
        @param predict_other: bool, if true, then the prediction of other-trajectory
                              will be drawn if agent number is less than three
        """

        animate_frame = sim_num
        if complete_trajectory_with_prediction:
            animate_frame = sim_num + np.shape(agents.agent[0].sol_series[0][0])[0] - 1

        fig_width = plot_boundary[2] - plot_boundary[0]
        fig_height = plot_boundary[3] - plot_boundary[1]

        if fig_width * 0.9 > 15.0:
            ratio = 15.0 / fig_width
            fig_width = 15.0
            fig_height = fig_height * ratio

        if costum_figsize is not None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=costum_figsize)
        else:
            fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(fig_width * 0.9, fig_height))

        def animate_traj(k):
            ax.clear()
            handle_list = None

            if time_step_and_safety_dis is not None and k in time_step_and_safety_dis:
                for i in range(agents.agent_num):
                    agents.agent[i].safety_dis = time_step_and_safety_dis[k][i]

            if k < sim_num:

                if labels_visible and k == 0:
                    handle_list, label_list = self.draw_timestep_trajectory2(
                        ax,
                        agents,
                        plot_boundary,
                        k,
                        boundary=boundary,
                        circle_obs=circle_obs,
                        rect_obs=rect_obs,
                        predict_other=predict_other)
                else:
                    self.draw_timestep_trajectory(
                        ax,
                        agents,
                        plot_boundary,
                        k,
                        boundary=boundary,
                        circle_obs=circle_obs,
                        rect_obs=rect_obs,
                        predict_other=predict_other)

            else:

                if labels_visible and k == 0:
                    handle_list, label_list = self.draw_prediction_animation2(
                        ax,
                        agents,
                        plot_boundary,
                        k - sim_num + 1,
                        sim_num - 1,
                        boundary=boundary,
                        circle_obs=circle_obs,
                        predict_other=predict_other)
                else:
                    self.draw_prediction_animation(
                        ax,
                        agents,
                        plot_boundary,
                        k - sim_num + 1,
                        sim_num - 1,
                        boundary=boundary,
                        circle_obs=circle_obs,
                        predict_other=predict_other,
                        show_full_predict=False)

            if axis_off:
                ax.axis('off')

            if running_time_visible:
                fig.suptitle('T = %.1f s' % (k * 0.1), fontsize=14, weight='bold', y=0.98)

            if handle_list is not None:
                col_num = len(label_list) // 2
                if len(label_list) % 2 != 0:
                    col_num += 1

                fig.legend(handle_list,
                           label_list,
                           handler_map={tuple: HandlerTuple(ndivide=None)},
                           loc='lower center',
                           ncol=col_num,
                           prop={"size": 8})

        # save animate
        save_video_name = f"figs/{save_name}demo.mp4"
        print("save animation...")

        ani = FuncAnimation(fig, animate_traj, frames=animate_frame, interval=100, repeat=False)
        ani.save(save_video_name, writer='ffmpeg', fps=10)

        # convert mp4 to gif
        self.convert_mp4_to_gif(save_video_name, save_name)

        # plt.show()

    def save_prediction_animation(self, agents, plot_boundary, save_name, time_step, boundary, circle_obs, rect_obs):
        self.save_timestep_plot(agents, plot_boundary, save_name, time_step,
                                    boundary=boundary, circle_obs=circle_obs, rect_obs=rect_obs, draw_animation=True)
        save_video_name = f"figs/{save_name}img_{time_step}.mp4"
        save_predict_name = f"{save_name}img_{time_step}_"
        self.convert_mp4_to_gif(save_video_name, save_predict_name)

    def save_closed_loop_process_plot(self, img_list, agents, plot_boundary, save_name, boundary, circle_obs,
                                      predict_other=True, time_step_and_safety_dis=None):

        # open case
        if circle_obs is None or circle_obs.obstacle_num == 0:
            self.draw_closed_loop_process(img_list, agents, plot_boundary, save_name,
                                              boundary=boundary, predict_other=predict_other, time_step_and_safety_dis=time_step_and_safety_dis)

        else:  # obstacle case
            self.draw_closed_loop_process(img_list, agents, plot_boundary, save_name,
                                              boundary=boundary, circle_obs=circle_obs, predict_other=predict_other, time_step_and_safety_dis=time_step_and_safety_dis)

    def save_solver_cost_time_plot(self, solver, solved_time_series, save_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        N = np.size(solved_time_series, 0) - 1
        agent_num = np.size(solved_time_series, 1)
        x = np.arange(1, N)

        if solver == "Centralized":
            ax.plot(x, solved_time_series[1:N, 0])
            mean_time = np.mean(solved_time_series[1:N, 0])

            subplot_title = "First Compiled + Executed Time = " + str(round(solved_time_series[0, 0], 3)) + "  (sec)" \
                            + "   Average Executed Time = " + str(round(mean_time, 3)) + " (sec)"
            ax.set_title(subplot_title, fontsize=8)
        else:
            for i in range(agent_num):
                label = 'agent ' + str(i)
                ax.plot(x, solved_time_series[1:N, i], label=label)

            ax.legend()

        # plt.yscale('log')

        fig.supxlabel('time step', fontsize=10)
        fig.supylabel('0 : compile + execute time, 1 ~ : execute time (sec)', fontsize=10)

        title = "Solver Compile & Execute Time  (" + solver + ")"
        fig.suptitle(title, fontsize=14)

        fig_name = save_name + "solved_time"

        self.save_plot(fig, fig_name)

    def save_plot(self, fig, name):
        fig.savefig(
            f"figs/{name}.png",
            transparent=False,
            facecolor="white", dpi=300)
        plt.close(fig)

    def convert_mp4_to_gif(self, save_video_name, save_name):
        videoClip = VideoFileClip(save_video_name)
        save_video_name = f"figs/{save_name}demo.gif"
        videoClip.write_gif(save_video_name)  # To slow down gif : fps=100

    ###############################
    # Helping Functions
    ###############################

    def generate_color_list(self, agent_num):

        color_list = None
        color_list_1 = None
        color_list_2 = None
        if agent_num <= 6:
            color_list = sns.color_palette("Paired", agent_num * 2)
        else:
            color_list_1 = sns.husl_palette(agent_num)
            color_list_2 = sns.husl_palette(agent_num, s=.6)

        self_color = []
        other_color = []
        for i in range(agent_num):
            if agent_num <= 6:
                other_color.append(color_list[i * 2])
                self_color.append(color_list[i * 2 + 1])
            else:
                other_color.append(color_list_2[i])
                self_color.append(color_list_1[i])

        return self_color, other_color

    def update_boundary_with_obstacle(self, plot_boundary, circle_obs):
        """
        Ensure that the plot includes at least half of each circle obstacle


        PreConditions :
        PostConditions : return new boundary for plotting

        @param plot_boundary: boundary that will use in plot
        @param circle_obs: circle obstacle

        @return (np.ndarray) plot_boundary: [xmin, ymin, xmax, ymax]
        """
        if circle_obs is not None:
            for i in range(circle_obs.obstacle_num):
                if circle_obs.pos[i][0] < plot_boundary[0]:
                    plot_boundary[0] = circle_obs.pos[i][0]

                if circle_obs.pos[i][0] > plot_boundary[2]:
                    plot_boundary[2] = circle_obs.pos[i][0]

                if circle_obs.pos[i][1] < plot_boundary[1]:
                    plot_boundary[1] = circle_obs.pos[i][1]

                if circle_obs.pos[i][1] > plot_boundary[3]:
                    plot_boundary[3] = circle_obs.pos[i][1]

        return plot_boundary

    def update_boundary_with_rect_obstacle(self, plot_boundary, rect_obs):

        if rect_obs is not None:
            for i in range(rect_obs.obstacle_num):
                x_min = rect_obs.pos[i][0][0]
                x_max = rect_obs.pos[i][2][0]
                y_min = rect_obs.pos[i][0][1]
                y_max = rect_obs.pos[i][2][1]

                if x_min < plot_boundary[0]:
                    plot_boundary[0] = x_min

                if x_max > plot_boundary[2]:
                    plot_boundary[2] = x_max

                if y_min < plot_boundary[1]:
                    plot_boundary[1] = y_min

                if y_max > plot_boundary[3]:
                    plot_boundary[3] = y_max

        return plot_boundary

    def update_boundary_with_agents(self, plot_boundary, agents):
        """
        Update the plot boundary with agents initial position, goal position and safety distance.


        PreConditions :
        PostConditions :
            return new plot boundary which includes the initial
            position and goal position with the safety circle
            of all the agents.

        @param plot_boundary
        @param agents

        @return plot_boundary: [xmin, ymin, xmax, ymax]
        """
        agent = agents.agent[0]
        radius = agents.agent[0].safety_dis / 2
        xmin = min(agent.initial_state[0] - radius, agent.target_state[0] - radius)
        xmax = max(agent.initial_state[0] + radius, agent.target_state[0] + radius)
        ymin = min(agent.initial_state[1] - radius, agent.target_state[1] - radius)
        ymax = max(agent.initial_state[1] + radius, agent.target_state[1] + radius)

        for i in range(1, agents.agent_num):
            agent = agents.agent[i]
            radius = agents.agent[i].safety_dis / 2
            xmin_tmp = min(agent.initial_state[0] - radius, agent.target_state[0] - radius)
            xmax_tmp = max(agent.initial_state[0] + radius, agent.target_state[0] + radius)
            ymin_tmp = min(agent.initial_state[1] - radius, agent.target_state[1] - radius)
            ymax_tmp = max(agent.initial_state[1] + radius, agent.target_state[1] + radius)

            if xmin_tmp < xmin:
                xmin = xmin_tmp

            if ymin_tmp < ymin:
                ymin = ymin_tmp

            if xmax_tmp > xmax:
                xmax = xmax_tmp

            if ymax_tmp > ymax:
                ymax = ymax_tmp

        if xmin < plot_boundary[0]:
            plot_boundary[0] = xmin

        if ymin < plot_boundary[1]:
            plot_boundary[1] = ymin

        if xmax > plot_boundary[2]:
            plot_boundary[2] = xmax

        if ymax > plot_boundary[3]:
            plot_boundary[3] = ymax

        return plot_boundary

    def update_plot_boundary(self, plot_boundary, agents, traj):
        """
        Update the plotting boundary to include all the trajectories

        -----------------------------------------------------------------
        PreConditions : agents.agent_num * agents.state_dim == np.size(traj, 1)
        PostConditions :
            Return new boundary that include all the agent's trajectory,
            this boundary will be use on the plot.

        @param plot_boundary: the plotting boundary
        @param agents: all agents
        @param traj: agents trajectories

        @return (np.ndarray) plot_boundary: [xmin, ymin, xmax, ymax]
        """
        agent_num = agents.agent_num
        state_dim = agents.state_dim

        xmin_list = np.zeros(agent_num)
        ymin_list = np.zeros(agent_num)
        xmax_list = np.zeros(agent_num)
        ymax_list = np.zeros(agent_num)

        for i in range(agent_num):
            n1 = i * state_dim
            x = agents.agent[i].position[0]
            y = agents.agent[i].position[1]
            radius = agents.agent[i].safety_dis / 2

            # find the min & max x-y in the trajectory, also consider the current position includes the safety distance)
            xmin_list[i] = min(min(traj[:, n1]), x - radius)
            xmax_list[i] = max(max(traj[:, n1]), x + radius)
            ymin_list[i] = min(min(traj[:, n1 + 1]), y - radius)
            ymax_list[i] = max(max(traj[:, n1 + 1]), y + radius)

        if (min(xmin_list) < plot_boundary[0]):
            plot_boundary[0] = min(xmin_list)

        if (min(ymin_list) < plot_boundary[1]):
            plot_boundary[1] = min(ymin_list)

        if (max(xmax_list) > plot_boundary[2]):
            plot_boundary[2] = max(xmax_list)

        if (max(ymax_list) > plot_boundary[3]):
            plot_boundary[3] = max(ymax_list)

        return plot_boundary

    def generate_constant_interval_img_list(self, img_total_num, list_length, horizon, complete_trajectory_with_prediction=False):

        if complete_trajectory_with_prediction:
            last_num = img_total_num + horizon
            # img_total_num = img_total_num + horizon
        else:
            last_num = img_total_num

        interval = img_total_num // (list_length - 1)
        img_list = [0]
        for i in range(1, (list_length - 1)):
            img_list.append(i * interval)
        img_list.append(last_num - 1)

        return img_list

    def check_collision(self, agents, constraints_threshold, obs=None):
        """
        if collision happen, return False, else return Ture
        """

        horizon = np.size(agents.agent[0].trajectory, 0)

        for k in range(horizon):
            for i in range(agents.agent_num):
                x_i = agents.agent[i].trajectory[k]

                # check agent collision
                for j in range(i + 1, agents.agent_num):
                    x_j = agents.agent[j].trajectory[k]

                    dis = (agents.agent[i].radius + agents.agent[j].radius) - \
                          np.sqrt((x_i[0] - x_j[0]) ** 2.0 + (x_i[1] - x_j[1]) ** 2.0)

                    if dis > constraints_threshold:
                        print("time step = ", k, "  agent collision : dis = ", dis)
                        if k < (horizon - 1):
                            return False, k + 1
                        else:
                            return False, k

                # check obstacle collision
                if obs is not None:
                    for j in range(obs.obstacle_num):
                        dis = (obs.radius[j] + agents.agent[i].radius) - np.sqrt(
                            (x_i[0] - obs.pos[j][0]) ** 2.0 + (x_i[1] - obs.pos[j][1]) ** 2.0)

                        if dis > constraints_threshold:
                            print("time step = ", k, "obstacle collision : dis = ", dis)
                            if k < (horizon - 1):
                                return False, k + 1
                            else:
                                return False, k

        return True, horizon

    def compute_trajectory_length(self, agents):
        trajectory_length = np.zeros(agents.agent_num)

        for i in range(agents.agent_num):
            traj = agents.agent[i].trajectory
            traj = np.asarray(traj)
            dis = 0
            for k in range(1, np.size(traj, 0)):
                dis = dis + ((traj[k, 0] - traj[k - 1, 0]) ** 2 + (traj[k, 1] - traj[k - 1, 1]) ** 2) ** 0.5

            dis = dis + ((agents.agent[i].target_state[0] - traj[-1, 0]) ** 2 + (agents.agent[i].target_state[1] - traj[-1, 1]) ** 2) ** 0.5

            trajectory_length[i] = dis

        print("trajectory_length : ", trajectory_length)
        return trajectory_length


    def count_total_simulation_time_and_check_reactive_safety_distance(self, folder, agent_num):
        file_count = 0
        time_step_and_safety_dis = None

        for fold_path in os.listdir(folder):
            # check if current path is a file
            if os.path.isfile(os.path.join(folder, fold_path)):
                file_string = fold_path.split('.')
                file_name = file_string[0]

                if file_name.split('_')[0][0:5] == "agent":
                    file_count += 1
                if file_name == "safety_dis":
                    time_step_and_safety_dis = self.load_reactive_safety_distance_info(folder)

        file_count = math.floor(file_count / agent_num)

        return file_count, time_step_and_safety_dis


    def draw_values_grad_norm(self, values, grad_norms):
        plt.figure(figsize=(6, 4))

        plt.subplot(1, 2, 1)
        plt.semilogy(values)
        plt.xlabel('iteration')
        plt.ylabel('obj')
        plt.title(f'final obj={values[-1]:.3f}')

        plt.subplot(1, 2, 2)
        plt.semilogy(grad_norms)
        plt.xlabel('iteration')
        plt.ylabel('norm(grad)')
        plt.title(f'final norm(grad)={grad_norms[-1]:.3f}')

        plt.suptitle('reacher traj opt (Adam)')

        plt.tight_layout()
        plt.show()


##############################

draw_fxn = DrawClass()

def redraw_simulation(folder_name, prediction_time_step=None):
    folder = "figs/" + folder_name

    # load npz file & rebuild agents, obstacles
    agents, circle_obs, boundary, horizon, boundary_range, solver_type, adaptive_backup_weight, hetero_solvers = draw_fxn.load_info(
        folder)

    rect_obs = None
    rect_obs = RectObstacles(obstacle_num=6,
                             pos=[np.array([[-3.5, -6.0], [-3.5, -4.0], [-1.5, -4.0], [-1.5, -6.0], [-3.5, -6.0]]),
                                  np.array([[-3.5, -3.0], [-3.5, -1.0], [-1.5, -1.0], [-1.5, -3.0], [-3.5, -3.0]]),
                                  np.array([[-3.5, 0.0], [-3.5, 2.0], [-1.5, 2.0], [-1.5, 0.0], [-3.5, 0.0]]),
                                  np.array([[1.5, -6.0], [1.5, -4.0], [3.5, -4.0], [3.5, -6.0], [1.5, -6.0]]),
                                  np.array([[1.5, -3.0], [1.5, -1.0], [3.5, -1.0], [3.5, -3.0], [1.5, -3.0]]),
                                  np.array([[1.5, 0.0], [1.5, 2.0], [3.5, 2.0], [3.5, 0.0], [1.5, 0.0]])
                                  ])

    total_num, time_step_and_safety_dis = draw_fxn.count_total_simulation_time_and_check_reactive_safety_distance(
        folder, agents.agent_num)
    solved_time_series = draw_fxn.load_solver_cost_time_info(folder)
    print('Total Num:', total_num)

    #############################
    # rebuild agents
    #############################
    print("rebuilding agents...")
    agents, plot_boundary = draw_fxn.rebuild_agents(folder_name, total_num, agents, boundary, circle_obs)

    # build folder
    save_name = folder + "rebuild_img/"
    if not os.path.exists(save_name):
        os.mkdir(save_name)
        print("Folder %s created!" % save_name)
    else:
        print("Folder %s already exists" % save_name)

    #############################
    # all single imgs & animation
    #############################
    save_name = folder_name + "rebuild_img/"
    save_env_img = save_name + "/environment"
    save_solved_time_img = save_name + "/solved_time"

    # open case need to reset boundary
    if np.shape(boundary) == ():
        draw_fxn.save_env_setting(agents, plot_boundary, save_env_img, circle_obs=circle_obs, rect_obs=rect_obs, axis_off=True)
        boundary = np.array([-1000000, -1000000, 1000000, 1000000])  # xmin, ymin, xmax, ymax
    else:
        draw_fxn.save_env_setting(agents, plot_boundary, save_env_img, boundary=boundary, circle_obs=circle_obs, rect_obs=rect_obs,
                                  axis_off=True)

    # save solver cost time
    if solved_time_series is not None:
        draw_fxn.save_solver_cost_time_plot(str(solver_type), solved_time_series, save_solved_time_img)

    #############################
    # Draw prediction video at certain time step
    #############################
    if prediction_time_step is not None:
        draw_fxn.save_prediction_animation(agents, plot_boundary, save_name, prediction_time_step, boundary, circle_obs, rect_obs)

    #############################
    # Redraw and save img & video
    #############################
    predict_other = True
    if solver_type == "Centralized" or agents.agent_num >= max_predict_agent_num:
        predict_other = False

    # use the distance error to determine whether to complete the trajectory with prediction or not
    total_error = 0
    use_prediction = False
    for i in range(agents.agent_num):
        traj = np.asarray(agents.agent[i].trajectory)
        dis = math.dist([agents.agent[i].target_state[0], agents.agent[i].target_state[1]], [traj[-1, 0], traj[-1, 1]])
        total_error += dis

    no_collision, last_time_step = draw_fxn.check_collision(agents, constraints_threshold=0.15, obs=circle_obs)  # 0.02

    if total_error >= 1.2 and no_collision:
        use_prediction = True

    # save img
    draw_fxn.save_multi_plot(agents, plot_boundary, save_name, total_num,
                             boundary=boundary, circle_obs=circle_obs, rect_obs=rect_obs,
                             axis_off=True, time_step_and_safety_dis=time_step_and_safety_dis)
    # save animate
    draw_fxn.save_animate(agents, plot_boundary, save_name, last_time_step,
                          boundary=boundary, circle_obs=circle_obs, rect_obs=rect_obs, # , costum_figsize=(8, 5.5)
                          predict_other=predict_other,
                          running_time_visible=True, labels_visible=False,
                          complete_trajectory_with_prediction=use_prediction,
                          axis_off=True, time_step_and_safety_dis=time_step_and_safety_dis)

    #############################
    # Draw closed-loop process
    #############################
    # choose plot img
    # img_list = [1, 32, 47, 106] # 1, 32, 47, 106
    img_list_len = 4
    img_list = draw_fxn.generate_constant_interval_img_list(last_time_step, img_list_len, horizon,
                                                            complete_trajectory_with_prediction=use_prediction)

    draw_fxn.save_closed_loop_process_plot(img_list, agents, plot_boundary, save_name, boundary, circle_obs,
                                           predict_other=predict_other,
                                           time_step_and_safety_dis=time_step_and_safety_dis)

    print("... Done !")

