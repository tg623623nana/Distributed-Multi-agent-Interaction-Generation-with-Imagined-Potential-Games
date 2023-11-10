# simulation.py
"""
Simulation Setting

Created on 2023/3/14
@author: Pin-Yun Hung
"""

from cloosed_loop_control import *
import numpy as onp

SIMULATION_NUMBER = 200


class Simulation:

    def run_simulaiton(self, agents, boundary, solver, sim_time, use_multi_thread,
                       circle_obs=None, adaptive_backup_weight=False, hetero_solver=None, random_boundary=None,
                       start_sim_num=0, time_step_and_safety_dis=None):
        print("Start simulation ...")

        if solver == "Centralized":
            sim_centralized(agents, sim_time=sim_time, sim_num=SIMULATION_NUMBER, boundary=boundary,
                            random_boundary=random_boundary, circle_obstacle=circle_obs,
                            adaptive_backup_weight=adaptive_backup_weight, start_sim_num=start_sim_num)

        elif solver == "IPG":
            sim_IPG(agents, sim_time=sim_time, sim_num=SIMULATION_NUMBER, boundary=boundary,
                    random_boundary=random_boundary, circle_obstacle=circle_obs,
                    use_multi_thread=use_multi_thread, adaptive_backup_weight=adaptive_backup_weight,
                    start_sim_num=start_sim_num)

        elif solver == "Vanilla":
            sim_Vanilla(agents, sim_time=sim_time, sim_num=SIMULATION_NUMBER, boundary=boundary,
                        random_boundary=random_boundary, circle_obstacle=circle_obs, use_multi_thread=use_multi_thread,
                        start_sim_num=start_sim_num)

        elif solver == "Brake":
            sim_Brake(agents, sim_time=sim_time, sim_num=SIMULATION_NUMBER, boundary=boundary,
                      random_boundary=random_boundary, circle_obstacle=circle_obs, use_multi_thread=use_multi_thread,
                      start_sim_num=start_sim_num)

        elif solver == "IPG_AdaptU":
            sim_IPG_AdaptU(agents, sim_time=sim_time, sim_num=SIMULATION_NUMBER, boundary=boundary,
                           random_boundary=random_boundary, circle_obstacle=circle_obs,
                           use_multi_thread=use_multi_thread, adaptive_backup_weight=adaptive_backup_weight,
                           start_sim_num=start_sim_num)

        elif solver == "Hetero":
            sim_Heterogeneous(agents, hetero_solver, sim_time=sim_time, sim_num=SIMULATION_NUMBER, boundary=boundary,
                              random_boundary=random_boundary, circle_obstacle=circle_obs,
                              use_multi_thread=use_multi_thread, adaptive_backup_weight=adaptive_backup_weight,
                              start_sim_num=start_sim_num,
                              time_step_and_safety_dis=time_step_and_safety_dis)

    def demo_open_2_agents(self, solver, use_multi_thread=False, adaptive_backup_weight=False, hetero_solver=None):
        """
       Trajectory planning with games, Open case

            - 2 agents
            - known others' goal
            - without obstacles
        """
        # setting
        boundary, agent_list = self.open_case_2_agent_setting()
        agents = MultiAgents(agent_list, agent_num=2)
        agents.update_limit(v_limit=[-10, 10], a_limit=[-50, 50], w_limit=[-np.pi / 2, np.pi / 2])

        # simulation
        self.run_simulaiton(agents, boundary, solver, sim_time=4, use_multi_thread=use_multi_thread,
                            adaptive_backup_weight=adaptive_backup_weight, hetero_solver=hetero_solver)

    def demo_open_3_agents(self, solver, use_multi_thread=False, adaptive_backup_weight=False, hetero_solver=None):
        """
       Trajectory planning with games, Open case

            - 3 agents
            - known others' goal
            - without obstacles
        """
        # setting
        boundary, agent_list = self.open_case_3_agent_setting()
        agents = MultiAgents(agent_list, agent_num=3)
        agents.update_limit(v_limit=[-10, 10], a_limit=[-50, 50], w_limit=[-np.pi / 2, np.pi / 2])

        # simulation
        self.run_simulaiton(agents, boundary, solver, sim_time=4, use_multi_thread=use_multi_thread,
                            adaptive_backup_weight=adaptive_backup_weight, hetero_solver=hetero_solver)

    def demo_narrow_way_2_agents(self, solver, use_multi_thread=False, adaptive_backup_weight=False,
                                 hetero_solver=None):
        """
        Trajectory planning with games, with obstacles

            - 2 agents
            - known others' goal
            - with obstacles (narrow road)
        """

        # boundary setting
        boundary = onp.array([-4, -3, 12, 3])

        # agents setting : different start & goal
        # agent_list = self.narrow_way_2_agent_setting1()
        agent_list = self.narrow_way_2_agent_setting2()
        # agent_list = self.narrow_way_2_agent_setting3()
        # agent_list = self.narrow_way_2_agent_setting4()
        # agent_list = self.narrow_way_2_agent_setting5()
        # agent_list = self.narrow_way_2_agent_setting6()

        agents = MultiAgents(agent_list, agent_num=2)
        agents.update_limit(v_limit=[-10, 10], a_limit=[-50, 50], w_limit=[-np.pi / 2, np.pi / 2])

        # circle obstacles setting
        circle_obs = self.narrow_way_obstacle_setting(circle_obstacle_num=2)

        # simulation
        self.run_simulaiton(agents, boundary, solver, sim_time=4,
                            use_multi_thread=use_multi_thread, circle_obs=circle_obs,
                            adaptive_backup_weight=adaptive_backup_weight, hetero_solver=hetero_solver)

    def demo_narrow_way_2_agents_hetero_solver(self, solver, use_multi_thread=False, adaptive_backup_weight=False,
                                               hetero_solver=None):
        """
        Trajectory planning with games, with obstacles

            - 2 agents
            - known others' goal
            - with obstacles (narrow road)
        """

        # boundary setting
        boundary = onp.array([-4, -3, 12, 3])

        # agents setting
        agent_list = self.narrow_way_2_agent_setting7()
        agents = MultiAgents(agent_list, agent_num=2)
        agents.update_limit(v_limit=[-10, 10], a_limit=[-50, 50], w_limit=[-np.pi / 2, np.pi / 2])

        # circle obstacles setting
        circle_obs = self.narrow_way_obstacle_setting(circle_obstacle_num=2)

        # simulation
        self.run_simulaiton(agents, boundary, solver, sim_time=4,
                            use_multi_thread=use_multi_thread, circle_obs=circle_obs,
                            adaptive_backup_weight=adaptive_backup_weight, hetero_solver=hetero_solver)

    def demo_T_intersection_2_agents(self, solver, use_multi_thread=False, adaptive_backup_weight=False,
                                     hetero_solver=None):
        """
        Trajectory planning with games, with obstacles

            - 2 agents
            - known others' goal
            - with obstacles (T-intersection)
        """
        # setting
        boundary, agent_list, circle_obs = self.T_intersection_setting()
        agents = MultiAgents(agent_list, agent_num=2)
        agents.update_limit(v_limit=[-10, 10], a_limit=[-15, 15], w_limit=[-np.pi / 2, np.pi / 2])

        # simulation
        self.run_simulaiton(agents, boundary, solver, sim_time=9.8,
                            use_multi_thread=use_multi_thread, circle_obs=circle_obs,
                            adaptive_backup_weight=adaptive_backup_weight, hetero_solver=hetero_solver)

    def demo_various_safety_weight(self, solver, use_multi_thread=False, adaptive_backup_weight=False,
                                   hetero_solver=None):
        # setting
        boundary, agent_list, circle_obs = self.various_safety_weight_setting()
        agents = MultiAgents(agent_list, agent_num=2)
        agents.update_limit(v_limit=[-10, 10], a_limit=[-15, 15], w_limit=[-np.pi / 2, np.pi / 2])

        # simulation
        self.run_simulaiton(agents, boundary, solver, sim_time=7,
                            use_multi_thread=use_multi_thread, circle_obs=circle_obs,
                            adaptive_backup_weight=adaptive_backup_weight, hetero_solver=hetero_solver)

    ###########################################
    # Random setting
    ###########################################

    def demo_random_open_case(self, solver, sim_time, agent_num=5, use_multi_thread=False, adaptive_backup_weight=False,
                              hetero_solver=False):
        """
        Open-case random setting


        The random part includes the agent number, agent size and safety distance.
        The agent size and safety distance will generate in a feasible range w.r.t
        the boundary range.

        Note that
        1. The boundary range is limited to |max(x, y)| = 20 & |min(x, y)| = 2.
        2. The vector start->goal and the direction of the goal will not be opposite.
        """
        # set the boundary range min_x, min_y = [-20, -2], max_x, max_y = [20, 2]
        boundary_range = onp.array([-5, -5, 5, 5])

        ##################################################
        # generate random setting
        #
        # Two type :
        # 1. agent is point mass
        # 2. agent is a circle which radius is not zero
        ##################################################
        agents, boundary, circle_obs = random_setting("open_case", boundary_range, agent_num=agent_num, Dis_weight=40)
        # agents, boundary, circle_obs = self.random_setting("open_case", boundary_range, agent_size_exist=True)

        save_env_img = "environment"
        draw_fxn.save_env_setting(agents, boundary_range, save_env_img)

        # simulation
        self.run_simulaiton(agents, boundary, solver, sim_time,
                            use_multi_thread=use_multi_thread, circle_obs=circle_obs,
                            adaptive_backup_weight=adaptive_backup_weight, hetero_solver=hetero_solver,
                            random_boundary=boundary_range)

    def demo_resolve_random_open_case(self, folder_name, use_multi_thread=False):
        """
        Reload the created random open-case setting and change the simulation part
        """
        # get folder name
        folder = "figs/" + folder_name

        # load npz file & rebuild agents, obstacles
        agents, circle_obs, boundary, horizon, boundary_range, solver_type, adaptive_backup_weight, _ = draw_fxn.load_info(
            folder)

        boundary = None

        # simulation
        self.run_simulaiton(agents, boundary, solver_type, sim_time=(horizon * agents.Ts),
                            use_multi_thread=use_multi_thread, circle_obs=circle_obs,
                            adaptive_backup_weight=adaptive_backup_weight, hetero_solver=False,
                            random_boundary=boundary_range)

    def demo_random_obstacle_case(self, boundary, circle_obs, solver, sim_time, agent_num=2,
                                  use_multi_thread=False, adaptive_backup_weight=False, hetero_solver=False):
        """
        Obstacle case with given scene

        @param boundary
        @param circle_obs

        Note: The start and goal will not be opposite when the problem type is "obs_narrow"
        """
        ##################################################
        # generate random setting
        #
        # Two type :
        # 1. free
        # 2. agent's start & goal always on the opposite
        #    side of the obstacle
        ##################################################
        # agents, boundary, circle_obs = self.random_setting(
        #     problem_type="obs",
        #     boundary_range=boundary,
        #     boundary_exist=True,
        #     circle_obstacle=circle_obs,
        #     agent_size_exist=True)

        agents, boundary, circle_obs = random_setting(
            problem_type="obs_narrow",
            boundary_range=boundary,
            boundary_exist=True,
            circle_obstacle=circle_obs,
            agent_size_exist=True,
            agent_num=agent_num,
            agent_size=0.4,
            Dis_weight=40)

        save_env_img = "environment"
        draw_fxn.save_env_setting(agents, boundary, save_env_img, boundary=boundary, circle_obs=circle_obs)

        # simulation
        self.run_simulaiton(agents, boundary, solver, sim_time,
                            use_multi_thread=use_multi_thread, circle_obs=circle_obs,
                            adaptive_backup_weight=adaptive_backup_weight, hetero_solver=hetero_solver,
                            random_boundary=boundary)

    def demo_resolve_random_obstacle_case(self, folder_name, use_multi_thread=False):
        """
        Reload the created random obstacle-case setting and change the simulation part
        """
        # get folder name
        folder = "figs/" + folder_name

        # load npz file & rebuild agents, obstacles
        agents, circle_obs, boundary, horizon, boundary_range, solver_type, adaptive_backup_weight, _ = draw_fxn.load_info(
            folder)

        print(solver_type)

        # simulation
        self.run_simulaiton(agents, boundary, solver_type, sim_time=(horizon * agents.Ts),
                            use_multi_thread=use_multi_thread, circle_obs=circle_obs,
                            adaptive_backup_weight=adaptive_backup_weight, hetero_solver=False,
                            random_boundary=boundary)

    ###############################################
    # Function for finishing the interrupted demo
    ###############################################

    def finish_closed_loop_demo(self, folder_name, use_multi_thread=False):
        # get folder name
        folder = "figs/" + folder_name

        # load npz file & rebuild agents, obstacles
        agents, circle_obs, boundary, horizon, boundary_range, solver_type, adaptive_backup_weight, hetero_solvers = draw_fxn.load_info(
            folder)
        sim_time = horizon * agents.Ts

        if boundary is None or boundary == None:
            boundary = None

        # find last step
        total_num, time_step_and_safety_dis = draw_fxn.count_total_simulation_time_and_check_reactive_safety_distance(
            folder, agents.agent_num)
        print('Total Num:', total_num)

        # rebuild agents
        print("rebuilding agents...")
        react_safety_dis = None
        if total_num > 0:
            agents, plot_boundary = draw_fxn.rebuild_agents(folder_name, total_num, agents, boundary, circle_obs)

            if time_step_and_safety_dis is not None:
                keysList = list(time_step_and_safety_dis.keys())
                react_safety_dis = onp.zeros((1, 1 + agents.agent_num))
                agents, react_safety_dis = update_react_param(agents, react_safety_dis, total_num)

                for i in range(1, len(keysList)):
                    if keysList[i] < total_num:
                        key = keysList[i]
                        new_safety_dis = onp.zeros((1, 1 + agents.agent_num))
                        new_safety_dis[0][0] = key
                        for j in range(agents.agent_num):
                            new_safety_dis[0][j + 1] = time_step_and_safety_dis[key][j]
                            agents.agent[j].safety_dis = time_step_and_safety_dis[key][j]
                        react_safety_dis = onp.append(react_safety_dis, new_safety_dis, axis=0)

        # simulation
        self.run_simulaiton(agents,
                            boundary,
                            solver_type,
                            sim_time=sim_time,
                            use_multi_thread=use_multi_thread,
                            circle_obs=circle_obs,
                            adaptive_backup_weight=adaptive_backup_weight,
                            hetero_solver=hetero_solvers,
                            random_boundary=boundary,
                            start_sim_num=np.size(agents.agent[0].trajectory, 0),
                            time_step_and_safety_dis=react_safety_dis)

    ###########################################
    # Parameters : Open Case
    ###########################################

    def open_case_2_agent_setting(self):
        # boundary setting
        boundary = None

        # agents setting
        agent_1 = SingleAgent(
            initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
            target_state=np.array([5.0, 0, 0, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=100,
            Back_weight=10,
            safety_dis=1.2)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([5, 0.00, np.pi, 0]),
            target_state=np.array([0, 0.0, np.pi, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=100,
            Back_weight=10,
            safety_dis=2.0)

        agent_list = [agent_1, agent_2]
        for agent in agent_list:
            agent.radius = 0

        return boundary, agent_list

    def open_case_3_agent_setting(self):
        """
        Problem setting : 3-agent, open case

        @return agent_list
        """
        # boundary setting
        boundary = None

        # agents setting
        agent_1 = SingleAgent(
            initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
            target_state=np.array([5.0, 0, 0, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=100,
            Back_weight=10,
            safety_dis=0.6)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([5, 0.00, np.pi, 0]),
            target_state=np.array([0, 0.0, np.pi, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=100,
            Back_weight=10,
            safety_dis=2.0)
        agent_3 = SingleAgent(
            initial_state=np.array([2.5, -2.0, np.pi / 2, 0]),
            target_state=np.array([2.5, 2.0, np.pi / 2, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=100,
            Back_weight=10,
            safety_dis=1.2)

        agent_list = [agent_1, agent_2, agent_3]

        for agent in agent_list:
            agent.radius = 0

        return boundary, agent_list

    ###########################################
    # Parameters : Narrow-way Case (2 agents)
    ###########################################

    def narrow_way_2_agent_setting1(self):
        """
        Problem setting : 2-agent, with obstacles

        @return agent_list
        """

        # agent setting
        agent_1 = SingleAgent(
            initial_state=np.array([2, 0.0, 0.0, 0.0]),
            target_state=np.array([7, 0, 0, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=1.2)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([6, 0.0, np.pi, 0]),
            target_state=np.array([1, 0.0, np.pi, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=2.0)

        agent_list = [agent_1, agent_2]

        return agent_list

    def narrow_way_2_agent_setting2(self):
        """
        Problem setting : 2-agent, with obstacles

        @return agent_list
        """

        # agent setting
        agent_1 = SingleAgent(
            initial_state=np.array([3.2, 0.0, 0.0, 0.0]),  # 3.2 2.5
            target_state=np.array([6.5, 0, 0, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=1.2)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([4.8, 0.0, np.pi, 0]),  # 4.8 4.3
            target_state=np.array([1.5, 0.0, np.pi, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=2.0)

        agent_list = [agent_1, agent_2]

        return agent_list

    def narrow_way_2_agent_setting3(self):
        """
        Problem setting : 2-agent, with obstacles

        @return agent_list
        """

        agent_1 = SingleAgent(
            initial_state=np.array([1, 1, 0, 0]),
            target_state=np.array([7.0, 0, 0, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=1.2)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([7, 1, np.pi, 0]),
            target_state=np.array([1, 0.0, np.pi, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=2.0)

        agent_list = [agent_1, agent_2]

        return agent_list

    def narrow_way_2_agent_setting4(self):
        """
        Problem setting : 2-agent, with obstacles

        @return agent_list
        """

        agent_1 = SingleAgent(
            initial_state=np.array([1, 1, 0, 0]),
            target_state=np.array([7.0, -1.0, 0, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=1.2)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([7, 1, np.pi, 0]),
            target_state=np.array([1.0, -1.0, np.pi, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=2.0)

        agent_list = [agent_1, agent_2]

        return agent_list

    def narrow_way_2_agent_setting5(self):
        """
        Problem setting : 2-agent, with obstacles

        @return agent_list
        """

        agent_1 = SingleAgent(
            initial_state=np.array([1.0, 0.5, 0.0, 0.0]),
            target_state=np.array([7.0, -1.0, 0.0, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=1.2)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([1.0, -0.5, 0, 0]),
            target_state=np.array([7.0, 1.0, 0.0, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=2.0)

        agent_list = [agent_1, agent_2]

        return agent_list

    def narrow_way_2_agent_setting6(self):
        """
        Problem setting : 2-agent, with obstacles

        @return agent_list
        """

        agent_1 = SingleAgent(
            initial_state=np.array([3.2, 0.0, 0.0, 0.0]),
            target_state=np.array([4.8, 0, 0, 0]),
            Q=np.array([0.0, 0.0, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=1.2)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([4.8, 0.0, np.pi, 0]),
            target_state=np.array([3.2, 0.0, np.pi, 0]),
            Q=np.array([0.0, 0.0, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=2.0)

        agent_list = [agent_1, agent_2]

        return agent_list

    def narrow_way_2_agent_setting7(self):
        """
        Problem setting : 2-agent, with obstacles

        @return agent_list
        """

        agent_1 = SingleAgent(
            initial_state=np.array([2.8, 0.0, 0.0, 0.0]),
            target_state=np.array([6.5, 0, 0, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=1.2)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([5.2, 0.0, np.pi, 0]),
            target_state=np.array([1.5, 0.0, np.pi, 0]),
            Q=np.array([0.01, 0.01, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=2.0)

        agent_list = [agent_1, agent_2]

        return agent_list

    def narrow_way_obstacle_setting(self, circle_obstacle_num):

        circle_obs = CirObstacles(obstacle_num=circle_obstacle_num,
                                  pos=[np.array([4.0, 2.5]), np.array([4.0, -2.5])],
                                  radius=[2, 2])

        return circle_obs

    ###########################################
    # Parameters : T-intersection Case
    ###########################################

    def T_intersection_setting(self):
        """
        Problem setting : 2-agent, with obstacles

        @return agent_list
        """
        # boundary setting
        boundary = onp.array([-2, -1.5, 10, 4])

        # agents setting
        agent_1 = SingleAgent(
            initial_state=np.array([1.0, -1.0, 0.0, 0.0]),
            target_state=np.array([8.0, -1.0, 0, 0]),
            Q=np.array([0.001, 0.001, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=10,
            Back_weight=10,
            safety_dis=1.2)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([4.0, 0.5, np.pi * 1.5, 0]),
            target_state=np.array([-1.0, -1.0, np.pi, 0]),
            Q=np.array([0.001, 0.001, 0, 0]),
            R=np.array([1, 1]),
            Dis_weight=10,
            Back_weight=10,
            safety_dis=2.0)

        agent_list = [agent_1, agent_2]

        # circle obstacles setting
        circle_obs = self.T_intersection_obstacle_setting(circle_obstacle_num=6)

        return boundary, agent_list, circle_obs

    def T_intersection_obstacle_setting(self, circle_obstacle_num):

        circle_obs = CirObstacles(obstacle_num=circle_obstacle_num,
                                  pos=[np.array([0.5, 2.5]), np.array([7.5, 2.5]),
                                       np.array([2.5, 0.5]), np.array([5.5, 0.5]),
                                       np.array([3.1, -0.2]), np.array([4.9, -0.2]),
                                       np.array([1.8, -0.2]), np.array([6.2, -0.2])],
                                  radius=[3, 3, 1.0, 1.0, 0.4, 0.4, 0.4, 0.4])

        return circle_obs

    ###########################################
    # Parameters : Various Safety weight D (2 agents)
    ###########################################

    def various_safety_weight_setting(self):
        """
        Problem setting : 2-agent, with obstacles

        @return agent_list
        """
        # boundary setting
        boundary = onp.array([-4, -2.5, 12, 2.5])

        # agent setting
        agent_1 = SingleAgent(
            initial_state=np.array([-0.5, 0.0, 0.0, 0.0]),
            target_state=np.array([8.5, 0.0, 0, 0]),
            Q=np.array([0.001, 0.001, 0, 0]),
            R=np.array([0.1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=1.2)  # 0.3
        agent_2 = SingleAgent(
            initial_state=np.array([8.5, 0.0, np.pi, 0]),
            target_state=np.array([-0.5, 0.0, np.pi, 0]),
            Q=np.array([0.001, 0.001, 0, 0]),
            R=np.array([0.1, 1]),
            Dis_weight=40,
            Back_weight=10,
            safety_dis=2.0)

        agent_list = [agent_1, agent_2]

        # circle obstacles setting
        circle_obs = self.various_safety_weight_2_obstacle_setting(circle_obstacle_num=4)

        return boundary, agent_list, circle_obs

    def various_safety_weight_2_obstacle_setting(self, circle_obstacle_num):

        circle_obs = CirObstacles(obstacle_num=circle_obstacle_num,
                                  pos=[np.array([1.5, 2.5]), np.array([1.5, -2.5]),
                                       np.array([6.5, 2.5]), np.array([6.5, -2.5])],
                                  radius=[1.5, 1.5, 1.5, 1.5])

        return circle_obs
