# test_multi_agent_problem.py
"""
Main Test

Created on 2023/3/14
@author: Pin-Yun Hung
"""

import simulation
import numpy as np


def test_open_case():
    solver_list = ["Centralized", "IPG_AdaptU", "Vanilla"]

    for solver in solver_list:
        ''' open case, 2 agents '''
        sim.demo_open_2_agents(solver=solver, use_multi_thread=True)

        ''' open case, 3 agents '''
        sim.demo_open_3_agents(solver=solver, use_multi_thread=True)


def test_narrow_way_case():
    solver_list = ["Centralized", "IPG", "IPG_AdaptU", "Vanilla", "Brake"]  # "IPG_AdaptU"

    for solver in solver_list:
        ''' narrow way, 2 agents '''
        sim.demo_narrow_way_2_agents(solver=solver, use_multi_thread=True)
        # sim.demo_narrow_way_2_agents(solver=solver, use_multi_thread=True, adaptive_backup_weight=True)


def test_narrow_way_case_hetero_solvers():
    """ narrow way, Hetero, 2 agents """
    agent_solver1 = {
        0: "IPG_React",
        1: "Vanilla"
    }

    agent_solver2 = {
        1: "IPG_React",
        0: "Ignore"
    }

    return agent_solver1, agent_solver2


def test_T_intersection_case():
    solver_list = ["Centralized", "IPG", "Vanilla"]

    for solver in solver_list:
        ''' T-intersection, 2 agents '''
        sim.demo_T_intersection_2_agents(solver=solver, use_multi_thread=True, adaptive_backup_weight=False)


def test_random_case():
    """
    random setting
    """

    ''' random setting, open case'''
    sim.demo_random_open_case(solver="IPG", sim_time=4, agent_num=3, use_multi_thread=True)
    # sim.demo_resolve_random_open_case(folder_name="3_agent_IPG_1012143602/")

    ''' random setting, obstacle case, 2 agents'''
    boundary = np.array([-2, -3, 10, 3])
    circle_obs = sim.narrow_way_obstacle_setting(circle_obstacle_num=2)
    sim.demo_random_obstacle_case(boundary, circle_obs, solver="IPG_AdaptU", sim_time=5, use_multi_thread=True)
    # sim.demo_resolve_random_obstacle_case(folder_name="2_agent_IPG_AdaptU_1012151832/")


if __name__ == '__main__':
    sim = simulation.Simulation()

    # test_open_case()

    test_narrow_way_case()
    # sim.demo_narrow_way_2_agents(solver="Centralized", adaptive_backup_weight=True)
    # agent_solver1, agent_solver2 = test_narrow_way_case_hetero_solvers()
    # sim.demo_narrow_way_2_agents_hetero_solver(solver="Hetero", hetero_solver=agent_solver1, use_multi_thread=True)
    # sim.demo_narrow_way_2_agents_hetero_solver(solver="Hetero", hetero_solver=agent_solver2, use_multi_thread=True)

    # test_T_intersection_case()

    '''Test Finish planning function'''
    # sim.finish_closed_loop_demo(
    #     folder_name="2_agent_Centralized_1014022515/")
