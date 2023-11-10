# cloosed_loop_control.py
"""
Closed loop control

Created on 2023/5/14
@author: Pin-Yun Hung
"""

import ilqr_trajectory_planner
from draw_functions import DrawClass
from utils import *
import jax.numpy as np
import numpy as onp
import time
import copy
import math
import concurrent.futures

from typing import Any

PyTree = Any
THREAD_NUM = 5
SOL_FEAS_THRESHOLD = 0.15
draw_fxn = DrawClass()


def sim_centralized(agents, sim_time, sim_num, random_boundary=None, boundary=None, circle_obstacle=None,
                         start_sim_num=0, adaptive_backup_weight=False, draw_timestep_plot=True):
    sim_name = "Centralized"

    # create folder
    fig_folder, folder = create_folder(agents.agent_num, sim_name)

    # parameter
    Ts, N, constraints_threshold, boundary_info, boundary, plot_boundary = initialize_params(agents, sim_time, boundary)

    # save information
    save_info(
        folder_name=fig_folder,
        agents=agents,
        boundary=boundary_info,
        horizon=N,
        solver_type=sim_name,
        constraints_threshold=constraints_threshold,
        circle_obs=circle_obstacle,
        boundary_range=random_boundary,
        adaptive_backup_weight=adaptive_backup_weight)

    # save environment plot
    draw_fxn.save_env_setting(agents, plot_boundary, folder + "environment", boundary=boundary_info,
                              circle_obs=circle_obstacle)

    ######################################
    # Solve with iLQR
    ######################################

    # update U reference
    for i in range(agents.agent_num):
        agents.agent[i].control_opt_series = np.zeros((N, agents.input_dim * agents.agent_num))

    # create solver parameters
    total_error, error_list = calc_total_distance_error(agents)
    solved_time = onp.zeros((1, 1))
    if start_sim_num > 0:
        solved_time = onp.zeros((start_sim_num + 1, 1))
    cost_args = generate_cost_args(agents)

    # set the starting and stopping time step
    time_step = start_sim_num
    stop_num = sim_num

    closed_loop_start = time.perf_counter()
    while time_step < stop_num:

        if adaptive_backup_weight:
            agents = update_cost_args_with_adaptive_backup_weight(agents, error_list)
            cost_args = generate_cost_args(agents)

        start = time.perf_counter()

        sol = ilqr_trajectory_planner.solve_centralized(
            U=agents.agent[0].control_opt_series,
            agents=agents,
            N=N,
            Ts=Ts,
            cost_args=cost_args,
            constraints_threshold=constraints_threshold,
            boundary=boundary_info,
            circle_obs=circle_obstacle)

        end = time.perf_counter()

        # check solution feasibility
        sol_feasibility = check_solution_collision(N, sol[0], agents, obs=circle_obstacle)

        # check collision
        no_collision = check_collision(agents, constraints_threshold=SOL_FEAS_THRESHOLD, obs=circle_obstacle)

        # update agent position, solution series
        # store solution for regenerating same closed-loop simulation
        for i in range(agents.agent_num):
            store_open_loop_solution_npz(fig_folder, i, sol, time_step)
            agents.agent[i].update(sol, i)
            agents.agent[i].position = agents.agent[i].next_step_prediction
        # update total error
        total_error, error_list = calc_total_distance_error(agents)

        # # update plot boundary
        plot_boundary = draw_fxn.update_plot_boundary(plot_boundary, agents, sol[0])
        # save img
        if draw_timestep_plot:
            draw_fxn.save_timestep_plot(agents, plot_boundary, folder, time_step,
                                        boundary=boundary, circle_obs=circle_obstacle, predict_other=False)

        # get iLQR solver compute cost
        cost = sol[8]

        solver_cost_time = end - start
        solved_time[time_step, 0] = solver_cost_time
        solved_time = onp.append(solved_time, np.zeros((1, 1)), axis=0)

        other_end = time.perf_counter()
        other_cost_time = (other_end - start) - solver_cost_time

        print(
            f"time_step = {time_step}  solved = {solver_cost_time: .3f} (sec) | other = {other_cost_time: .3f} (sec) | err = {total_error:.3f} | cost = {cost:.2f} | iter_ilqr = {sol[10]} | iter_al = {sol[11]}")
        time_step += 1

        # check stop conditions
        if total_error < 1.0:
            print(f"Finished simulation !!  ( Total distance remaining to the goal = {total_error: .3f}  (m) )")
            break
        elif time_step > 1 and sol_feasibility == False:
            print(f"Cannot find feasible solution !")
            break
        elif no_collision == False:
            print(f"Collision happened !")
            break

    sim_num = time_step
    end = time.perf_counter()
    print("Total Execute Time : {} sec".format((end - closed_loop_start)))

    ######################################
    # Plot & Save
    ######################################

    start = time.perf_counter()
    # write to file
    store_solved_time(fig_folder, solved_time=solved_time)
    draw_fxn.save_solver_cost_time_plot(sim_name, solved_time, folder)

    # save plot at each time-step
    draw_fxn.save_multi_plot(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle,
                             predict_other=False)
    # save animate
    draw_fxn.save_animate(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle,
                          predict_other=False, axis_off=True)
    # save closed loop process plot
    img_list = draw_fxn.generate_constant_interval_img_list(sim_num, horizon=N, list_length=4)
    draw_fxn.save_closed_loop_process_plot(img_list, agents, plot_boundary, folder, boundary, circle_obstacle,
                                           predict_other=False)

    end = time.perf_counter()
    print("Plotting Time : {} sec".format((end - start)))

    print("... Done !")


def sim_IPG(agents, sim_time, random_boundary=None, boundary=None, circle_obstacle=None,
                 sim_num=None, start_sim_num=0, use_multi_thread=True, adaptive_backup_weight=False, draw_timestep_plot=True):
    sim_name = "IPG"

    # create folder
    fig_folder, folder = create_folder(agents.agent_num, sim_name)

    # parameter
    Ts, N, constraints_threshold, boundary_info, boundary, plot_boundary = initialize_params(agents, sim_time, boundary)

    # save information
    save_info(
        folder_name=fig_folder,
        agents=agents,
        boundary=boundary_info,
        horizon=N,
        solver_type=sim_name,
        constraints_threshold=constraints_threshold,
        circle_obs=circle_obstacle,
        boundary_range=random_boundary,
        adaptive_backup_weight=adaptive_backup_weight)

    # save environment plot
    draw_fxn.save_env_setting(agents, plot_boundary, folder + "environment", boundary=boundary_info,
                              circle_obs=circle_obstacle)

    ######################################
    # Solve with iLQR
    ######################################

    # create thread if using multi-thread
    thread_pool = None
    if use_multi_thread:
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_NUM)

    # update U reference
    for i in range(agents.agent_num):
        agents.agent[i].control_opt_series = np.zeros((N, agents.input_dim * agents.agent_num))

    # create solver parameters
    total_error, error_list = calc_total_distance_error(agents)
    solved_time = onp.zeros((1, agents.agent_num))
    if start_sim_num > 0:
        solved_time = onp.zeros((start_sim_num + 1, agents.agent_num))

    # set the starting and stopping time step
    time_step = start_sim_num
    stop_num = N
    if sim_num is not None:
        stop_num = sim_num

    total_start = time.perf_counter()
    while (time_step <= stop_num):

        if adaptive_backup_weight:
            agents = update_cost_args_with_adaptive_backup_weight(agents, error_list)

        cost = 0.0
        start = time.perf_counter()

        if use_multi_thread:
            run_thread_num = math.ceil(agents.agent_num / THREAD_NUM)
            agent_count = 0
            for j in range(run_thread_num):
                thread_num_start = THREAD_NUM * j
                thread_num_end = THREAD_NUM * (j + 1)
                if thread_num_end > agents.agent_num:
                    thread_num_end = agents.agent_num

                results = [thread_pool.submit(single_agent_plan_IPG,
                                              agents, i, circle_obstacle, N, Ts, constraints_threshold,
                                              boundary_info, fig_folder, time_step)
                           for i in range(thread_num_start, thread_num_end)]

                concurrent.futures.wait(results)

                # calc cost
                for result in results:
                    cost = cost + result.result()[0]
                    solved_time[time_step, agent_count] = result.result()[1]
                    agent_count += 1
        else:
            for i in range(agents.agent_num):
                single_cost, solve_time = single_agent_plan_IPG(agents, i, circle_obstacle, N, Ts,
                                                                     constraints_threshold,
                                                                     boundary_info, fig_folder, time_step)

                # calc cost
                cost += single_cost
                solved_time[time_step, i] = solve_time

        # check collision
        no_collision = check_collision(agents, constraints_threshold=SOL_FEAS_THRESHOLD, obs=circle_obstacle)
        # update plot boundary & agent position
        for i in range(agents.agent_num):
            x_opt = agents.agent[i].sol_series[-1][0]
            plot_boundary = draw_fxn.update_plot_boundary(plot_boundary, agents, x_opt)
            agents.agent[i].position = agents.agent[i].next_step_prediction
        # update total error
        total_error, error_list = calc_total_distance_error(agents)

        # save img
        if draw_timestep_plot:
            draw_fxn.save_timestep_plot(agents, plot_boundary, folder, time_step,
                                        boundary=boundary, circle_obs=circle_obstacle, predict_other=True)

        end = time.perf_counter()

        solved_time = onp.append(solved_time, np.zeros((1, agents.agent_num)), axis=0)
        solver_cost_time = max(solved_time[time_step, :])
        other_cost_time = (end - start) - solver_cost_time

        print(
            f"time_step = {time_step}  solved = {solver_cost_time: .3f} (sec) | other = {other_cost_time: .3f} (sec) | err = {total_error:.3f} | cost = {cost:.3f}")
        time_step += 1

        if total_error < 1.0 or no_collision == False:
            break

    sim_num = time_step
    total_end = time.perf_counter()
    print("Total Execute Time : {} sec".format((total_end - total_start)))

    ######################################
    # Plot & Save
    ######################################
    start = time.perf_counter()

    store_solved_time(fig_folder, solved_time=solved_time)
    draw_fxn.save_solver_cost_time_plot(sim_name, solved_time, folder)

    draw_fxn.save_multi_plot(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle)

    # save animate
    draw_fxn.save_animate(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle)

    # save closed loop process plot
    img_list = draw_fxn.generate_constant_interval_img_list(sim_num, horizon=N, list_length=4)
    draw_fxn.save_closed_loop_process_plot(img_list, agents, plot_boundary, folder, boundary, circle_obstacle)

    end = time.perf_counter()
    print("Plotting Time : {} sec".format((end - start)))
    print("... Done !")


def sim_IPG_AdaptU(agents, sim_time, random_boundary=None, boundary=None, circle_obstacle=None,
                        sim_num=None, start_sim_num=0, use_multi_thread=True, adaptive_backup_weight=False, draw_timestep_plot=True):
    sim_name = "IPG_AdaptU"

    # create folder
    fig_folder, folder = create_folder(agents.agent_num, sim_name)

    # parameter
    Ts, N, constraints_threshold, boundary_info, boundary, plot_boundary = initialize_params(agents, sim_time, boundary)

    # save information
    save_info(
        folder_name=fig_folder,
        agents=agents,
        boundary=boundary_info,
        horizon=N,
        solver_type=sim_name,
        constraints_threshold=constraints_threshold,
        circle_obs=circle_obstacle,
        boundary_range=random_boundary,
        adaptive_backup_weight=adaptive_backup_weight)

    # save environment plot
    draw_fxn.save_env_setting(agents, plot_boundary, folder + "environment", boundary=boundary_info,
                              circle_obs=circle_obstacle)

    ######################################
    # Solve with iLQR
    ######################################

    # create thread if using multi-thread
    thread_pool = None
    if use_multi_thread:
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_NUM)

    # update U reference
    for i in range(agents.agent_num):
        agents.agent[i].control_opt_series = np.zeros((N, agents.input_dim * agents.agent_num))

    # create solver parameters
    total_error, error_list = calc_total_distance_error(agents)
    solved_time = onp.zeros((1, agents.agent_num))
    if start_sim_num > 0:
        solved_time = onp.zeros((start_sim_num + 1, agents.agent_num))

    # set the starting and stopping time step
    time_step = start_sim_num
    stop_num = N
    if sim_num is not None:
        stop_num = sim_num

    total_start = time.perf_counter()
    while (time_step <= stop_num):

        if adaptive_backup_weight:
            agents = update_cost_args_with_adaptive_backup_weight(agents, error_list)

        cost = 0.0
        start = time.perf_counter()

        if use_multi_thread:
            run_thread_num = math.ceil(agents.agent_num / THREAD_NUM)
            agent_count = 0
            for j in range(run_thread_num):
                thread_num_start = THREAD_NUM * j
                thread_num_end = THREAD_NUM * (j + 1)
                if thread_num_end > agents.agent_num:
                    thread_num_end = agents.agent_num

                results = [thread_pool.submit(single_agent_plan_IPG_adaptU,
                                              agents, i, circle_obstacle, N, Ts, constraints_threshold,
                                              boundary_info, fig_folder, time_step)
                           for i in range(thread_num_start, thread_num_end)]

                concurrent.futures.wait(results)

                # calc cost
                for result in results:
                    cost = cost + result.result()[0]
                    solved_time[time_step, agent_count] = result.result()[1]
                    agent_count += 1
        else:
            for i in range(agents.agent_num):
                single_cost, solve_time = single_agent_plan_IPG_adaptU(agents, i, circle_obstacle, N, Ts,
                                                                            constraints_threshold,
                                                                            boundary_info, fig_folder, time_step)

                # calc cost
                cost += single_cost
                solved_time[time_step, i] = solve_time

        # check collision
        no_collision = check_collision(agents, constraints_threshold=SOL_FEAS_THRESHOLD, obs=circle_obstacle)
        # update plot boundary & agent position
        for i in range(agents.agent_num):
            x_opt = agents.agent[i].sol_series[-1][0]
            plot_boundary = draw_fxn.update_plot_boundary(plot_boundary, agents, x_opt)
            agents.agent[i].position = agents.agent[i].next_step_prediction
        # update total error
        total_error, error_list = calc_total_distance_error(agents)

        # save img
        if draw_timestep_plot:
            draw_fxn.save_timestep_plot(agents, plot_boundary, folder, time_step,
                                        boundary=boundary, circle_obs=circle_obstacle, predict_other=True)

        end = time.perf_counter()

        solved_time = onp.append(solved_time, np.zeros((1, agents.agent_num)), axis=0)
        solver_cost_time = max(solved_time[time_step, :])
        other_cost_time = (end - start) - solver_cost_time

        print(
            f"time_step = {time_step}  solved = {solver_cost_time: .3f} (sec) | other = {other_cost_time: .3f} (sec) | err = {total_error:.3f} | cost = {cost:.3f}")
        time_step += 1

        if total_error < 1.0 or no_collision == False:
            break

    sim_num = time_step
    total_end = time.perf_counter()
    print("Total Execute Time : {} sec".format((total_end - total_start)))

    ######################################
    # Plot & Save
    ######################################
    start = time.perf_counter()

    store_solved_time(fig_folder, solved_time=solved_time)
    draw_fxn.save_solver_cost_time_plot(sim_name, solved_time, folder)

    draw_fxn.save_multi_plot(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle)

    # save animate
    draw_fxn.save_animate(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle)

    # save closed loop process plot
    img_list = draw_fxn.generate_constant_interval_img_list(sim_num, horizon=N, list_length=4)
    draw_fxn.save_closed_loop_process_plot(img_list, agents, plot_boundary, folder, boundary, circle_obstacle)

    end = time.perf_counter()
    print("Plotting Time : {} sec".format((end - start)))
    print("... Done !")


def sim_Vanilla(agents, sim_time, random_boundary=None, boundary=None, circle_obstacle=None,
            sim_num=None, start_sim_num=0, use_multi_thread=True, draw_timestep_plot=True):
    sim_name = "Vanilla"

    # create folder
    fig_folder, folder = create_folder(agents.agent_num, sim_name)

    # parameter
    Ts, N, constraints_threshold, boundary_info, boundary, plot_boundary = initialize_params(agents, sim_time, boundary)

    # save information
    save_info(
        folder_name=fig_folder,
        agents=agents,
        boundary=boundary_info,
        horizon=N,
        solver_type=sim_name,
        constraints_threshold=constraints_threshold,
        circle_obs=circle_obstacle,
        boundary_range=random_boundary)

    # save environment plot
    draw_fxn.save_env_setting(agents, plot_boundary, folder + "environment", boundary=boundary_info,
                              circle_obs=circle_obstacle)

    ######################################
    # Solve with iLQR
    ######################################

    # create thread if using multi-thread
    thread_pool = None
    if use_multi_thread:
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_NUM)

    # update U reference
    for i in range(agents.agent_num):
        agents.agent[i].control_opt_series = np.zeros((N, agents.input_dim))

    # create solver parameters
    total_error, error_list = calc_total_distance_error(agents)
    solved_time = onp.zeros((1, agents.agent_num))
    if start_sim_num > 0:
        solved_time = onp.zeros((start_sim_num + 1, agents.agent_num))

    # set the starting and stopping time step
    time_step = start_sim_num
    stop_num = N
    if sim_num is not None:
        stop_num = sim_num

    total_start = time.perf_counter()
    while (time_step <= stop_num):

        # generate prediction
        start = time.perf_counter()
        agents = create_vanilla_prediction(N, Ts, agents, boundary_info, circle_obstacle)
        # end = time.perf_counter()
        # print(f"Create single path : {(end - start): .3f} (sec)")

        cost = 0.0
        # start = time.perf_counter()

        if use_multi_thread:
            run_thread_num = math.ceil(agents.agent_num / THREAD_NUM)
            agent_count = 0
            for j in range(run_thread_num):
                thread_num_start = THREAD_NUM * j
                thread_num_end = THREAD_NUM * (j + 1)
                if thread_num_end > agents.agent_num:
                    thread_num_end = agents.agent_num

                results = [thread_pool.submit(single_agent_plan_Vanilla,
                                              agents, i, circle_obstacle, N, Ts, constraints_threshold,
                                              boundary_info, fig_folder, time_step)
                           for i in range(thread_num_start, thread_num_end)]

                concurrent.futures.wait(results)

                # calc cost
                for result in results:
                    cost = cost + result.result()[0]
                    solved_time[time_step, agent_count] = result.result()[1]
                    agent_count += 1
        else:
            for i in range(agents.agent_num):
                single_cost, solve_time = single_agent_plan_Vanilla(agents, i, circle_obstacle, N, Ts, constraints_threshold, boundary_info, fig_folder, time_step)

                # calc cost
                cost += single_cost
                solved_time[time_step, i] = solve_time

        # check collision
        no_collision = check_collision(agents, constraints_threshold=0.15, obs=circle_obstacle)
        if time_step > 0 and no_collision:
            no_collision = check_solution_stay_in_boundary(agents, N, boundary=boundary)

        # update plot boundary & agent position
        for i in range(agents.agent_num):
            x_opt = agents.agent[i].sol_series[-1][0]
            plot_boundary = draw_fxn.update_plot_boundary(plot_boundary, agents, x_opt)

            agents.agent[i].position = agents.agent[i].next_step_prediction
        # update total error
        total_error, error_list = calc_total_distance_error(agents)

        # save img
        if draw_timestep_plot:
            draw_fxn.save_timestep_plot(agents, plot_boundary, folder, time_step,
                                        boundary=boundary, circle_obs=circle_obstacle, predict_other=True)

        end = time.perf_counter()

        solved_time = onp.append(solved_time, np.zeros((1, agents.agent_num)), axis=0)
        solver_cost_time = max(solved_time[time_step, :])
        other_cost_time = (end - start) - solver_cost_time

        print(
            f"time_step = {time_step}  solved = {solver_cost_time: .3f} (sec) | other = {other_cost_time: .3f} (sec) | err = {total_error:.3f} | cost = {cost:.3f}")
        time_step += 1

        if total_error < 1.0 or no_collision == False:
            break

    sim_num = time_step
    total_end = time.perf_counter()
    print("Total Execute Time : {} sec".format((total_end - total_start)))

    ######################################
    # Plot & Save
    ######################################
    start = time.perf_counter()

    store_solved_time(fig_folder, solved_time=solved_time)
    draw_fxn.save_solver_cost_time_plot(sim_name, solved_time, folder)

    draw_fxn.save_multi_plot(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle)

    # save animate
    draw_fxn.save_animate(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle)

    # save closed loop process plot
    img_list = draw_fxn.generate_constant_interval_img_list(sim_num, horizon=N, list_length=4)
    draw_fxn.save_closed_loop_process_plot(img_list, agents, plot_boundary, folder, boundary, circle_obstacle)

    end = time.perf_counter()
    print("Plotting Time : {} sec".format((end - start)))
    print("... Done !")


def sim_Brake(agents, sim_time, random_boundary=None, boundary=None, circle_obstacle=None,
            sim_num=None, start_sim_num=0, use_multi_thread=True, draw_timestep_plot=True):
    sim_name = "Brake"

    # create folder
    fig_folder, folder = create_folder(agents.agent_num, sim_name)

    # parameter
    Ts, N, constraints_threshold, boundary_info, boundary, plot_boundary = initialize_params(agents, sim_time, boundary)

    # save information
    save_info(
        folder_name=fig_folder,
        agents=agents,
        boundary=boundary_info,
        horizon=N,
        solver_type=sim_name,
        constraints_threshold=constraints_threshold,
        circle_obs=circle_obstacle,
        boundary_range=random_boundary)

    # save environment plot
    draw_fxn.save_env_setting(agents, plot_boundary, folder + "environment", boundary=boundary_info,
                              circle_obs=circle_obstacle)

    ######################################
    # Solve with iLQR
    ######################################

    # create thread if using multi-thread
    thread_pool = None
    if use_multi_thread:
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_NUM)

    # update U reference
    for i in range(agents.agent_num):
        agents.agent[i].control_opt_series = np.zeros((N, agents.input_dim))

    # create solver parameters
    total_error, error_list = calc_total_distance_error(agents)
    solved_time = onp.zeros((1, agents.agent_num))
    if start_sim_num > 0:
        solved_time = onp.zeros((start_sim_num + 1, agents.agent_num))

    # set the starting and stopping time step
    time_step = start_sim_num
    stop_num = N
    if sim_num is not None:
        stop_num = sim_num

    total_start = time.perf_counter()
    while (time_step <= stop_num):

        # generate prediction
        start = time.perf_counter()
        agents = create_vanilla_prediction(N, Ts, agents, boundary_info, circle_obstacle)
        # end = time.perf_counter()
        # print(f"Create single path : {(end - start): .3f} (sec)")

        cost = 0.0
        # start = time.perf_counter()

        if use_multi_thread:
            run_thread_num = math.ceil(agents.agent_num / THREAD_NUM)
            agent_count = 0
            for j in range(run_thread_num):
                thread_num_start = THREAD_NUM * j
                thread_num_end = THREAD_NUM * (j + 1)
                if thread_num_end > agents.agent_num:
                    thread_num_end = agents.agent_num

                results = [thread_pool.submit(single_agent_plan_Brake,
                                              agents, i, circle_obstacle, N, Ts, constraints_threshold,
                                              boundary_info, fig_folder, time_step)
                           for i in range(thread_num_start, thread_num_end)]

                concurrent.futures.wait(results)

                # calc cost
                for result in results:
                    cost = cost + result.result()[0]
                    solved_time[time_step, agent_count] = result.result()[1]
                    agent_count += 1
        else:
            for i in range(agents.agent_num):
                single_cost, solve_time = single_agent_plan_Brake(agents, i, circle_obstacle, N, Ts, constraints_threshold, boundary_info, fig_folder, time_step)

                # calc cost
                cost += single_cost
                solved_time[time_step, i] = solve_time

        # check collision
        no_collision = check_collision(agents, constraints_threshold=0.15, obs=circle_obstacle)
        if time_step > 0 and no_collision:
            no_collision = check_solution_stay_in_boundary(agents, N, boundary=boundary)

        # update plot boundary & agent position
        for i in range(agents.agent_num):
            x_opt = agents.agent[i].sol_series[-1][0]
            plot_boundary = draw_fxn.update_plot_boundary(plot_boundary, agents, x_opt)

            agents.agent[i].position = agents.agent[i].next_step_prediction
        # update total error
        total_error, error_list = calc_total_distance_error(agents)

        # save img
        if draw_timestep_plot:
            draw_fxn.save_timestep_plot(agents, plot_boundary, folder, time_step,
                                        boundary=boundary, circle_obs=circle_obstacle, predict_other=True)

        end = time.perf_counter()

        solved_time = onp.append(solved_time, np.zeros((1, agents.agent_num)), axis=0)
        solver_cost_time = max(solved_time[time_step, :])
        other_cost_time = (end - start) - solver_cost_time

        print(
            f"time_step = {time_step}  solved = {solver_cost_time: .3f} (sec) | other = {other_cost_time: .3f} (sec) | err = {total_error:.3f} | cost = {cost:.3f}")
        time_step += 1

        if total_error < 1.0 or no_collision == False:
            break

    sim_num = time_step
    total_end = time.perf_counter()
    print("Total Execute Time : {} sec".format((total_end - total_start)))

    ######################################
    # Plot & Save
    ######################################
    start = time.perf_counter()

    store_solved_time(fig_folder, solved_time=solved_time)
    draw_fxn.save_solver_cost_time_plot(sim_name, solved_time, folder)

    draw_fxn.save_multi_plot(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle)

    # save animate
    draw_fxn.save_animate(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle)

    # save closed loop process plot
    img_list = draw_fxn.generate_constant_interval_img_list(sim_num, horizon=N, list_length=4)
    draw_fxn.save_closed_loop_process_plot(img_list, agents, plot_boundary, folder, boundary, circle_obstacle)

    end = time.perf_counter()
    print("Plotting Time : {} sec".format((end - start)))
    print("... Done !")


def sim_Heterogeneous(agents, hetero_solver, sim_time, random_boundary=None, boundary=None, circle_obstacle=None,
            sim_num=None, start_sim_num=0, use_multi_thread=True, adaptive_backup_weight=False, time_step_and_safety_dis=None, draw_timestep_plot=True):
    sim_name = "Hetero"

    # create folder
    fig_folder, folder = create_folder(agents.agent_num, sim_name)

    # parameter
    Ts, N, constraints_threshold, boundary_info, boundary, plot_boundary = initialize_params(agents, sim_time, boundary)

    # save information
    save_info(
        folder_name=fig_folder,
        agents=agents,
        boundary=boundary_info,
        horizon=N,
        solver_type=sim_name,
        constraints_threshold=constraints_threshold,
        circle_obs=circle_obstacle,
        boundary_range=random_boundary,
        agents_use_games=hetero_solver,
        adaptive_backup_weight=adaptive_backup_weight)

    # save environment plot
    draw_fxn.save_env_setting(agents, plot_boundary, folder + "environment", boundary=boundary_info,
                              circle_obs=circle_obstacle)

    ######################################
    # Solve with iLQR
    ######################################

    # create thread if using multi-thread
    thread_pool = None
    if use_multi_thread:
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_NUM)

    # update U reference
    for i in range(agents.agent_num):
        solver_name = hetero_solver[i].split('_')[0]
        if solver_name == "IPG":
            agents.agent[i].control_opt_series = np.zeros((N, agents.input_dim * agents.agent_num))
        else:
            agents.agent[i].control_opt_series = np.zeros((N, agents.input_dim))

    # create solver parameters
    total_error, error_list = calc_total_distance_error(agents)
    solved_time = onp.zeros((1, agents.agent_num))
    if start_sim_num > 0:
        solved_time = onp.zeros((start_sim_num + 1, agents.agent_num))

    # reactive parameters
    if time_step_and_safety_dis is None:
        time_step_and_safety_dis = onp.zeros((1, 1 + agents.agent_num))
        agents, time_step_and_safety_dis = update_react_param(agents, time_step_and_safety_dis, start_sim_num)

    # set the starting and stopping time step
    time_step = start_sim_num
    stop_num = N
    if sim_num is not None:
        stop_num = sim_num

    total_start = time.perf_counter()
    while (time_step <= stop_num):

        if adaptive_backup_weight:
            for i in range(agents.agent_num):
                if hetero_solver[i] == "IPG" or hetero_solver[i] == "IPG_AdaptU":
                    start_p = [agents.agent[i].initial_state[0], agents.agent[i].initial_state[1]]
                    end_p = [agents.agent[i].target_state[0], agents.agent[i].target_state[1]]
                    dis = math.dist(start_p, end_p) * 0.9
                    if error_list[i] < dis:
                        agents.agent[i].B_cost = 100
                    else:
                        agents.agent[i].B_cost = 0

        # generate prediction
        start = time.perf_counter()
        agents = create_vanilla_prediction(N, Ts, agents, boundary_info, circle_obstacle, hetero_solver=hetero_solver)

        cost = 0.0

        if use_multi_thread:
            run_thread_num = math.ceil(agents.agent_num / THREAD_NUM)
            agent_count = 0
            for j in range(run_thread_num):
                thread_num_start = THREAD_NUM * j
                thread_num_end = THREAD_NUM * (j + 1)
                if thread_num_end > agents.agent_num:
                    thread_num_end = agents.agent_num

                results = [thread_pool.submit(single_agent_planner,
                                              hetero_solver[i],
                                              agents, i, circle_obstacle, N, Ts, constraints_threshold,
                                              boundary_info, fig_folder, time_step)
                           for i in range(thread_num_start, thread_num_end)]

                concurrent.futures.wait(results)

                # calc cost
                for result in results:
                    cost = cost + result.result()[0]
                    solved_time[time_step, agent_count] = result.result()[1]
                    agent_count += 1
        else:
            for i in range(agents.agent_num):
                single_cost, solve_time = single_agent_planner(hetero_solver[i], agents, i, circle_obstacle, N, Ts, constraints_threshold,
                                          boundary_info, fig_folder, time_step)

                # calc cost
                cost += single_cost
                solved_time[time_step, i] = solve_time

        # update safety distance
        time_step_and_safety_dis = update_safety_distance_info(time_step, agents, time_step_and_safety_dis)

        # check collision
        no_collision = check_collision(agents, constraints_threshold=0.15, obs=circle_obstacle)
        if time_step > 0 and no_collision:
            no_collision = check_solution_stay_in_boundary(agents, N, boundary=boundary)
        # update plot boundary & agent position
        for i in range(agents.agent_num):
            x_opt = agents.agent[i].sol_series[-1][0]
            plot_boundary = draw_fxn.update_plot_boundary(plot_boundary, agents, x_opt)

            agents.agent[i].position = agents.agent[i].next_step_prediction
        # update total error
        total_error, error_list = calc_total_distance_error(agents)

        # save img
        if draw_timestep_plot:
            draw_fxn.save_timestep_plot(agents, plot_boundary, folder, time_step,
                                        boundary=boundary, circle_obs=circle_obstacle, predict_other=True)

        end = time.perf_counter()

        solved_time = onp.append(solved_time, np.zeros((1, agents.agent_num)), axis=0)
        solver_cost_time = max(solved_time[time_step, :])
        other_cost_time = (end - start) - solver_cost_time

        print(
            f"time_step = {time_step}  solved = {solver_cost_time: .3f} (sec) | other = {other_cost_time: .3f} (sec) | err = {total_error:.3f} | cost = {cost:.3f}")
        time_step += 1

        if total_error < 1.0 or no_collision == False:
            break

    sim_num = time_step

    if np.size(time_step_and_safety_dis, 0) > 1:
        save_reactive_safety_distance(folder_name=fig_folder, time_step_and_safety_dis=time_step_and_safety_dis)
        safety_dis_list = {}
        for i in range(np.size(time_step_and_safety_dis, 0)):
            safety_dis_list[int(time_step_and_safety_dis[i, 0])] = np.array(time_step_and_safety_dis[i, 1:])
    else:
        safety_dis_list = None

    total_end = time.perf_counter()
    print("Total Execute Time : {} sec".format((total_end - total_start)))

    ######################################
    # Plot & Save
    ######################################
    start = time.perf_counter()

    store_solved_time(fig_folder, solved_time=solved_time)
    draw_fxn.save_solver_cost_time_plot(sim_name, solved_time, folder)

    draw_fxn.save_multi_plot(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle,
                             time_step_and_safety_dis=safety_dis_list)

    # save animate
    draw_fxn.save_animate(agents, plot_boundary, folder, sim_num, boundary=boundary, circle_obs=circle_obstacle,
                          time_step_and_safety_dis=safety_dis_list)

    # save closed loop process plot
    img_list = draw_fxn.generate_constant_interval_img_list(sim_num, horizon=N, list_length=4)
    draw_fxn.save_closed_loop_process_plot(img_list, agents, plot_boundary, folder, boundary, circle_obstacle, time_step_and_safety_dis=safety_dis_list)

    end = time.perf_counter()
    print("Plotting Time : {} sec".format((end - start)))
    print("... Done !")


##################################

def single_agent_planner(solver, agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold, boundary_info,
                         fig_folder, time_step):

    if solver == "IPG":
        return single_agent_plan_IPG(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold, boundary_info, fig_folder, time_step)
    elif solver == "Vanilla":
        return single_agent_plan_Vanilla(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold,
                                     boundary_info, fig_folder, time_step)
    elif solver == "Brake":
        return single_agent_plan_Brake(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold,
                                     boundary_info, fig_folder, time_step)
    elif solver == "IPG_AdaptU":
        return single_agent_plan_IPG_adaptU(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold,
                                           boundary_info, fig_folder, time_step)
    elif solver == "IPG_React":
        return single_agent_plan_IPG_ReactSafety(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold,
                                           boundary_info, fig_folder, time_step)
    elif solver == "Ignore":
        return single_agent_plan_Ignore(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold,
                                             boundary_info, fig_folder, time_step)


def single_agent_plan_IPG(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold, boundary_info,
                          fig_folder, time_step):
    N = horizon
    i = agent_id

    agents_tmp = copy.deepcopy(agents)
    # check how many agents can be seen
    sense_list, new_agent_id = sense_agent(agent_id=i, agents=agents_tmp, circle_obs=circle_obstacle)

    # build new multi-agents based on the sensor
    agents_tmp.agent[i].rebuild_sense_control_opt_series(sense_list, N)
    agents_tmp.agent_num = len(sense_list)

    new_agents_tmp = copy.deepcopy(agents_tmp)
    new_agents_tmp.agent = []
    for sense_id in sense_list:
        new_agents_tmp.agent.append(agents_tmp.agent[sense_id])

    U = new_agents_tmp.agent[new_agent_id].control_opt_series
    if not agents.agent[i].feasibility:
        U = np.zeros(np.shape(U))

    cost_args = generate_cost_args(new_agents_tmp)

    start = time.perf_counter()

    sol = ilqr_trajectory_planner.solve_IPG(
        U=U,
        agent_id=new_agent_id,
        agents=new_agents_tmp,
        N=N,
        Ts=Ts,
        cost_args=cost_args,
        constraints_threshold=constraints_threshold,
        boundary=boundary_info,
        circle_obs=circle_obstacle)

    end = time.perf_counter()
    feas = check_solution_collision(N, sol[0], new_agents_tmp, obs=circle_obstacle)

    # check feasibility
    if agents.agent[i].feasibility and feas == False:
        start = time.perf_counter()

        sol = ilqr_trajectory_planner.solve_IPG(
            U=np.zeros(np.shape(U)),
            agent_id=new_agent_id,
            agents=new_agents_tmp,
            N=N,
            Ts=Ts,
            cost_args=cost_args,
            constraints_threshold=constraints_threshold,
            boundary=boundary_info,
            circle_obs=circle_obstacle)

        end = time.perf_counter()
        feas = check_solution_collision(N, sol[0], new_agents_tmp, obs=circle_obstacle)

    agents.agent[i].feasibility = feas

    sol = rebuild_sense_sol(N, sol, sense_list=sense_list, agents=agents)
    agents.agent[i].update(sol, i)

    # store solution for regenerating same closed-loop simulation
    store_open_loop_solution_npz(fig_folder, i, sol, time_step)

    # calc cost
    cost = sol[8]

    # calc ilqr solver executed time
    solver_cost_time = end - start

    return cost, solver_cost_time


def single_agent_plan_IPG_adaptU(agents, agent_id, circle_obstacle, horizon, Ts,
                                  constraints_threshold, boundary_info, fig_folder, time_step):
    N = horizon
    i = agent_id

    agents_tmp = copy.deepcopy(agents)
    # check how many agents can be seen
    sense_list, new_agent_id = sense_agent(agent_id=i, agents=agents_tmp, circle_obs=circle_obstacle)

    # build new multi-agents based on the sensor
    agents_tmp.agent[i].rebuild_sense_control_opt_series(sense_list, N)
    agents_tmp.agent_num = len(sense_list)

    new_agents_tmp = copy.deepcopy(agents_tmp)
    new_agents_tmp.agent = []
    for sense_id in sense_list:
        new_agents_tmp.agent.append(agents_tmp.agent[sense_id])

    # initialize control input : U
    U = new_agents_tmp.agent[new_agent_id].control_opt_series

    feasibility = new_agents_tmp.agent[new_agent_id].feasibility
    if time_step > 0 and feasibility:  # true
        for j in range(len(sense_list)):
            if j == new_agent_id:
                continue
            sense_id = sense_list[j]
            j_acc_predict = new_agents_tmp.agent[new_agent_id].sol_series[-1][1][0, sense_id * agents.input_dim]
            j_acc = new_agents_tmp.agent[j].control_input[0]

            if j_acc_predict * j_acc <= 0:
                U = np.zeros(np.shape(U))

                feasibility = False
                print('j_acc_predict : ', j_acc_predict, '   j_acc : ', j_acc, ' -------- predict_is_wrong ')
                break

    elif not feasibility:  # false
        U = np.zeros(np.shape(U))

    cost_args = generate_cost_args(new_agents_tmp)

    start = time.perf_counter()

    sol = ilqr_trajectory_planner.solve_IPG(
        U=U,
        agent_id=new_agent_id,
        agents=new_agents_tmp,
        N=N,
        Ts=Ts,
        cost_args=cost_args,
        constraints_threshold=constraints_threshold,
        boundary=boundary_info,
        circle_obs=circle_obstacle)

    end = time.perf_counter()
    feas = check_solution_collision(N, sol[0], new_agents_tmp, obs=circle_obstacle)

    # check feasibility
    if agents.agent[i].feasibility and feas == False:
        start = time.perf_counter()

        sol = ilqr_trajectory_planner.solve_IPG(
            U=np.zeros(np.shape(U)),
            agent_id=new_agent_id,
            agents=new_agents_tmp,
            N=N,
            Ts=Ts,
            cost_args=cost_args,
            constraints_threshold=constraints_threshold,
            boundary=boundary_info,
            circle_obs=circle_obstacle)

        end = time.perf_counter()
        feas = check_solution_collision(N, sol[0], new_agents_tmp, obs=circle_obstacle)

    agents.agent[i].feasibility = feas

    sol = rebuild_sense_sol(N, sol, sense_list=sense_list, agents=agents)

    agents.agent[i].update(sol, i)

    # store solution for regenerating same closed-loop simulation
    store_open_loop_solution_npz(fig_folder, i, sol,
                                 time_step)

    # calc cost
    cost = sol[8]

    # calc ilqr solver executed time
    solver_cost_time = end - start

    return cost, solver_cost_time


def single_agent_plan_IPG_ReactSafety(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold,
                                      boundary_info, fig_folder, time_step):
    N = horizon
    i = agent_id

    agents_tmp = copy.deepcopy(agents)
    # check how many agents can be seen
    sense_list, new_agent_id = sense_agent(agent_id=i, agents=agents_tmp, circle_obs=circle_obstacle)

    # build new multi-agents based on the sensor
    agents_tmp.agent[i].rebuild_sense_control_opt_series(sense_list, N)
    agents_tmp.agent_num = len(sense_list)

    new_agents_tmp = copy.deepcopy(agents_tmp)
    new_agents_tmp.agent = []
    for sense_id in sense_list:
        new_agents_tmp.agent.append(agents_tmp.agent[sense_id])

    # initialize control input : U
    U = new_agents_tmp.agent[new_agent_id].control_opt_series
    if not agents.agent[i].feasibility:
        U = np.zeros(np.shape(U))

    cost_args = generate_cost_args(new_agents_tmp)

    # check stuck
    agents.agent[i].check_getting_stuck(time_step, sense_list, new_agents_tmp)
    if agents.agent[i].safety_dis_is_updated:
        new_agents_tmp.agent[new_agent_id].safety_dis = agents.agent[i].safety_dis
        U = np.zeros(np.shape(U))

    start = time.perf_counter()

    sol = ilqr_trajectory_planner.solve_IPG(
        U=U,
        agent_id=new_agent_id,
        agents=new_agents_tmp,
        N=N,
        Ts=Ts,
        cost_args=cost_args,
        constraints_threshold=constraints_threshold,
        boundary=boundary_info,
        circle_obs=circle_obstacle)

    end = time.perf_counter()
    feas = check_solution_collision(N, sol[0], new_agents_tmp, obs=circle_obstacle)

    # check feasibility
    if agents.agent[i].feasibility and feas == False:
        start = time.perf_counter()
        sol = ilqr_trajectory_planner.solve_IPG(
            U=np.zeros(np.shape(U)),
            agent_id=new_agent_id,
            agents=new_agents_tmp,
            N=N,
            Ts=Ts,
            cost_args=cost_args,
            constraints_threshold=constraints_threshold,
            boundary=boundary_info,
            circle_obs=circle_obstacle)

        end = time.perf_counter()
        feas = check_solution_collision(N, sol[0], new_agents_tmp, obs=circle_obstacle)

    agents.agent[i].feasibility = feas

    sol = rebuild_sense_sol(N, sol, sense_list=sense_list, agents=agents)

    agents.agent[i].update(sol, i)

    # store solution for regenerating same closed-loop simulation
    store_open_loop_solution_npz(fig_folder, i, sol,
                                 time_step)

    # calc cost
    cost = sol[8]

    # calc ilqr solver executed time
    solver_cost_time = end - start

    return cost, solver_cost_time


def single_agent_plan_Vanilla(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold, boundary_info,
                                  fig_folder, time_step):
    N = horizon
    i = agent_id

    agents_tmp = copy.deepcopy(agents)
    # check how many agents can be seen
    sense_list, new_agent_id = sense_agent(agent_id=i, agents=agents_tmp, circle_obs=circle_obstacle)

    # build new multi-agents based on the sensor
    agents_tmp.agent_num = len(sense_list)

    new_agents_tmp = copy.deepcopy(agents_tmp)
    new_agents_tmp.agent = []
    for sense_id in sense_list:
        new_agents_tmp.agent.append(agents_tmp.agent[sense_id])

    U = new_agents_tmp.agent[new_agent_id].control_opt_series
    if not agents.agent[i].feasibility:
        U = np.zeros(np.shape(U))

    cost_args = generate_cost_args(new_agents_tmp)

    start = time.perf_counter()

    sol = ilqr_trajectory_planner.solve_vanilla(
        U=U,
        agent_id=new_agent_id,
        agent=new_agents_tmp,
        N=N,
        Ts=Ts,
        cost_args=cost_args,
        constraints_threshold=constraints_threshold,
        boundary=boundary_info,
        circle_obs=circle_obstacle)

    end = time.perf_counter()
    feas = check_nogames_solution_collision(N, sol[0], new_agent_id, new_agents_tmp, obs=circle_obstacle)

    # check feasibility
    if agents.agent[i].feasibility and feas == False:
        start = time.perf_counter()

        sol = ilqr_trajectory_planner.solve_vanilla(
            U=np.zeros(np.shape(U)),
            agent_id=new_agent_id,
            agent=new_agents_tmp,
            N=N,
            Ts=Ts,
            cost_args=cost_args,
            constraints_threshold=constraints_threshold,
            boundary=boundary_info,
            circle_obs=circle_obstacle)

        end = time.perf_counter()
        feas = check_nogames_solution_collision(N, sol[0], new_agent_id, new_agents_tmp, obs=circle_obstacle)

    # update agent
    agents.agent[i].feasibility = feas
    sol = rebuild_sense_sol_for_no_cooperation(N, sol, agent_id=i, sense_list=sense_list, agents=agents)
    agents.agent[i].update(sol, i)

    # store solution for regenerating same closed-loop simulation
    store_open_loop_solution_npz(fig_folder, i, sol,
                                 time_step)

    # calc cost
    cost = sol[8]

    # calc ilqr solver executed time
    solver_cost_time = end - start

    return cost, solver_cost_time


def single_agent_plan_Ignore(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold, boundary_info,
                                 fig_folder, time_step):
    N = horizon
    i = agent_id

    agents_tmp = copy.deepcopy(agents)

    # print("Agent ignores others.")
    sense_list = [i]
    new_agent_id = 0

    # build new multi-agents based on the sensor
    agents_tmp.agent_num = len(sense_list)

    new_agents_tmp = copy.deepcopy(agents_tmp)
    new_agents_tmp.agent = []
    for sense_id in sense_list:
        new_agents_tmp.agent.append(agents_tmp.agent[sense_id])

    U = new_agents_tmp.agent[new_agent_id].control_opt_series
    if not agents.agent[i].feasibility:
        U = np.zeros(np.shape(U))

    cost_args = generate_cost_args(new_agents_tmp)

    start = time.perf_counter()

    sol = ilqr_trajectory_planner.solve_IPG(
        U=U,
        agent_id=new_agent_id,
        agents=new_agents_tmp,
        N=N,
        Ts=Ts,
        cost_args=cost_args,
        constraints_threshold=constraints_threshold,
        boundary=boundary_info,
        circle_obs=circle_obstacle)

    end = time.perf_counter()
    feas = check_nogames_solution_collision(N, sol[0], new_agent_id, new_agents_tmp, obs=circle_obstacle)

    # check feasibility
    if agents.agent[i].feasibility and feas == False:
        start = time.perf_counter()

        sol = ilqr_trajectory_planner.solve_IPG(
            U=np.zeros(np.shape(U)),
            agent_id=new_agent_id,
            agents=new_agents_tmp,
            N=N,
            Ts=Ts,
            cost_args=cost_args,
            constraints_threshold=constraints_threshold,
            boundary=boundary_info,
            circle_obs=circle_obstacle)

        end = time.perf_counter()
        feas = check_nogames_solution_collision(N, sol[0], new_agent_id, new_agents_tmp, obs=circle_obstacle)

    # update agent
    agents.agent[i].feasibility = feas
    sol = rebuild_sense_sol(N, sol, sense_list=sense_list, agents=agents)
    agents.agent[i].update(sol, i)

    # store solution for regenerating same closed-loop simulation
    store_open_loop_solution_npz(fig_folder, i, sol,
                                 time_step)

    # calc cost
    cost = sol[8]

    # calc ilqr solver executed time
    solver_cost_time = end - start

    return cost, solver_cost_time


def single_agent_plan_Brake(agents, agent_id, circle_obstacle, horizon, Ts, constraints_threshold, boundary_info,
                                fig_folder, time_step):
    N = horizon
    i = agent_id

    agents_tmp = copy.deepcopy(agents)
    # check how many agents can be seen
    sense_list, new_agent_id = sense_agent(agent_id=i, agents=agents_tmp, circle_obs=circle_obstacle)

    # build new multi-agents based on the sensor
    agents_tmp.agent_num = len(sense_list)

    new_agents_tmp = copy.deepcopy(agents_tmp)
    new_agents_tmp.agent = []
    for sense_id in sense_list:
        new_agents_tmp.agent.append(agents_tmp.agent[sense_id])

    U = new_agents_tmp.agent[new_agent_id].control_opt_series
    if not agents.agent[i].feasibility:
        U = np.zeros(np.shape(U))

    cost_args = generate_cost_args(new_agents_tmp)

    start = time.perf_counter()
    sol = ilqr_trajectory_planner.solve_vanilla(
        U=U,
        agent_id=new_agent_id,
        agent=new_agents_tmp,
        N=N,
        Ts=Ts,
        cost_args=cost_args,
        constraints_threshold=constraints_threshold,
        boundary=boundary_info,
        circle_obs=circle_obstacle)

    end = time.perf_counter()
    feas = check_nogames_solution_collision(N, sol[0], new_agent_id, new_agents_tmp, obs=circle_obstacle)
    # feas = np.where(sol[7] > SOL_FEAS_THRESHOLD, False, True)

    # check feasibility
    if agents.agent[i].feasibility and feas == False:
        start = time.perf_counter()

        sol = ilqr_trajectory_planner.solve_vanilla(
            U=np.zeros(np.shape(U)),
            agent_id=new_agent_id,
            agent=new_agents_tmp,
            N=N,
            Ts=Ts,
            cost_args=cost_args,
            constraints_threshold=constraints_threshold,
            boundary=boundary_info,
            circle_obs=circle_obstacle)

        end = time.perf_counter()
        feas = check_nogames_solution_collision(N, sol[0], new_agent_id, new_agents_tmp, obs=circle_obstacle)

    # update agent
    agents.agent[i].feasibility = feas

    if not feas and agents.agent[i].position[3] != 0:
        sol = generate_braking_solution_nonCoop(N, sol, agent_id=i, sense_list=sense_list, agents=agents, boundary=boundary_info, circle_obs=circle_obstacle)
    elif not feas and agents.agent[i].position[3] == 0:
        sol = rebuild_sense_sol_for_no_cooperation_when_braking(N, sol, agent_id=i, sense_list=sense_list,
                                                                agents=agents)
    else:
        sol = rebuild_sense_sol_for_no_cooperation(N, sol, agent_id=i, sense_list=sense_list, agents=agents)

    agents.agent[i].update(sol, i)

    # store solution for regenerating same closed-loop simulation
    store_open_loop_solution_npz(fig_folder, i, sol,
                                 time_step)

    # calc cost
    cost = sol[8]

    # calc ilqr solver executed time
    solver_cost_time = end - start

    return cost, solver_cost_time

#############################################################################


def initialize_params(agents, sim_time, boundary):
    # parameter
    Ts = float(agents.agent[0].Ts)  # sec
    N = int(sim_time / Ts)

    constraints_threshold = 1.0e-3

    boundary_info = boundary
    if boundary is None:
        boundary = np.array([-1000000, -1000000, 1000000, 1000000])  # xmin, ymin, xmax, ymax

    plot_boundary = [1000000, 1000000, -1000000, -1000000]
    plot_boundary = update_boundary_with_agents(plot_boundary, agents)  # xmin, ymin, xmax, ymax

    return Ts, N, constraints_threshold, boundary_info, boundary, plot_boundary
