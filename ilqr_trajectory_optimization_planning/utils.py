# utils.py
"""
Tool Functions

Created on 2023/11/9
@author: Pin-Yun Hung
"""

from datetime import datetime
import pathlib
from typing import AnyStr
from scene_objects import *
import random
import jax
import jax.numpy as jnp

################################
# Functions for Saving Data
################################

def create_folder(agent_num: int, sim_name: AnyStr) -> (AnyStr, AnyStr):
    """
    Create a folder for the new simulation

    @param agent_num : number of agents
    @param sim_name : solver type
    """
    now = datetime.now()  # current date and time
    folder = f"{agent_num}_agent_" + sim_name + "_" + now.strftime("%m%d%H%M%S") + "/"
    folder_path = "figs/" + folder

    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    return folder_path, folder


def save_info(folder_name, agents, boundary, horizon, solver_type, constraints_threshold, circle_obs=None,
              boundary_range=None, agents_use_games=None, adaptive_backup_weight=False):
    """
    Save problem information as 'txt' & 'npz'

    PreConditions : Have a folder to save the information files
    PostConditions :
        Save the information of scene & agents as 'txt' & 'npz' file.
        - 'txt' is for checking the information conveniently
        - 'npz' is for loading the information and plot

    @param folder_name
    @param      agents: all agents in the scene
    @param    boundary: boundary of the scene
    @param     horizon: iLQR solver horizon
    @param solver_type: the iLQR solver type which use to solve the problem
    @param constraints_threshold
    @param circle_obs: circle obstacle
    """

    # save as txt
    save_info_file = folder_name + "/info.txt"

    f = open(save_info_file, "a")

    f.write("------------ Agents information ------------\n")
    f.write("agents number = " + str(agents.agent_num) + "\n")
    f.write("state dimension = " + str(agents.state_dim) + "\n")
    f.write("input dimension = " + str(agents.input_dim) + "\n")
    f.write("Sampling Time = " + str(agents.Ts) + "\n")
    for i in range(agents.agent_num):
        agent = agents.agent[i]
        f.write("\n ---- agent " + str(i + 1) + " : \n")
        f.write("size (radius) = " + str(agent.radius) + "\n")
        f.write("initial_state = " + str(agent.initial_state) + "\n")
        f.write("target_state = " + str(agent.target_state) + "\n")
        f.write("vmin & vmax = " + str(agent.vmin) + ", " + str(agent.vmax) + "\n")
        f.write("umin & umax = " + str(agent.umin) + ", " + str(agent.umax) + "\n")
        f.write("safety_dis = " + str(agent.safety_dis) + "\n")
        f.write("Q_cost = " + str(agent.Q_cost) + "\n")
        f.write("R_cost = " + str(agent.R_cost) + "\n")
        f.write("D_cost = " + str(agent.D_cost) + "\n")
        f.write("B_cost = " + str(agent.B_cost) + "\n")

    f.write("\n------------ boundary information ------------\n")
    if boundary is None:
        f.write("boundary = None\n")
    else:
        f.write("boundary = " + np.array2string(boundary, precision=2, separator=',', suppress_small=True) + "\n")
    if boundary_range is None:
        f.write("boundary_range = None\n")
    else:
        f.write("boundary_range = " + np.array2string(boundary_range, precision=2, separator=',',
                                                       suppress_small=True) + "\n")

    if circle_obs is not None:
        f.write("\n------------ circle obstacle information ------------\n")
        f.write("circle obstacle number = " + str(circle_obs.obstacle_num) + "\n")
        for i in range(circle_obs.obstacle_num):
            f.write("   circle obstacle " + str(i + 1) + " : " + str(circle_obs.pos[i]) + ", radius = " + str(
                circle_obs.radius[i]) + "\n")

    f.write("\n------------ solver information ------------\n")
    f.write("horizon = " + str(horizon) + "\n")
    f.write("ilqr solver type = " + str(solver_type) + "\n")
    f.write("constraints threshold = " + str(constraints_threshold) + "\n")

    if adaptive_backup_weight:
        f.write("adaptive backup weight = True \n")

    if agents_use_games is not None:
        f.write("\nHeterogeneous Solvers = \n")
        for key in agents_use_games.keys():
            f.write(str(key) + ": " + agents_use_games[key] + "\n")

    f.close()

    # save npz
    save_info_file = folder_name + "/info.npz"

    agent = agents.agent[0]
    agent_radius = agent.radius
    initial_state = agent.initial_state
    target_state = agent.target_state
    vmin = agent.vmin
    vmax = agent.vmax
    umin = agent.umin
    umax = agent.umax
    safety_dis = agent.safety_dis
    Q_cost = agent.Q_cost
    R_cost = agent.R_cost
    D_cost = agent.D_cost
    B_cost = agent.B_cost
    for i in range(1, agents.agent_num):
        agent = agents.agent[i]
        agent_radius = np.vstack((agent_radius, agent.radius))
        initial_state = np.vstack((initial_state, agent.initial_state))
        target_state = np.vstack((target_state, agent.target_state))
        vmin = np.vstack((vmin, agent.vmin))
        vmax = np.vstack((vmax, agent.vmax))
        umin = np.vstack((umin, agent.umin))
        umax = np.vstack((umax, agent.umax))
        safety_dis = np.vstack((safety_dis, agent.safety_dis))
        Q_cost = np.vstack((Q_cost, agent.Q_cost))
        R_cost = np.vstack((R_cost, agent.R_cost))
        D_cost = np.vstack((D_cost, agent.D_cost))
        B_cost = np.vstack((B_cost, agent.B_cost))

    circle_obs_num = 0
    circle_obs_pos = []
    circle_obs_radius = []
    if circle_obs is not None:
        circle_obs_num = circle_obs.obstacle_num,
        circle_obs_pos = circle_obs.pos,
        circle_obs_radius = circle_obs.radius

    agents_with_games = []
    if agents_use_games is not None:
        for key in agents_use_games.keys():
            key_solver = [key, agents_use_games[key]]
            agents_with_games.append(key_solver)

    np.savez(save_info_file,
             agent_num=agents.agent_num,
             state_dim=agents.state_dim,
             input_dim=agents.input_dim,
             Ts=agents.Ts,
             agent_radius=agent_radius,
             initial_state=initial_state,
             target_state=target_state,
             vmin=vmin,
             vmax=vmax,
             umin=umin,
             umax=umax,
             safety_dis=safety_dis,
             Q_cost=Q_cost,
             R_cost=R_cost,
             D_cost=D_cost,
             B_cost=B_cost,
             boundary=boundary,
             boundary_range=boundary_range,
             circle_obs_num=circle_obs_num,
             circle_obs_pos=circle_obs_pos,
             circle_obs_radius=circle_obs_radius,
             horizon=horizon,
             solver_type=solver_type,
             constraints_threshold=constraints_threshold,
             agents_with_games=agents_with_games,
             adaptive_backup_weight=adaptive_backup_weight)


def save_reactive_safety_distance(folder_name, time_step_and_safety_dis):
    # save npz
    save_info_file = folder_name + "safety_dis.npz"

    np.savez(save_info_file, time_step_and_safety_dis=time_step_and_safety_dis)


def store_open_loop_solution(fig_folder, agent_id, sol, time_step):
    """
    store the open-loop solutions of all agents.
    """

    # write to file
    file_name = fig_folder + "agent" + str(agent_id + 1) + "_traj" + str(time_step) + ".txt"
    f = open(file_name, "w")

    for t_i in range(len(sol[0])):
        for t_j in range(len(sol[0][0])):
            content = str(sol[0][t_i, t_j]) + " "
            f.write(content)

        for t_j in range(len(sol[1][0])):
            content = str(sol[1][t_i, t_j]) + " "
            f.write(content)
        f.write("\n")
    f.close()


def store_open_loop_solution_npz(fig_folder, agent_id, sol, time_step):
    """
    store the open-loop solutions of all agents.
    """
    # write to file
    file_name = fig_folder + "agent" + str(agent_id + 1) + "_traj" + str(time_step) + ".npz"

    X = sol[0]
    U = sol[1]

    np.savez(file_name, X=X, U=U)


def store_closed_loop_trajectory(folder_path, agents, sim_num):
    file_name = folder_path + "/closedLoop_trajectory.txt"
    f = open(file_name, "w")
    for k in range(sim_num):
        for id in range(agents.agent_num):
            traj = np.asarray(agents.agent[id].trajectory)
            input = np.asarray(agents.agent[id].control_series)
            for j in range(agents.state_dim):
                content = str(traj[k, j]) + " "
                f.write(content)

            for j in range(agents.input_dim):
                content = str(input[k, j]) + " "
                f.write(content)
        f.write("\n")
    f.close()

def store_solved_time(folder_name, solved_time):
    # save npz
    save_info_file = folder_name + "solver_cost_time.npz"

    np.savez(save_info_file, solver_cost_time=solved_time)


################################
# Functions for Feasibility Checking
################################

def no_rect_circle_collision(circle_center, circle_radius, rectangle_center, rectangle_width, rectangle_height, constraints_threshold=0.15):
    # Calculate the distance between the circle center and the closest point on the rectangle
    closest_x = max(rectangle_center[0] - rectangle_width / 2,
                    min(circle_center[0], rectangle_center[0] + rectangle_width / 2))
    closest_y = max(rectangle_center[1] - rectangle_height / 2,
                    min(circle_center[1], rectangle_center[1] + rectangle_height / 2))

    distance = math.sqrt((circle_center[0] - closest_x) ** 2 + (circle_center[1] - closest_y) ** 2)
    # Check if the distance is less than or equal to the radius of the circle
    return distance > (circle_radius - constraints_threshold)

def check_solution_collision(N, X, agents, constraints_threshold=None, obs=None, rect_obs=None, boundary=None):
    """
    Check whether any collision exist in the trajectory


    This function is used to check whether the solution of the iLQR solver
    exist obstacle collision or agent collision. However, it also can check
    the real trajectory. The important thing is to input a right "X".
    For example, if N = 10, agents = [a1, a2, a2], then X must be a 11 x 12
    array.
    (11 is N + 1, and 12 is state dimension * agents number = 4 * 3 = 12).
    Therefore, X will be :

    [[x1_k0, y1_k0, th1_k0, v1_k0, x2_k0, y2_k0, th2_k0, v2_k0, x3_k0, y3_k0, th3_k0, v3_k0]
     [x1_k1, y1_k1, th1_k1, v1_k1, x2_k1, y2_k1, th2_k1, v2_k1, x3_k1, y3_k1, th3_k1, v3_k1]
     ...
     [x1_k10, y1_k10, th1_k10, v1_k10, x2_k10, y2_k10, th2_k10, v2_k10, x3_k10, y3_k10, th3_k10, v3_k10]
    ]

    ------------------------------------------------------------------------
    PreConditions : Have trajectories (can be real or prediction) of agents
    PostConditions :
        If collision happens, print the time step and the type of collision
        (agent or obstacle) and return false. If no collision, return true.

    @param      N : horizon
    @param      X : states of agents at the whole horizon, a 2D array
    @param agents : all agents in the solution
    @param constraints_threshold
    @param    obs : circle obstacle

    @return (bool) no_collision
    """

    no_collision = True

    for k in range(N + 1):
        for i in range(agents.agent_num):
            n1 = i * agents.state_dim
            x_i = X[k, n1:n1+2]
            r_i = agents.agent[i].radius

            infeasible_threshold = agents.agent[i].radius / 3
            # check agent collision
            for j in range(i + 1, agents.agent_num):
                n2 = j * agents.state_dim
                x_j = X[k, n2:n2 + 2]

                dis = (r_i + agents.agent[j].radius) - math.dist(x_i, x_j)

                if dis > infeasible_threshold:
                    # print("t = ", k , "agent collision : dis = ", round(dis, 2))
                    no_collision = False
                    return no_collision

            # check obstacle collision
            if obs is not None:
                for j in range(obs.obstacle_num):
                    dis = (obs.radius[j] + agents.agent[i].radius) - math.dist(x_i, obs.pos[j])

                    if dis > infeasible_threshold:
                        # print("t = ", k , "obstacle collision : dis = ", round(dis, 2))
                        no_collision = False
                        return no_collision

            if rect_obs is not None:
                for j in range(rect_obs.obstacle_num):
                    x_min = rect_obs.pos[j][0][0]
                    x_max = rect_obs.pos[j][2][0]
                    y_min = rect_obs.pos[j][0][1]
                    y_max = rect_obs.pos[j][2][1]

                    rect_center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                    rect_w = x_max - x_min
                    rect_h = y_max - y_min

                    no_collision = no_rect_circle_collision(circle_center=x_i, circle_radius=r_i,
                                                rectangle_center=rect_center, rectangle_width=rect_w, rectangle_height=rect_h)
                    if not no_collision:
                        return no_collision

            if boundary is not None:
                px = x_i[0]
                py = x_i[1]
                boundary_constraint1 = boundary[0] - px - r_i
                boundary_constraint2 = boundary[1] - py - r_i
                boundary_constraint3 = px + r_i - boundary[2]
                boundary_constraint4 = py + r_i - boundary[3]

                no_collision = np.all(
                    np.array([boundary_constraint1, boundary_constraint2, boundary_constraint3, boundary_constraint4]) < 0)
                if not no_collision:
                    return no_collision

    return no_collision


def check_nogames_solution_collision(N, X, agent_id, agents, constraints_threshold=None, obs=None):
    """
    Check whether any collision exist in the trajectory
    """

    no_collision = True

    for k in range(N + 1):
        x_i = X[k, :]
        infeasible_threshold = agents.agent[agent_id].radius / 3
        # check agent collision
        for j in range(agents.agent_num):
            if j == agent_id:
                continue

            x_j = agents.agent[j].pred_non_cooperated_traj[k, :]

            dis = (agents.agent[agent_id].radius + agents.agent[j].radius) - \
                  np.sqrt((x_i[0] - x_j[0]) ** 2.0 + (x_i[1] - x_j[1]) ** 2.0)

            if dis > infeasible_threshold:
                # print("t = ", k, "agent collision : dis = ", dis)
                no_collision = False
                return no_collision

        # check obstacle collision
        if obs is not None:
            for j in range(obs.obstacle_num):
                dis = (obs.radius[j] + agents.agent[agent_id].radius) - np.sqrt(
                    (x_i[0] - obs.pos[j][0]) ** 2.0 + (x_i[1] - obs.pos[j][1]) ** 2.0)

                if dis > infeasible_threshold:
                    # print("t = ", k, "obstacle collision : dis = ", dis)
                    no_collision = False
                    return no_collision

    return no_collision


def check_collision(agents, constraints_threshold, obs=None, rect_obs=None, boundary=None):
    """
    Check whether the collision happens at the current position.
    If the collision happens, return False, else return Ture.
    """

    for i in range(agents.agent_num):
        x_i = agents.agent[i].position[:2]
        r_i = agents.agent[i].radius - constraints_threshold

        # check agent collision
        for j in range(i + 1, agents.agent_num):
            x_j = agents.agent[j].position[:2]

            dis = (agents.agent[i].radius + agents.agent[j].radius) - math.dist(x_i, x_j)

            if dis > constraints_threshold:
                print("agent collision : dis = ", dis)
                return False

        # check obstacle collision
        if obs is not None:
            for j in range(obs.obstacle_num):
                dis = (obs.radius[j] + agents.agent[i].radius) - math.dist(x_i, obs.pos[j])

                if dis > constraints_threshold:
                    print("obstacle collision : dis = ", dis)
                    return False

        if rect_obs is not None:
            for j in range(rect_obs.obstacle_num):
                x_min = rect_obs.pos[j][0][0]
                x_max = rect_obs.pos[j][2][0]
                y_min = rect_obs.pos[j][0][1]
                y_max = rect_obs.pos[j][2][1]

                rect_center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                rect_w = x_max - x_min
                rect_h = y_max - y_min

                no_collision = no_rect_circle_collision(circle_center=x_i, circle_radius=r_i,
                                                        rectangle_center=rect_center, rectangle_width=rect_w,
                                                        rectangle_height=rect_h)
                if not no_collision:
                    return no_collision

        if boundary is not None:
            px = x_i[0]
            py = x_i[1]
            boundary_constraint1 = boundary[0] - px - r_i
            boundary_constraint2 = boundary[1] - py - r_i
            boundary_constraint3 = px + r_i - boundary[2]
            boundary_constraint4 = py + r_i - boundary[3]

            no_collision = np.all(
                np.array([boundary_constraint1, boundary_constraint2, boundary_constraint3, boundary_constraint4]) < 0)
            if not no_collision:
                return no_collision

    return True

def check_solution_stay_in_boundary(agents, N, boundary=None):
    for i in range(agents.agent_num):
        x_opt = agents.agent[i].sol_series[-1][0]
        r = agents.agent[i].radius
        for k in range(N + 1):
            x = x_opt[k, 0]
            y = x_opt[k, 1]
            # if (x - r) < boundary[0] or (y - r) < boundary[1] or (x + r) > boundary[2] or (y + r) > boundary[3]:
            #     print("solution out of boundary!!")
            #     return False
            if x < boundary[0] or y < boundary[1] or x > boundary[2] or y > boundary[3]:
                print("solution out of boundary!!")
                return False

    return True


def check_terminal_state(X, agents):

    terminal_constraint_work = True
    for i in range(agents.agent_num):
        n1 = i * agents.state_dim
        x_i = X[-1, n1:n1 + 2]
        xf = agents.agent[i].target_state[:2]
        dis = math.dist(x_i, xf)

        if dis > agents.agent[i].radius * 2:
            terminal_constraint_work = False
            return terminal_constraint_work

    return terminal_constraint_work

################################
# Functions for Sensing
################################

def sense_agent(agent_id, agents, circle_obs=None):
    sense_list = []
    new_agent_id = agent_id
    sensor = Lidar()

    sense_too_far_list = []
    sense_at_behind_list = []
    sense_blind_by_obstacle_list = []
    sense_blind_by_agent_list = []

    # current agent direction
    agent_front_x = agents.agent[agent_id].radius * np.cos(agents.agent[agent_id].position[2])
    agent_front_y = agents.agent[agent_id].radius * np.sin(agents.agent[agent_id].position[2])
    agent_dir = np.array([agent_front_x, agent_front_y])

    def check_is_blinding_by_obstacle(agents, agent_id, other_agent_id, circle_obs):
        is_blind = False
        tangent_point1, tangent_point2 = find_tangency_points(
            point=agents.agent[agent_id].position[:2],
            circle_center=agents.agent[other_agent_id].position[:2],
            circle_radius=agents.agent[other_agent_id].radius)

        for j in range(circle_obs.obstacle_num):
            # tangent_point1
            is_blind1 = line_crosses_circle(
                line_start=agents.agent[agent_id].position[:2],
                line_end=tangent_point1,
                circle_center=circle_obs.pos[j],
                circle_radius=circle_obs.radius[j])

            # tangent_point2
            is_blind2 = line_crosses_circle(
                line_start=agents.agent[agent_id].position[:2],
                line_end=tangent_point2,
                circle_center=circle_obs.pos[j],
                circle_radius=circle_obs.radius[j])
            if np.all(np.array([is_blind1, is_blind2])):
                is_blind = True
                return is_blind

        return is_blind

    def check_is_blinding_by_agents(agents, agent_id, other_agent_id):
        tangent_point1, tangent_point2 = find_tangency_points(
            point=agents.agent[agent_id].position[:2],
            circle_center=agents.agent[other_agent_id].position[:2],
            circle_radius=agents.agent[other_agent_id].radius)

        for j_id in range(agents.agent_num):

            if j_id == agent_id or j_id == other_agent_id:
                continue

            circle_center = agents.agent[j_id].position[:2]
            circle_radius = agents.agent[j_id].radius

            # tangent_point1
            is_blind1 = line_crosses_circle(
                line_start=agents.agent[agent_id].position[:2],
                line_end=tangent_point1,
                circle_center=circle_center,
                circle_radius=circle_radius)

            # tangent_point2
            is_blind2 = line_crosses_circle(
                line_start=agents.agent[agent_id].position[:2],
                line_end=tangent_point2,
                circle_center=circle_center,
                circle_radius=circle_radius)

            if np.all(np.array([is_blind1, is_blind2])):
                return True

        return False

    # check agent is in the front
    for i in range(agents.agent_num):
        if i != agent_id:
            other_dir = np.array([agents.agent[i].position[0] - agents.agent[agent_id].position[0],
                                   agents.agent[i].position[1] - agents.agent[agent_id].position[1]])
            # dot product
            dot_product = agent_dir[0] * other_dir[0] + agent_dir[1] * other_dir[1]

            # if agent is at behind, check if it is in the lidar circle
            dis = np.sqrt(other_dir[0] ** 2 + other_dir[1] ** 2)
            if dot_product < 0:
                if dis < sensor.radius:
                    is_blind = check_is_blinding_by_agents(agents, agent_id, i)

                    if is_blind:
                        sense_blind_by_agent_list.append(i)
                    else:
                        if circle_obs is not None:
                            is_blind = check_is_blinding_by_obstacle(agents, agent_id, i, circle_obs)

                        if not is_blind:
                            sense_list.append(i)
                        else:
                            sense_blind_by_obstacle_list.append(i)
                else:
                    sense_at_behind_list.append(i)
            elif dot_product >= 0:  # if is in the front
                # check if obstacle blind the agent and the agent is close enough to be seen
                if dis < 100.0:

                    is_blind = check_is_blinding_by_agents(agents, agent_id, i)

                    if is_blind:
                        sense_blind_by_agent_list.append(i)
                    else:

                        if circle_obs is not None:
                            is_blind = check_is_blinding_by_obstacle(agents, agent_id, i, circle_obs)

                        if not is_blind:
                            sense_list.append(i)
                        else:
                            sense_blind_by_obstacle_list.append(i)
                else:
                    sense_too_far_list.append(i)
        else:
            sense_list.insert(0, i)
            new_agent_id = 0
            # sense_list.append(i)
            # new_agent_id = len(sense_list) - 1

    # print("id = ", agent_id, "The agent is at behind : ", sense_at_behind_list)
    # print("id = ", agent_id, "The agent is too far to see : ", sense_too_far_list)
    # print("id = ", agent_id, "The obstacles blind the agent : ", sense_blind_by_obstacle_list)
    # print("id = ", agent_id, "Other agents blind the agent : ", sense_blind_by_agent_list)
    # print("id = ", agent_id, "Sense List : ", sense_list, "     ")
    return sense_list, new_agent_id

def line_crosses_circle(line_start, line_end, circle_center, circle_radius):
    # Calculate the direction vector of the line
    line_direction = (line_end[0] - line_start[0], line_end[1] - line_start[1])

    # Calculate the vector from the line start to the circle center
    start_to_center = (circle_center[0] - line_start[0], circle_center[1] - line_start[1])

    # Calculate the dot product of the line direction and the vector to the circle center
    dot_product = line_direction[0] * start_to_center[0] + line_direction[1] * start_to_center[1]

    # Check if the line is perpendicular to the vector from the line start to the circle center
    if dot_product < 0:
        return False

    # Calculate the squared length of the line direction vector
    line_length_squared = line_direction[0] ** 2 + line_direction[1] ** 2

    # Calculate the squared distance between the line start and the circle center
    start_to_center_squared = start_to_center[0] ** 2 + start_to_center[1] ** 2

    # Calculate the squared distance between the line and the circle center
    distance_squared = start_to_center_squared - (dot_product ** 2 / line_length_squared)

    # Check if the squared distance is less than or equal to the squared circle radius
    if distance_squared <= circle_radius ** 2 and line_length_squared >= start_to_center_squared:
        return True

    return False


def line_segment_intersects_circle(segment_start, segment_end, circle_center, circle_radius):
    # Calculate the vector from the line segment start to the end
    segment_vector = (segment_end[0] - segment_start[0], segment_end[1] - segment_start[1])

    # Calculate the vector from the line segment start to the circle center
    start_to_center = (circle_center[0] - segment_start[0], circle_center[1] - segment_start[1])

    # Calculate the squared length of the segment vector
    segment_length_squared = segment_vector[0] ** 2 + segment_vector[1] ** 2

    # Calculate the dot product of the start-to-center vector and the segment vector
    dot_product = start_to_center[0] * segment_vector[0] + start_to_center[1] * segment_vector[1]

    # Calculate the parameters for the quadratic equation
    a = segment_length_squared
    b = 2 * dot_product
    c = start_to_center[0] ** 2 + start_to_center[1] ** 2 - circle_radius ** 2

    # Calculate the discriminant
    discriminant = b ** 2 - 4 * a * c

    # Check if the line segment intersects with the circle
    if discriminant >= 0:
        # Calculate the two possible solutions for t
        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)

        # Check if any solution lies within the range of the line segment
        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True

    return False


def find_tangency_points(point, circle_center, circle_radius):
    # Calculate the vector from the circle center to the given point
    center_to_point = (point[0] - circle_center[0], point[1] - circle_center[1])

    # Calculate the distance between the circle center and the given point
    distance = math.sqrt(center_to_point[0] ** 2 + center_to_point[1] ** 2)

    # Check if the given point is inside the circle
    if distance < circle_radius:
        return None  # No tangency points

    # Calculate the angle between the center-to-point vector and the positive x-axis
    angle = math.atan2(center_to_point[1], center_to_point[0])

    # Calculate the angle between the tangent lines and the positive x-axis
    tangent_angle = math.acos(circle_radius / distance)

    # Calculate the angles of the tangent points
    angle1 = angle + tangent_angle
    angle2 = angle - tangent_angle

    # Calculate the coordinates of the tangent points
    tangent_point1 = [
        circle_center[0] + circle_radius * math.cos(angle1),
        circle_center[1] + circle_radius * math.sin(angle1)
    ]
    tangent_point2 = [
        circle_center[0] + circle_radius * math.cos(angle2),
        circle_center[1] + circle_radius * math.sin(angle2)
    ]

    return tangent_point1, tangent_point2


def rebuild_sense_sol(N, sol, sense_list, agents):
    x_opt = np.ones((N + 1, agents.state_dim * agents.agent_num))
    u_opt = np.zeros((N, agents.input_dim * agents.agent_num))

    for i in range(agents.agent_num):
        n1 = i * agents.state_dim
        n2 = i * agents.input_dim

        if i in sense_list:
            index = sense_list.index(i)
            n1_sense = index * agents.state_dim
            n2_sense = index * agents.input_dim
            x_opt[:, n1:n1 + agents.state_dim] = sol[0][:, n1_sense:n1_sense + agents.state_dim]
            u_opt[:, n2:n2 + agents.input_dim] = sol[1][:, n2_sense:n2_sense + agents.input_dim]
        else:
            for j in range(agents.state_dim):
                x_opt[:, n1 + j] = x_opt[:, n1 + j] * agents.agent[i].position[j]

    new_sol = tuple()

    for i in range(len(sol)):
        if i == 0:
            new_sol = new_sol + (jax.device_put(x_opt),)
        elif i == 1:
            new_sol = new_sol + (jax.device_put(u_opt),)
        else:
            new_sol = new_sol + (sol[i],)

    return new_sol

def rebuild_sense_sol_for_no_cooperation(N, sol, agent_id, sense_list, agents):
    x_opt = np.ones((N + 1, agents.state_dim * agents.agent_num))
    u_opt = np.zeros((N, agents.input_dim * agents.agent_num))

    for i in range(agents.agent_num):
        n1 = i * agents.state_dim
        n2 = i * agents.input_dim

        if i == agent_id:
            x_opt[:, n1:n1 + agents.state_dim] = sol[0]
            u_opt[:, n2:n2 + agents.input_dim] = sol[1]
        elif i in sense_list:
            # index = sense_list.index(i)
            x_opt[:, n1:n1 + agents.state_dim] = agents.agent[i].pred_non_cooperated_traj
        else:
            x_opt[:, n1:n1 + agents.state_dim] = agents.agent[i].position

    new_sol = tuple()

    for i in range(len(sol)):
        if i == 0:
            new_sol = new_sol + (jax.device_put(x_opt),)
        elif i == 1:
            new_sol = new_sol + (jax.device_put(u_opt),)
        else:
            new_sol = new_sol + (sol[i],)

    return new_sol


def rebuild_sense_sol_for_no_cooperation_when_braking(N, sol, agent_id, sense_list, agents):
    x_opt = np.ones((N + 1, agents.state_dim * agents.agent_num))
    u_opt = np.zeros((N, agents.input_dim * agents.agent_num))

    for i in range(agents.agent_num):
        n1 = i * agents.state_dim
        n2 = i * agents.input_dim

        if i == agent_id:
            x_opt[:, n1:n1 + agents.state_dim] = agents.agent[i].position
            # x_opt[:, n1:n1 + agents.state_dim] = sol[0]
            # u_opt[:, n2:n2 + agents.input_dim] = sol[1]
        elif i in sense_list:
            index = sense_list.index(i)
            x_opt[:, n1:n1 + agents.state_dim] = agents.agent[index].pred_non_cooperated_traj
        else:
            x_opt[:, n1:n1 + agents.state_dim] = agents.agent[i].position

    new_sol = tuple()

    for i in range(len(sol)):
        if i == 0:
            new_sol = new_sol + (jax.device_put(x_opt),)
        elif i == 1:
            new_sol = new_sol + (jax.device_put(u_opt),)
        else:
            new_sol = new_sol + (sol[i],)

    return new_sol

def is_line_crossing_rectangle(line_start, line_end, rect_bottom_left, rect_top_right):
    # Extract coordinates for readability
    x1, y1 = line_start
    x2, y2 = line_end
    x_min, y_min = rect_bottom_left
    x_max, y_max = rect_top_right

    # Check if the line is entirely outside the rectangle
    if (x1 < x_min and x2 < x_min) or (x1 > x_max and x2 > x_max) or \
       (y1 < y_min and y2 < y_min) or (y1 > y_max and y2 > y_max):
        return False

    # Check if the line is entirely inside the rectangle
    if (x1 >= x_min and x1 <= x_max and y1 >= y_min and y1 <= y_max) and \
       (x2 >= x_min and x2 <= x_max and y2 >= y_min and y2 <= y_max):
        return True

    # Check if the line intersects with any of the rectangle's sides
    # Check for intersection with the left vertical side of the rectangle
    if x1 < x_min and x2 >= x_min:
        y_intersection = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
        if y_intersection >= y_min and y_intersection <= y_max:
            return True

    # Check for intersection with the right vertical side of the rectangle
    if x1 > x_max and x2 <= x_max:
        y_intersection = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
        if y_intersection >= y_min and y_intersection <= y_max:
            return True

    # Check for intersection with the top horizontal side of the rectangle
    if y1 < y_min and y2 >= y_min:
        x_intersection = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
        if x_intersection >= x_min and x_intersection <= x_max:
            return True

    # Check for intersection with the bottom horizontal side of the rectangle
    if y1 > y_max and y2 <= y_max:
        x_intersection = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
        if x_intersection >= x_min and x_intersection <= x_max:
            return True

    return False


def sense_agent_rect(agent_id, agents, circle_obs=None, rect_obs=None):
    sense_list = []
    new_agent_id = agent_id
    sensor = Lidar()

    sense_too_far_list = []
    sense_at_behind_list = []
    sense_blind_by_obstacle_list = []
    sense_blind_by_agent_list = []

    # current agent direction
    agent_front_x = agents.agent[agent_id].radius * np.cos(agents.agent[agent_id].position[2])
    agent_front_y = agents.agent[agent_id].radius * np.sin(agents.agent[agent_id].position[2])
    agent_dir = np.array([agent_front_x, agent_front_y])

    def check_is_blinding_by_obstacle(agents, agent_id, other_agent_id, circle_obs):
        is_blind = False
        tangent_point1, tangent_point2 = find_tangency_points(
            point=agents.agent[agent_id].position[:2],
            circle_center=agents.agent[other_agent_id].position[:2],
            circle_radius=agents.agent[other_agent_id].radius)

        for j in range(circle_obs.obstacle_num):
            # tangent_point1
            is_blind1 = line_crosses_circle(
                line_start=agents.agent[agent_id].position[:2],
                line_end=tangent_point1,
                circle_center=circle_obs.pos[j],
                circle_radius=circle_obs.radius[j])

            # tangent_point2
            is_blind2 = line_crosses_circle(
                line_start=agents.agent[agent_id].position[:2],
                line_end=tangent_point2,
                circle_center=circle_obs.pos[j],
                circle_radius=circle_obs.radius[j])
            if np.all(np.array([is_blind1, is_blind2])):
                is_blind = True
                return is_blind

        return is_blind

    def check_is_blinding_by_agents(agents, agent_id, other_agent_id):
        tangent_point1, tangent_point2 = find_tangency_points(
            point=agents.agent[agent_id].position[:2],
            circle_center=agents.agent[other_agent_id].position[:2],
            circle_radius=agents.agent[other_agent_id].radius)

        for j_id in range(agents.agent_num):

            if j_id == agent_id or j_id == other_agent_id:
                continue

            circle_center = agents.agent[j_id].position[:2]
            circle_radius = agents.agent[j_id].radius

            # tangent_point1
            is_blind1 = line_crosses_circle(
                line_start=agents.agent[agent_id].position[:2],
                line_end=tangent_point1,
                circle_center=circle_center,
                circle_radius=circle_radius)

            # tangent_point2
            is_blind2 = line_crosses_circle(
                line_start=agents.agent[agent_id].position[:2],
                line_end=tangent_point2,
                circle_center=circle_center,
                circle_radius=circle_radius)

            if np.all(np.array([is_blind1, is_blind2])):
                return True

        return False

    def check_is_blinding_by_rectangular_obstacle(agents, agent_id, other_agent_id, rect_obs):
        is_blind = False
        tangent_point1, tangent_point2 = find_tangency_points(
            point=agents.agent[agent_id].position[:2],
            circle_center=agents.agent[other_agent_id].position[:2],
            circle_radius=agents.agent[other_agent_id].radius)

        for j in range(rect_obs.obstacle_num):
            rect_bottom_left = rect_obs.pos[j][0]
            rect_top_right = rect_obs.pos[j][2]

            # tangent_point1
            is_blind1 = is_line_crossing_rectangle(
                line_start=agents.agent[agent_id].position[:2],
                line_end=tangent_point1,
                rect_bottom_left=rect_bottom_left,
                rect_top_right=rect_top_right)

            # tangent_point2
            is_blind2 = is_line_crossing_rectangle(
                line_start=agents.agent[agent_id].position[:2],
                line_end=tangent_point2,
                rect_bottom_left=rect_bottom_left,
                rect_top_right=rect_top_right)
            if np.all(np.array([is_blind1, is_blind2])):
                is_blind = True
                return is_blind

        return is_blind

    # check agent is in the front
    for i in range(agents.agent_num):
        if i != agent_id:
            other_dir = np.array([agents.agent[i].position[0] - agents.agent[agent_id].position[0],
                                   agents.agent[i].position[1] - agents.agent[agent_id].position[1]])
            # dot product
            dot_product = agent_dir[0] * other_dir[0] + agent_dir[1] * other_dir[1]

            # if agent is at behind, check if it is in the lidar circle
            dis = np.sqrt(other_dir[0] ** 2 + other_dir[1] ** 2)
            if dot_product < 0:
                if dis < sensor.radius:
                    is_blind = check_is_blinding_by_agents(agents, agent_id, i)

                    if is_blind:
                        sense_blind_by_agent_list.append(i)
                    else:
                        if circle_obs is not None:
                            is_blind = check_is_blinding_by_obstacle(agents, agent_id, i, circle_obs)
                        if rect_obs is not None:
                            is_blind = check_is_blinding_by_rectangular_obstacle(agents, agent_id, i, rect_obs)

                        if not is_blind:
                            sense_list.append(i)
                        else:
                            sense_blind_by_obstacle_list.append(i)
                else:
                    sense_at_behind_list.append(i)
            elif dot_product >= 0:  # if is in the front
                # check if obstacle blind the agent and the agent is close enough to be seen
                if dis < 100.0:

                    is_blind = check_is_blinding_by_agents(agents, agent_id, i)

                    if is_blind:
                        sense_blind_by_agent_list.append(i)
                    else:

                        if circle_obs is not None:
                            is_blind = check_is_blinding_by_obstacle(agents, agent_id, i, circle_obs)
                        if rect_obs is not None:
                            is_blind = check_is_blinding_by_rectangular_obstacle(agents, agent_id, i, rect_obs)

                        if not is_blind:
                            sense_list.append(i)
                        else:
                            sense_blind_by_obstacle_list.append(i)
                else:
                    sense_too_far_list.append(i)
        else:
            sense_list.insert(0, i)
            new_agent_id = 0
            # sense_list.append(i)
            # new_agent_id = len(sense_list) - 1

    # print("id = ", agent_id, "The agent is at behind : ", sense_at_behind_list)
    # print("id = ", agent_id, "The agent is too far to see : ", sense_too_far_list)
    # print("id = ", agent_id, "The obstacles blind the agent : ", sense_blind_by_obstacle_list)
    # print("id = ", agent_id, "Other agents blind the agent : ", sense_blind_by_agent_list)
    # print("id = ", agent_id, "Sense List : ", sense_list, "     ")
    return sense_list, new_agent_id


#################################################
# Functions for Generating iLQR Solver Parameters
#################################################

def generate_cost_args(agents):
    agent_num = agents.agent_num

    Q = jnp.zeros((agent_num, 4))
    R = jnp.zeros((agent_num, 2))
    D = jnp.zeros(agent_num)
    Back = jnp.zeros(agent_num)

    for i in range(agent_num):
        for j in range(4):
            Q = Q.at[i, j].set(agents.agent[i].Q_cost[j])
            if j < 2:
                R = R.at[i, j].set(agents.agent[i].R_cost[j])
        D = D.at[i].set(agents.agent[i].D_cost)
        Back = Back.at[i].set(agents.agent[i].B_cost)

    cost_args = {
        'Q': Q,
        'R': R,
        'D': D,
        'B': Back
    }

    return cost_args

def update_cost_args_with_adaptive_backup_weight(agents, error_list):
    for i in range(agents.agent_num):
        start_p = [agents.agent[i].initial_state[0], agents.agent[i].initial_state[1]]
        end_p = [agents.agent[i].target_state[0], agents.agent[i].target_state[1]]
        dis = math.dist(start_p, end_p) * 0.9
        if error_list[i] < dis:
            agents.agent[i].B_cost = 100
        else:
            agents.agent[i].B_cost = 0

    return agents

####################################
# Functions for Calculating
####################################

def calc_total_distance_error(agents: MultiAgents):
    """
    Calculate the total distance error between each agent and its goal.
    """
    total_error = 0
    error_list = []
    for i in range(agents.agent_num):
        # update error
        start_p = [agents.agent[i].position[0], agents.agent[i].position[1]]
        end_p = [agents.agent[i].target_state[0], agents.agent[i].target_state[1]]

        dis = math.dist(start_p, end_p)
        error_list.append(dis)
        total_error += dis

    return total_error, error_list


def calc_cost(X, U, agent_num, state_dim, input_dim, agent_target, input_ref, safety_dis, agent_radius, cost_args):
    N = np.size(X, 0)
    n = state_dim
    m = input_dim
    Q = cost_args["Q"]
    R = cost_args["R"]
    D = cost_args["D"]
    B = cost_args["B"]

    def Ctr_i(x_i, u_i, Q_i, R_i, xref_i, uref_i):
        delta_x = x_i - xref_i
        delta_u = u_i - uref_i

        state_cost: float = 0.0
        for i in range(n):
            state_cost += delta_x[i] ** 2 * Q_i[i]

        input_cost: float = 0.0
        for i in range(m):
            input_cost += delta_u[i] ** 2 * R_i[i]

        running_cost = state_cost + input_cost
        return running_cost

    # safety distance cost for two agent
    def Ca_ij(x_i, x_j, safe_dis, D_i):
        d_ij = ((x_i[0] - x_j[0]) ** 2 + (x_i[1] - x_j[1]) ** 2) ** 0.5
        d_prox = safe_dis  # meter

        return np.where(d_ij < d_prox, ((d_ij - d_prox) * D_i * (d_ij - d_prox)), 0)

    # back up cost for single agent
    def C_back(x_i, B_i):
        return np.where(x_i[3] < 0, -B_i * x_i[3], 0)

    # compute total cost
    def stage_cost(x, u, t):
        """ stage_cost = p(x, u, t) """

        sum_Ctr_i = 0
        sum_Ca_ij = 0
        sum_C_back = 0

        for i in range(agent_num):
            x_i = x[i * 4:(i * 4 + 4)]
            u_i = u[i * 2:(i * 2 + 2)]

            sum_Ctr_i += Ctr_i(x_i=x_i, u_i=u_i,
                               Q_i=Q[i], R_i=R[i],
                               xref_i=agent_target[i, :],
                               uref_i=input_ref[i, :])

            for j in range(i + 1, agent_num):
                x_j = x[j * 4:(j * 4 + 4)]
                u_j = u[j * 2:(j * 2 + 2)]

                dis_tmp1 = safety_dis[i] / 2 + agent_radius[j]
                dis_tmp2 = safety_dis[j] / 2 + agent_radius[i]
                safe_dis = np.where(dis_tmp1 < dis_tmp2, dis_tmp2, dis_tmp1)

                sum_Ca_ij += Ca_ij(x_i=x_i, x_j=x_j, safe_dis=safe_dis, D_i=D[i])

            sum_C_back += C_back(x_i=x_i, B_i=B[i])

        return sum_Ctr_i + sum_Ca_ij + sum_C_back

    cost = 0.0
    for t in range(N):
        x = X[t, :]
        u = U[t, :]
        cost = cost + stage_cost(x, u, t)

    return cost


###############################################
# Functions for Reactive safety distance
###############################################

def update_safety_distance_info(time_step, agents, time_step_and_safety_dis):
    safety_dis_is_updated = False
    for i in range(agents.agent_num):
        if agents.agent[i].safety_dis_is_updated and safety_dis_is_updated == False:
            new_safety_dis = np.zeros((1, 1 + agents.agent_num))
            new_safety_dis[0][0] = time_step
            for j in range(agents.agent_num):
                new_safety_dis[0][j + 1] = agents.agent[j].safety_dis
            time_step_and_safety_dis = np.append(time_step_and_safety_dis, new_safety_dis, axis=0)
            safety_dis_is_updated = False
            agents.agent[i].safety_dis_is_updated = False

            print("time_step_and_safety_dis : ", time_step_and_safety_dis)
        elif agents.agent[i].safety_dis_is_updated:
            agents.agent[i].safety_dis_is_updated = False

    return time_step_and_safety_dis


def update_react_param(agents, time_step_and_safety_dis, start_sim_num):
    for i in range(agents.agent_num):
        time_step_and_safety_dis[0][i + 1] = agents.agent[i].org_safety_dis

        stuck_threshold = agents.agent[i].stuck_time_threshold
        if start_sim_num >= stuck_threshold:
            agents.agent[i].last_stuck_time_step = start_sim_num - stuck_threshold

    return agents, time_step_and_safety_dis

###############################################
# Functions for No games Solvers
###############################################

def predict_nonCoop_constant_velocity(agents, N, Ts, boundary=None, circle_obs=None):
    x_predict = jnp.ones((N + 1, agents.state_dim))
    u_predict = jnp.zeros((N, agents.input_dim))

    agent = agents.agent[0]
    position = agent.position
    x_predict = x_predict.at[0, :].set(position)
    is_collision = False
    for i in range(1, N + 1):

        if not is_collision:
            x = position[0] + Ts * position[3] * np.cos(position[2])
            y = position[1] + Ts * position[3] * np.sin(position[2])
            theta = position[2]  # + Ts * control_input[1]
            v = position[3]

            # update next state
            # limit the position inside the boundary or the max/min x-y of all position start & goal
            # if position is in range/boundary and not hit the obstacle, update the position with constant velocity,
            # else remain the last time position
            if boundary[0] < (x - agent.radius) and (x + agent.radius) < boundary[2] \
                    and boundary[1] < (y - agent.radius) and (y + agent.radius) < boundary[3]:
                if circle_obs is None:
                    position = np.array([x, y, theta, v])
                else:
                    collision_obs_index = -1
                    for obs in range(circle_obs.obstacle_num):
                        dis = (circle_obs.radius[obs] + agent.radius) - np.sqrt(
                            (x - circle_obs.pos[obs][0]) ** 2.0 + (y - circle_obs.pos[obs][1]) ** 2.0)
                        if dis > 0:
                            collision_obs_index = obs
                    if collision_obs_index < 0:
                        position = np.array([x, y, theta, v])
                    else:
                        dis = np.sqrt((position[0] - circle_obs.pos[collision_obs_index][0]) ** 2.0 +
                                      (position[1] - circle_obs.pos[collision_obs_index][1]) ** 2.0)

                        increase_dis = dis - circle_obs.radius[collision_obs_index] - agent.radius
                        x = position[0] + increase_dis * np.cos(position[2])
                        y = position[1] + increase_dis * np.sin(position[2])
                        v = 0
                        position = np.array([x, y, theta, v])
                        is_collision = True
            else:
                is_collision = True

        x_predict = x_predict.at[i, :].set(position)

    predict_sol = tuple()

    predict_sol = predict_sol + (x_predict,)
    predict_sol = predict_sol + (u_predict,)

    return predict_sol


def create_vanilla_prediction(N, Ts, agents, boundary, circle_obstacle, hetero_solver=None):
    for i in range(agents.agent_num):
        if hetero_solver is not None:
            solver_name = hetero_solver[i].split('_')[0]
            if solver_name != "IPG":
                agents.agent[i].rebuild_control_opt_series(i)
        else:
            agents.agent[i].rebuild_control_opt_series(i)

        agents_single = copy.copy(agents)
        agents_single.agent_num = 1
        agents_single.agent = [agents_single.agent[i]]

        predict_boundary = [1000000, 1000000, -1000000, -1000000]
        if boundary is None:
            predict_boundary = update_boundary_with_agents(predict_boundary, agents)
        else:
            predict_boundary = boundary
        sol = predict_nonCoop_constant_velocity(agents_single, N, Ts, boundary=predict_boundary,
                                                circle_obs=circle_obstacle)

        agents.agent[i].update_non_cooperated_traj(sol)

    return agents


def generate_braking_solution_nonCoop(N, sol, agent_id, sense_list, agents, boundary=None, circle_obs=None):
    x_opt = np.ones((N + 1, agents.state_dim * agents.agent_num))
    u_opt = np.zeros((N, agents.input_dim * agents.agent_num))

    for i in range(agents.agent_num):
        n1 = i * agents.state_dim
        n2 = i * agents.input_dim

        if i == agent_id:
            agent = agents.agent[agent_id]
            Ts = agent.Ts
            position = agent.position

            acc = 0
            if position[3] > 0.0:
                acc = max((0.0 - position[3]) / Ts, agent.umin[0])
            elif position[3] < 0.0:
                acc = min((0.0 - position[3]) / Ts, agent.umax[0])
            control_input = np.array([acc, 0.0])

            x_opt[0, n1:n1 + agents.state_dim] = position
            u_opt[0, n2:n2 + agents.input_dim] = control_input
            is_collision = False
            for k in range(1, N + 1):
                if not is_collision:

                    if position[3] != 0.0:
                        v = position[3] + Ts * control_input[0]

                        # check sign
                        if (v * position[3]) < 0.0:
                            v = 0.0
                            control_input = control_input.at[0].set((v - position[3]) / Ts)

                        x = position[0] + Ts * v * np.cos(position[2])
                        y = position[1] + Ts * v * np.sin(position[2])
                        theta = position[2]

                        # update next state
                        if boundary[0] < x < boundary[2] and boundary[1] < y < boundary[3]:
                            if circle_obs is None:
                                position = np.array([x, y, theta, v])
                            else:
                                dis_list = []
                                for obs in range(circle_obs.obstacle_num):
                                    dis = (circle_obs.radius[obs] + agents.agent[0].radius) - np.sqrt(
                                        (x - circle_obs.pos[obs][0]) ** 2.0 + (y - circle_obs.pos[obs][1]) ** 2.0)
                                    dis_list.append(dis)
                                if np.all(np.asarray(dis_list) < 0):
                                    position = np.array([x, y, theta, v])
                                else:
                                    is_collision = True
                                    control_input = control_input.at[0].set(0.0)
                        else:
                            is_collision = True
                            control_input = control_input.at[0].set(0.0)
                    else:
                        control_input = control_input.at[0].set(0.0)

                x_opt[k, n1:n1 + agents.state_dim] = position
                if k < N:
                    u_opt[k, n2:n2 + agents.input_dim] = control_input
            # x_opt[:, n1:n1 + agents.state_dim] = sol[0]
            # u_opt[:, n2:n2 + agents.input_dim] = sol[1]
        elif i in sense_list:
            index = sense_list.index(i)
            x_opt[:, n1:n1 + agents.state_dim] = agents.agent[index].pred_non_cooperated_traj
        else:
            x_opt[:, n1:n1 + agents.state_dim] = agents.agent[i].position

    new_sol = tuple()

    for i in range(len(sol)):
        if i == 0:
            new_sol = new_sol + (jax.device_put(x_opt),)
        elif i == 1:
            new_sol = new_sol + (jax.device_put(u_opt),)
        else:
            new_sol = new_sol + (sol[i],)

    return new_sol


###############################################
# Functions for Generating Random Setting
###############################################

def random_setting(problem_type,
                   boundary_range,
                   boundary_exist=False,
                   circle_obstacle=None,
                   rect_obstacle=None,
                   agent_num=0,
                   agent_size_exist=False,
                   agent_size=None,
                   Dis_weight=20,
                   Back_weight=10,
                   safety_dis: np.ndarray = np.array([])):
    """
    Generate random setting


    Generate the random agents, boundary and obstacles setting with
    different requirement.

    Situation 1 :
        Already have the scene, and just want to generate
    different start and goal position.


    --------------------------------------------------------
    PreConditions :
    PostConditions :

    @param problem_type
    @param boundary_range: onp.array([xmin, ymix, xmax, ymax]), maximum range is +-20, minimum range is +-2
    @param boundary_exist
    @param circle_obstacle
    @param agent_num: 2-6
    @param agent_size_exist
    @param agent_size
    @param Dis_weight
    @param Back_weight
    @param safety_dis: 0.4 - 2.0, this is circle diameter

    @return agents, boundary, circle_obs
    """

    # assert boundary_range not exceed the maximum range
    assert min(boundary_range) >= -20 and min(boundary_range) <= -2, "boundary range exceed the range +-20 ~ +-2"
    assert max(boundary_range) <= 20 and max(boundary_range) >= 2, "boundary range exceed the range +-20 ~ +-2"

    # if not give agent number
    # generate agent_num, Q, R, Dis_weight, Back_weight, safety_dis
    Q = np.array([0.01, 0.01, 0, 0])
    R = np.array([1, 1])
    size_ratio_lb = 0.025
    size_ratio_ub = 0.125
    safety_dis_ub = 0.4 # 0.33
    boundary_length = min(boundary_range[2] - boundary_range[0], boundary_range[3] - boundary_range[1])
    max_safety_dis = round(boundary_length * safety_dis_ub, 1)

    def calc_space(boundary_range, circle_obs):
        space_list = []

        for i in range(circle_obs.obstacle_num):
            for j in range(i + 1, circle_obs.obstacle_num):
                dis = np.sqrt(pow(circle_obs.pos[j][0] - circle_obs.pos[i][0], 2)
                               + pow(circle_obs.pos[j][1] - circle_obs.pos[i][1], 2))
                space_length = dis - (circle_obs.radius[i] + circle_obs.radius[j])
                assert space_length >= 0, "obstacle overlap"
                space_list.append(space_length)


            # with boundary
            dis = circle_obs.pos[i][0] - boundary_range[0]
            space_length = dis - circle_obs.radius[i]
            if space_length >= 0:
                space_list.append(space_length)

            dis = boundary_range[2] - circle_obs.pos[i][0]
            space_length = dis - circle_obs.radius[i]
            if space_length >= 0:
                space_list.append(space_length)

            dis = circle_obs.pos[i][1] - boundary_range[1]
            space_length = dis - circle_obs.radius[i]
            if space_length >= 0:
                space_list.append(space_length)

            dis = boundary_range[3] - circle_obs.pos[i][1]
            space_length = dis - circle_obs.radius[i]
            if space_length >= 0:
                space_list.append(space_length)

        max_space = max(space_list)
        min_space = min(space_list)

        return min_space, max_space

    if agent_size_exist is True:
        if agent_size is None:
            if circle_obstacle is not None:
                min_space, max_space = calc_space(boundary_range, circle_obstacle)
                agent_size = round(random.uniform(min_space * 0.6, min_space) / 2, 1)
            else:
                agent_size = round(boundary_length * random.uniform(size_ratio_lb, size_ratio_ub), 1)
    else:
        agent_size = 0

    if agent_num == 0:
        agent_num = random.randint(2, 6)

    if np.size(safety_dis) < 1:
        safety_dis = np.random.randint(size=agent_num, low=(agent_size * 2) * 10, high=max_safety_dis * 10)

    ###################################################################
    # Situation 1
    #
    # Given all the scene information
    # Only need to generate random start & goal position
    ###################################################################

    # generate one set of start & goal heading angle
    theta0_list = np.random.uniform(size=agent_num, low=0, high=2 * np.pi)
    thetaf_list = np.random.uniform(size=agent_num, low=0, high=2 * np.pi)
    agent_list = []

    def check_same_side(segment_start, segment_end, point1, point2):
        # Calculate the vector representations of the line segment
        segment_vector = [segment_end[0] - segment_start[0], segment_end[1] - segment_start[1]]

        # Calculate the vectors from the segment start to the two points
        vector1 = [point1[0] - segment_start[0], point1[1] - segment_start[1]]
        vector2 = [point2[0] - segment_start[0], point2[1] - segment_start[1]]

        # Calculate the cross products
        cross_product1 = vector1[0] * segment_vector[1] - vector1[1] * segment_vector[0]
        cross_product2 = vector2[0] * segment_vector[1] - vector2[1] * segment_vector[0]

        # Check if the cross products have the same sign (same side) or different signs (opposite sides)
        if cross_product1 * cross_product2 >= 0:
            return False  # "Same side"
        else:
            return True  # "Opposite sides"

    def calculate_acute_angle_range(segment1_start, segment1_end, segment2_start, segment2_end):
        # Calculate the vectors representing the line segments
        vector1 = [segment1_end[0] - segment1_start[0], segment1_end[1] - segment1_start[1]]
        vector2 = [segment2_end[0] - segment2_start[0], segment2_end[1] - segment2_start[1]]

        # Calculate the dot product of the vectors
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

        # Calculate the magnitudes of the vectors
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # Calculate the cosine of the angle
        cosine = dot_product / (magnitude1 * magnitude2)

        # Calculate the angle in radians using the inverse cosine (acos) function
        angle_radians = math.acos(cosine)

        # Calculate the acute angle range (from 0 to 2π)
        angle_range_start = 0
        angle_range_end = 2 * math.pi

        if angle_radians < math.pi:
            angle_range_start = angle_radians
            angle_range_end = 2 * math.pi - angle_radians

        return angle_range_start, angle_range_end

    def generate_start_goal(agent_size, agent_list, circle_obs=None, rect_obs=None):
        boundary_pad = boundary_length * 0.05
        x0 = random.uniform(boundary_range[0] + agent_size + boundary_pad, boundary_range[2] - agent_size - boundary_pad)
        y0 = random.uniform(boundary_range[1] + agent_size + boundary_pad, boundary_range[3] - agent_size - boundary_pad)

        # check start position does not hit any others' start position & obstacles
        is_collision = [True, 0]
        while is_collision[0]:

            if is_collision[1] > 0:
                x0 = random.uniform(boundary_range[0] + agent_size + boundary_pad,
                                    boundary_range[2] - agent_size - boundary_pad)
                y0 = random.uniform(boundary_range[1] + agent_size + boundary_pad,
                                    boundary_range[3] - agent_size - boundary_pad)

            # avoid other agent start position
            is_collision[0] = False
            for agent in agent_list:
                x0_other = agent.initial_state[0]
                y0_other = agent.initial_state[1]
                dis = np.sqrt(pow(x0_other - x0, 2) + pow(y0_other - y0, 2))
                if dis < (agent_size * 2):
                    is_collision[0] = True
                    break

            print("is_collision...", is_collision[0])

            if is_collision[0] is True:
                is_collision[1] += 1
                continue

            # avoid circle obstacle
            if circle_obs is not None:
                for obs in range(circle_obs.obstacle_num):
                    dis = np.sqrt(pow(circle_obs.pos[obs][0] - x0, 2) + pow(circle_obs.pos[obs][1] - y0, 2))
                    if dis < (agent_size + circle_obs.radius[obs]):
                        is_collision[0] = True
                        break
                print("avoid obstacle...", is_collision[0])

            if is_collision[0] is True:
                is_collision[1] += 1
                continue

            if rect_obs is not None:
                for obs in rect_obs.pos:
                    x_min = obs[0][0]
                    x_max = obs[2][0]
                    y_min = obs[0][1]
                    y_max = obs[2][1]

                    rect_center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                    rect_width = x_max - x_min
                    rect_height = y_max - y_min

                    # Calculate the distance between the circle center and the closest point on the rectangle
                    closest_x = np.maximum(rect_center[0] - rect_width / 2,
                                           np.minimum(x0, rect_center[0] + rect_width / 2))
                    closest_y = np.maximum(rect_center[1] - rect_height / 2,
                                           np.minimum(y0, rect_center[1] + rect_height / 2))

                    distance = np.sqrt((x0 - closest_x) ** 2 + (y0 - closest_y) ** 2)

                    if distance < agent_size:
                        is_collision[0] = True
                        break
                print("avoid obstacle...", is_collision[0])

            is_collision[1] += 1

        xf = random.uniform(boundary_range[0] + agent_size + boundary_pad,
                            boundary_range[2] - agent_size - boundary_pad)
        yf = random.uniform(boundary_range[1] + agent_size + boundary_pad,
                            boundary_range[3] - agent_size - boundary_pad)
        theta0 = random.uniform(0, 2 * np.pi)
        thetaf = random.uniform(0, 2 * np.pi)

        # check start & goal no collision and
        # goal position does not hit any others' goal position
        is_collision = [True, 0]
        while is_collision[0]:

            if is_collision[1] > 0:
                xf = random.uniform(boundary_range[0] + agent_size + boundary_pad,
                                    boundary_range[2] - agent_size - boundary_pad)
                yf = random.uniform(boundary_range[1] + agent_size + boundary_pad,
                                    boundary_range[3] - agent_size - boundary_pad)

            # avoid moving backward
            angle = np.arctan2(yf - y0, xf - x0)
            angle_threshold = np.pi / 3
            thetaf = random.uniform(angle - angle_threshold, angle + angle_threshold)

            is_collision[0] = False
            for agent in agent_list:
                xf_other = agent.target_state[0]
                yf_other = agent.target_state[1]
                dis = np.sqrt(pow(xf_other - xf, 2) + pow(yf_other - yf, 2))
                if dis < (agent_size * 2):
                    is_collision[0] = True
                    break

            print("is_collision...", is_collision[0])

            if is_collision[0] is True:
                is_collision[1] += 1
                continue

            # avoid circle obstacle
            if circle_obs is not None:
                for obs in range(circle_obs.obstacle_num):
                    dis = np.sqrt(pow(circle_obs.pos[obs][0] - xf, 2) + pow(circle_obs.pos[obs][1] - yf, 2))
                    if dis < (agent_size + circle_obs.radius[obs]):
                        is_collision[0] = True
                        break
                print("avoid obstacle...", is_collision[0])

                if is_collision[0] is False and problem_type == "obs_narrow":
                    obs_id = random.sample(range(circle_obs.obstacle_num), 2)
                    is_opposite = check_same_side(
                        segment_start=circle_obs.pos[obs_id[0]],
                        segment_end=circle_obs.pos[obs_id[1]],
                        point1=[x0, y0],
                        point2=[xf, yf])
                    if is_opposite is False:
                        is_collision[0] = True
                        print("At opposite...", is_opposite)

                    # angle_range_start, angle_range_end = calculate_acute_angle_range(
                    #     segment1_start=[x0, y0],
                    #     segment1_end=circle_obs.pos[obs_id[0]],
                    #     segment2_start=[x0, y0],
                    #     segment2_end=circle_obs.pos[obs_id[1]])
                    angle = np.arctan2(yf - y0, xf - x0)
                    angle_threshold = np.pi / 2
                    theta0 = random.uniform(angle - angle_threshold, angle + angle_threshold)
                    thetaf = random.uniform(angle - angle_threshold, angle + angle_threshold)

            if is_collision[0] is True:
                is_collision[1] += 1
                continue

            if rect_obs is not None:
                for obs in rect_obs.pos:
                    x_min = obs[0][0]
                    x_max = obs[2][0]
                    y_min = obs[0][1]
                    y_max = obs[2][1]

                    rect_center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                    rect_width = x_max - x_min
                    rect_height = y_max - y_min

                    # Calculate the distance between the circle center and the closest point on the rectangle
                    closest_x = np.maximum(rect_center[0] - rect_width / 2,
                                           np.minimum(xf, rect_center[0] + rect_width / 2))
                    closest_y = np.maximum(rect_center[1] - rect_height / 2,
                                           np.minimum(yf, rect_center[1] + rect_height / 2))

                    distance = np.sqrt((xf - closest_x) ** 2 + (yf - closest_y) ** 2)

                    if distance < agent_size:
                        is_collision[0] = True
                        break
                print("avoid obstacle...", is_collision[0])

            is_collision[1] += 1

        return x0, y0, xf, yf, theta0, thetaf

    # generate multi-agents
    for i in range(agent_num):
        x0, y0, xf, yf, theta0, thetaf = generate_start_goal(agent_size, agent_list, circle_obs=circle_obstacle, rect_obs=rect_obstacle)

        agent = SingleAgent(
            initial_state=np.array([round(x0, 2), round(y0, 2), theta0, 0]),
            target_state=np.array([round(xf, 2), round(yf, 2), thetaf, 0]),
            Q=Q,
            R=R,
            Dis_weight=Dis_weight,
            Back_weight=Back_weight,
            safety_dis=round(safety_dis[i] * 0.1, 1))

        agent.radius = agent_size

        agent_list.append(agent)

    agents = MultiAgents(agent_list, agent_num)

    # boundary
    boundary = None
    if boundary_exist is True:
        boundary = boundary_range

    print("Generate random demo setting!")
    return agents, boundary, circle_obstacle


###############################################
# Functions for plotting
###############################################

def update_boundary_with_agents(plot_boundary, agents):
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
