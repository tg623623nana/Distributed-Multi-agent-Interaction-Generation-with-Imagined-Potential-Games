# ilqr_trajectory_planner.py
"""
The iLQR solver for multi-agents trajectory optimization problem

Created on 2023/3/14
@author: Pin-Yun Hung
"""

import functools
import jax.numpy as np
import numpy as onp
from trajax import optimizers
from trajax.integrators import euler


def solve_centralized(U, agents, N, Ts, cost_args, constraints_threshold, boundary=None, circle_obs=None):
    """
    Solve multi-agents centralized trajectory planning with/without obstacles using iLQR solver

    ------------------------------------------------------------------------------
    PreConditions :
    PostConditions :
        return optimized solution, include optimized state & input, optimized cost,
        constraint violation and so on. (Detail looks trajax.optimizers)

    @param          U: reference control input series
    @param   boundary: 2D x-y boundary [xmin, ymin, xmax, ymax]
    @param circle_obs: circle obstacle
    @param    agents: all agents
    @param         N: horizon
    @param        Ts: sampling time
    @param cost_args: penalty weight [Q, R, D, B] (detail looks SingleAgent class)
    @param constraints_threshold

    @return: (Tuple) sol
    """

    # parameter
    n = agents.state_dim  # state = [px, py, theta, v]
    m = agents.input_dim  # input = [a, w]

    ############################
    # dynamic equation
    ############################
    def multi_car(x, u, t):
        del t
        agent_dyn = np.zeros(n * agents.agent_num)

        for i in range(agents.agent_num):
            agent_dyn = agent_dyn.at[i * 4].set(x[3 + i * 4] * np.cos(x[2 + i * 4]))
            agent_dyn = agent_dyn.at[i * 4 + 1].set(x[3 + i * 4] * np.sin(x[2 + i * 4]))
            agent_dyn = agent_dyn.at[i * 4 + 2].set(u[1 + i * 2])
            agent_dyn = agent_dyn.at[i * 4 + 3].set(u[0 + i * 2])

        return agent_dyn

    dynamics = euler(multi_car, dt=Ts)

    ############################
    # objective function
    ############################
    def cost(x, u, t, Q, R, D, B):

        # running cost for single agent
        def Ctr_i(x_i, u_i, Q_i, R_i, xref_i, uref_i):
            delta_x = x_i - xref_i
            delta_u = u_i - uref_i

            state_cost = np.matmul(delta_x, np.matmul(np.diag(Q_i), np.transpose(delta_x)))
            input_cost = np.matmul(delta_u, np.matmul(np.diag(R_i), np.transpose(delta_u)))
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

            for i in range(agents.agent_num):
                x_i = x[i * 4:(i * 4 + 4)]
                u_i = u[i * 2:(i * 2 + 2)]

                sum_Ctr_i += Ctr_i(x_i=x_i, u_i=u_i,
                                   Q_i=Q[i], R_i=R[i],
                                   xref_i=agents.agent[i].target_state,
                                   uref_i=agents.agent[i].input_ref)

                for j in range(i + 1, agents.agent_num):
                    x_j = x[j * 4:(j * 4 + 4)]
                    u_j = u[j * 2:(j * 2 + 2)]

                    dis_tmp1 = agents.agent[i].safety_dis / 2 + agents.agent[j].radius
                    dis_tmp2 = agents.agent[j].safety_dis / 2 + agents.agent[i].radius
                    safe_dis = np.where(dis_tmp1 < dis_tmp2, dis_tmp2, dis_tmp1)

                    sum_Ca_ij += Ca_ij(x_i=x_i, x_j=x_j, safe_dis=safe_dis, D_i=D[i])

                sum_C_back += C_back(x_i=x_i, B_i=B[i])

            return sum_Ctr_i + sum_Ca_ij + sum_C_back

        cost = stage_cost(x, u, t)

        return np.where(t == N, 0.0, cost)

    ############################
    # equality constraint
    ############################
    def equality_constraint(x, u, t):
        del u

        # constraint dimension
        dim = n * agents.agent_num

        def terminate_constraint(x):
            err = np.ones(dim)
            for i in range(agents.agent_num):
                err = err.at[i * n].set(x[i * n] - agents.agent[i].target_state[0])
                err = err.at[i * n + 1].set(x[i * n + 1] - agents.agent[i].target_state[1])
                err = err.at[i * n + 2].set(x[i * n + 2] - agents.agent[i].target_state[2])
                err = err.at[i * n + 3].set(x[i * n + 3] - agents.agent[i].target_state[3])
            return err

        return np.where(t >= N, terminate_constraint(x), np.zeros(dim))  # terminal constraint

    ############################
    # inequality constraint
    ############################
    def inequality_constraint(x, u, t):

        # boundary limits
        def state_limit(x):
            x_lower = np.ones(agents.agent_num * n)
            x_upper = np.ones(agents.agent_num * n)

            M = 1000000  # a large number means no limit
            for i in range(agents.agent_num):
                x_lower = x_lower.at[i * n].set(boundary[0] + agents.agent[i].radius)
                x_lower = x_lower.at[i * n + 1].set(boundary[1] + agents.agent[i].radius)
                x_lower = x_lower.at[i * n + 2].set(-M)
                x_lower = x_lower.at[i * n + 3].set(agents.agent[i].vmin)

                x_upper = x_upper.at[i * n].set(boundary[2] - agents.agent[i].radius)
                x_upper = x_upper.at[i * n + 1].set(boundary[3] - agents.agent[i].radius)
                x_upper = x_upper.at[i * n + 2].set(M)
                x_upper = x_upper.at[i * n + 3].set(agents.agent[i].vmax)

            return np.concatenate((x_lower - x, x - x_upper))

        # control limits
        def control_limits(u):
            u_lower = np.ones(agents.agent_num * m)
            u_upper = np.ones(agents.agent_num * m)

            for i in range(agents.agent_num):
                u_lower = u_lower.at[i * m].set(agents.agent[i].umin[0])
                u_lower = u_lower.at[i * m + 1].set(agents.agent[i].umin[1])

                u_upper = u_upper.at[i * m].set(agents.agent[i].umax[0])
                u_upper = u_upper.at[i * m + 1].set(agents.agent[i].umax[1])

            return np.concatenate((u_lower - u, u - u_upper))

        # circle obstacle avoidance
        def cirObstacles_constraint(x, obs):
            obs_constraint = np.zeros(agents.agent_num * obs.obstacle_num)
            count = 0
            for i in range(agents.agent_num):
                n1 = i * agents.state_dim
                for j in range(obs.obstacle_num):
                    dis = (obs.radius[j] + agents.agent[i].radius) - np.sqrt(
                        (x[n1] - obs.pos[j][0]) ** 2.0 + (x[n1 + 1] - obs.pos[j][1]) ** 2.0)
                    obs_constraint = obs_constraint.at[count].set(dis)
                    count += 1

            return obs_constraint

        # agent collision avoidance
        def agentsize_constraint(x):
            num = 0
            for i in range(agents.agent_num):
                num += i
            agentsize_constraint = np.zeros(num)
            count = 0
            for i in range(agents.agent_num):
                n1 = i * agents.state_dim
                for j in range(i + 1, agents.agent_num):
                    nj = j * agents.state_dim
                    dis = (agents.agent[i].radius + agents.agent[j].radius) - \
                          np.sqrt((x[n1] - x[nj]) ** 2.0 + (x[n1 + 1] - x[nj + 1]) ** 2.0)
                    agentsize_constraint = agentsize_constraint.at[count].set(dis)
                    count += 1

            return agentsize_constraint

        num = 0
        for i in range(agents.agent_num):
            num += i

        if circle_obs is None:
            if boundary is None:  # no circle obstacle & no boundary limitation
                return np.where(t == N, np.zeros(agents.agent_num * m * 2 + num),
                                np.concatenate((control_limits(u), agentsize_constraint(x))))
            else:  # no circle obstacle & has boundary limitation
                return np.where(t == N, np.zeros(agents.agent_num * n * 2 + agents.agent_num * m * 2 + num),
                                np.concatenate((state_limit(x), control_limits(u), agentsize_constraint(x))))
        else:
            if boundary is None:  # has circle obstacle & no boundary limitation
                return np.where(t == N,
                                np.concatenate((np.zeros(agents.agent_num * m * 2 + num),
                                                cirObstacles_constraint(x, circle_obs))),
                                np.concatenate((control_limits(u), agentsize_constraint(x),
                                                cirObstacles_constraint(x, circle_obs))))
            else:  # has circle obstacle & has boundary limitation
                return np.where(t == N,
                                np.concatenate((np.zeros(agents.agent_num * n * 2 + agents.agent_num * m * 2 + num),
                                                cirObstacles_constraint(x, circle_obs))),
                                np.concatenate((state_limit(x), control_limits(u), agentsize_constraint(x),
                                                cirObstacles_constraint(x, circle_obs))))

    # initial state
    x0 = onp.zeros(n * agents.agent_num)
    for i in range(agents.agent_num):
        x0[i * 4] = agents.agent[i].position[0]
        x0[i * 4 + 1] = agents.agent[i].position[1]
        x0[i * 4 + 2] = agents.agent[i].position[2]
        x0[i * 4 + 3] = agents.agent[i].position[3]

    # solve optimization problem
    sol = optimizers.constrained_ilqr(
        functools.partial(cost, **cost_args), dynamics, x0, U,
        equality_constraint=equality_constraint,
        inequality_constraint=inequality_constraint,
        constraints_threshold=constraints_threshold,
        maxiter_al=10)

    return sol


def solve_IPG(U, agent_id, agents, N, Ts, cost_args, constraints_threshold, boundary=None, circle_obs=None):
    """
    Solve multi-agents distributed planning with games (with obstacles) using iLQR solver


    In the distributed planning, we assume that the target state
    of all agents are known and the only thing they don't know is
    others' safety distance.

    ------------------------------------------------------------------------------
    PreConditions :
    PostConditions :
        return optimized solution, include optimized state & input, optimized cost,
        constraint violation and so on. (Detail looks trajax.optimizers)

    @param          U: reference control input series
    @param   boundary: 2D x-y boundary [xmin, ymin, xmax, ymax]
    @param circle_obs: circle obstacle
    @param  agent_id: the order of the agent which is solving the problem
    @param    agents: all agents
    @param         N: horizon
    @param        Ts: sampling time
    @param cost_args: penalty weight [Q, R, D, B] (detail looks SingleAgent class)
    @param constraints_threshold

    @return: (Tuple) sol
    """

    # parameter
    n = agents.state_dim  # state = [px, py, theta, v]
    m = agents.input_dim  # input = [a, w]

    ############################
    # dynamic equation
    ############################
    def multi_car(x, u, t):
        del t
        agent_dyn = np.zeros(n * agents.agent_num)

        for i in range(agents.agent_num):
            agent_dyn = agent_dyn.at[i * 4].set(x[3 + i * 4] * np.cos(x[2 + i * 4]))
            agent_dyn = agent_dyn.at[i * 4 + 1].set(x[3 + i * 4] * np.sin(x[2 + i * 4]))
            agent_dyn = agent_dyn.at[i * 4 + 2].set(u[1 + i * 2])
            agent_dyn = agent_dyn.at[i * 4 + 3].set(u[0 + i * 2])

        return agent_dyn

    dynamics = euler(multi_car, dt=Ts)

    ############################
    # objective function
    ############################
    def cost(x, u, t, Q, R, D, B):

        # running cost for single agent
        def Ctr_i(x_i, u_i, Q_i, R_i, xref_i, uref_i):
            delta_x = x_i - xref_i
            delta_u = u_i - uref_i

            state_cost = np.matmul(delta_x, np.matmul(np.diag(Q_i), np.transpose(delta_x)))
            input_cost = np.matmul(delta_u, np.matmul(np.diag(R_i), np.transpose(delta_u)))
            running_cost = state_cost + input_cost
            return running_cost

        # safety distance cost for two agent
        def Ca_ij(x_i, x_j, D_i, other_agent_id):
            d_ij = ((x_i[0] - x_j[0]) ** 2 + (x_i[1] - x_j[1]) ** 2) ** 0.5
            d_prox = agents.agent[agent_id].safety_dis / 2 + agents.agent[other_agent_id].radius  # meter

            return np.where(d_ij < d_prox, ((d_ij - d_prox) * D_i * (d_ij - d_prox)), 0)  # D_i = 20

        # back up cost for single agent
        def C_back(x_i, B_i):
            return np.where(x_i[3] < 0, -B_i * x_i[3], 0)

        # compute total cost in one time step
        def stage_cost(x, u, t):
            """ stage_cost = p(x, u, t) """

            sum_Ctr_i = 0
            sum_Ca_ij = 0
            sum_C_back = 0

            for i in range(agents.agent_num):
                x_i = x[i * 4:(i * 4 + 4)]
                u_i = u[i * 2:(i * 2 + 2)]

                sum_Ctr_i += Ctr_i(x_i=x_i, u_i=u_i,
                                   Q_i=Q[i], R_i=R[i],
                                   xref_i=agents.agent[i].target_state,
                                   uref_i=agents.agent[i].input_ref)

                for j in range(i + 1, agents.agent_num):
                    x_j = x[j * 4:(j * 4 + 4)]

                    sum_Ca_ij += Ca_ij(x_i=x_i, x_j=x_j, D_i=D[agent_id], other_agent_id=j)

                sum_C_back += C_back(x_i=x_i, B_i=B[i])

            return sum_Ctr_i + sum_Ca_ij + sum_C_back

        cost = stage_cost(x, u, t)

        return np.where(t == N, 0.0, cost)

    ############################
    # equality constraint
    ############################
    def equality_constraint(x, u, t):
        del u

        # constraint dimension
        dim = n * agents.agent_num

        def terminate_constraint(x):
            err = np.ones(dim)
            for i in range(agents.agent_num):

                err = err.at[i * n].set(x[i * n] - agents.agent[i].target_state[0])
                err = err.at[i * n + 1].set(x[i * n + 1] - agents.agent[i].target_state[1])
                err = err.at[i * n + 2].set(x[i * n + 2] - agents.agent[i].target_state[2])
                err = err.at[i * n + 3].set(x[i * n + 3] - agents.agent[i].target_state[3])
            return err

        return np.where(t >= N, terminate_constraint(x), np.zeros(dim))

    ############################
    # inequality constraint
    ############################
    def inequality_constraint(x, u, t):

        # boundary limits
        def state_limit(x):
            x_lower = np.ones(agents.agent_num * n)
            x_upper = np.ones(agents.agent_num * n)

            M = 1000000  # a large number means no limit
            for i in range(agents.agent_num):
                x_lower = x_lower.at[i * 4].set(boundary[0] + agents.agent[i].radius)
                x_lower = x_lower.at[i * 4 + 1].set(boundary[1] + agents.agent[i].radius)
                x_lower = x_lower.at[i * 4 + 2].set(-M)
                x_lower = x_lower.at[i * 4 + 3].set(agents.agent[i].vmin)

                x_upper = x_upper.at[i * 4].set(boundary[2] - agents.agent[i].radius)
                x_upper = x_upper.at[i * 4 + 1].set(boundary[3] - agents.agent[i].radius)
                x_upper = x_upper.at[i * 4 + 2].set(M)
                x_upper = x_upper.at[i * 4 + 3].set(agents.agent[i].vmax)

            return np.concatenate((x_lower - x, x - x_upper))

        # control limits
        def control_limits(u):
            u_lower = np.ones(agents.agent_num * m)
            u_upper = np.ones(agents.agent_num * m)

            for i in range(agents.agent_num):
                u_lower = u_lower.at[i * 2].set(agents.agent[i].umin[0])
                u_lower = u_lower.at[i * 2 + 1].set(agents.agent[i].umin[1])
                u_upper = u_upper.at[i * 2].set(agents.agent[i].umax[0])
                u_upper = u_upper.at[i * 2 + 1].set(agents.agent[i].umax[1])

            return np.concatenate((u_lower - u, u - u_upper))

        # circle obstacle avoidance
        def cirObstacles_constraint(x, obs):
            obs_constraint = np.zeros(agents.agent_num * obs.obstacle_num)
            count = 0
            for i in range(agents.agent_num):
                n1 = i * agents.state_dim
                for j in range(obs.obstacle_num):
                    dis = (obs.radius[j] + agents.agent[i].radius) - np.sqrt(
                        (x[n1] - obs.pos[j][0]) ** 2.0 + (x[n1 + 1] - obs.pos[j][1]) ** 2.0)
                    obs_constraint = obs_constraint.at[count].set(dis)
                    count += 1

            return obs_constraint

        # agent collision avoidance
        def agentsize_constraint(x):
            num = 0
            for i in range(agents.agent_num):
                num += i
            agentsize_constraint = np.zeros(num)
            count = 0
            for i in range(agents.agent_num):
                n1 = i * agents.state_dim
                for j in range(i + 1, agents.agent_num):
                    nj = j * agents.state_dim
                    dis = (agents.agent[i].radius + agents.agent[j].radius) - \
                          np.sqrt((x[n1] - x[nj]) ** 2.0 + (x[n1 + 1] - x[nj + 1]) ** 2.0)
                    agentsize_constraint = agentsize_constraint.at[count].set(dis)
                    count += 1

            return agentsize_constraint

        num = 0
        for i in range(agents.agent_num):
            num += i

        if circle_obs is None:
            if boundary is None:  # no circle obstacle & no boundary limitation
                return np.where(t == N, np.zeros(agents.agent_num * m * 2 + num),
                                np.concatenate((control_limits(u), agentsize_constraint(x))))
            else:  # no circle obstacle & has boundary limitation
                return np.where(t == N, np.zeros(agents.agent_num * n * 2 + agents.agent_num * m * 2 + num),
                                np.concatenate((state_limit(x), control_limits(u), agentsize_constraint(x))))
        else:
            if boundary is None:  # has circle obstacle & no boundary limitation
                return np.where(t == N,
                                np.concatenate((np.zeros(agents.agent_num * m * 2 + num),
                                                cirObstacles_constraint(x, circle_obs))),
                                np.concatenate((control_limits(u), agentsize_constraint(x),
                                                cirObstacles_constraint(x, circle_obs))))
            else:  # has circle obstacle & has boundary limitation
                return np.where(t == N,
                                np.concatenate((np.zeros(agents.agent_num * n * 2 + agents.agent_num * m * 2 + num),
                                                cirObstacles_constraint(x, circle_obs))),
                                np.concatenate((state_limit(x), control_limits(u), agentsize_constraint(x),
                                                cirObstacles_constraint(x, circle_obs))))


    # initial state
    x0 = onp.zeros(n * agents.agent_num)
    for i in range(agents.agent_num):
        x0[i * 4] = agents.agent[i].position[0]
        x0[i * 4 + 1] = agents.agent[i].position[1]
        x0[i * 4 + 2] = agents.agent[i].position[2]
        x0[i * 4 + 3] = agents.agent[i].position[3]

    # solve optimization problem
    sol = optimizers.constrained_ilqr(
        functools.partial(cost, **cost_args), dynamics, x0, U,
        equality_constraint=equality_constraint,
        inequality_constraint=inequality_constraint,
        constraints_threshold=constraints_threshold,
        maxiter_al=10)

    return sol


def solve_vanilla(U, agent_id, agent, N, Ts, cost_args, constraints_threshold, boundary, circle_obs):
    """
    Solve the one agent trajectory optimization problem without predicting other agents will have
    cooperated behavior
    """

    # parameter
    n = agent.state_dim  # state = [px, py, theta, v]
    m = agent.input_dim  # input = [a, w]

    ############################
    # dynamic equation
    ############################
    def multi_car(x, u, t):
        del t
        agent_dyn = np.zeros(n)

        agent_dyn = agent_dyn.at[0].set(x[3] * np.cos(x[2]))
        agent_dyn = agent_dyn.at[1].set(x[3] * np.sin(x[2]))
        agent_dyn = agent_dyn.at[2].set(u[1])
        agent_dyn = agent_dyn.at[3].set(u[0])

        return agent_dyn

    dynamics = euler(multi_car, dt=Ts)

    ############################
    # objective function
    ############################
    def cost(x, u, t, Q, R, D, B):

        # running cost for single agent
        def Ctr_i(x_i, u_i, Q_i, R_i, xref_i, uref_i):
            delta_x = x_i - xref_i
            delta_u = u_i - uref_i

            state_cost = np.matmul(delta_x, np.matmul(np.diag(Q_i), np.transpose(delta_x)))
            input_cost = np.matmul(delta_u, np.matmul(np.diag(R_i), np.transpose(delta_u)))
            running_cost = state_cost + input_cost
            return running_cost

        # inter-agent cost for two agent
        def Ca_ij(x_i, x_j, D_i, other_agent_id):
            d_ij = ((x_i[0] - x_j[0]) ** 2 + (x_i[1] - x_j[1]) ** 2) ** 0.5
            d_prox = agent.agent[agent_id].safety_dis / 2 + agent.agent[other_agent_id].radius  # meter

            return np.where(d_ij < d_prox, ((d_ij - d_prox) * D_i * (d_ij - d_prox)), 0)  # D_i = 20

        def C_back(x_i, B_i):
            return np.where(x_i[3] < 0, -B_i * x_i[3], 0)

        def stage_cost(x, u, t):
            """ stage_cost = p(x, u, t) """

            sum_Ctr_i = 0
            sum_Ca_ij = 0
            sum_C_back = 0

            x_i = x  # [0:n]
            u_i = u  # [0:m]

            i = agent_id
            sum_Ctr_i += Ctr_i(x_i=x_i, u_i=u_i,
                               Q_i=Q[i], R_i=R[i],
                               xref_i=agent.agent[i].target_state,
                               uref_i=agent.agent[i].input_ref)

            for j in range(agent.agent_num):
                if j != agent_id:
                    x_j = agent.agent[j].pred_non_cooperated_traj[t, :]

                    sum_Ca_ij += Ca_ij(x_i=x_i, x_j=x_j, D_i=D[agent_id], other_agent_id=j)

            sum_C_back += C_back(x_i=x_i, B_i=B[agent_id])

            return sum_Ctr_i + sum_Ca_ij + sum_C_back

        cost = stage_cost(x, u, t)

        return np.where(t == N, 0.0, cost)

    ############################
    # equality constraint
    ############################
    def equality_constraint(x, u, t):
        # maximum constraint dimension across time steps
        dim = n

        def terminate_constraint(x):
            i = agent_id
            err = np.ones(dim)

            err = err.at[0].set(x[0] - agent.agent[i].target_state[0])
            err = err.at[1].set(x[1] - agent.agent[i].target_state[1])
            err = err.at[2].set(x[2] - agent.agent[i].target_state[2])
            err = err.at[3].set(x[3] - agent.agent[i].target_state[3])
            return err

        return np.where(t == N, terminate_constraint(x), np.zeros(dim))

    ############################
    # inequality constraint
    ############################
    def inequality_constraint(x, u, t):

        # boundary
        def state_limit(x):
            x_lower = np.ones(n)
            x_upper = np.ones(n)

            M = 1000000  # a large number means no limit
            i = agent_id
            x_lower = x_lower.at[0].set(boundary[0] + agent.agent[i].radius)
            x_lower = x_lower.at[1].set(boundary[1] + agent.agent[i].radius)
            x_lower = x_lower.at[2].set(-M)
            x_lower = x_lower.at[3].set(agent.agent[i].vmin)

            x_upper = x_upper.at[0].set(boundary[2] - agent.agent[i].radius)
            x_upper = x_upper.at[1].set(boundary[3] - agent.agent[i].radius)
            x_upper = x_upper.at[2].set(M)
            x_upper = x_upper.at[3].set(agent.agent[i].vmax)

            return np.concatenate((x_lower - x, x - x_upper))

        def velocity_limit(x):
            # x_lower = -M
            x_lower = agent.agent[0].vmin

            # x_upper = M
            x_upper = agent.agent[0].vmax

            return np.hstack((x_lower - x[3], x[3] - x_upper))

        # control limits
        def control_limits(u):
            u_lower = np.ones(m)
            u_upper = np.ones(m)

            i = agent_id
            u_lower = u_lower.at[0].set(agent.agent[i].umin[0])
            u_lower = u_lower.at[1].set(agent.agent[i].umin[1])

            u_upper = u_upper.at[0].set(agent.agent[i].umax[0])
            u_upper = u_upper.at[1].set(agent.agent[i].umax[1])

            return np.concatenate((u_lower - u, u - u_upper))

        def cirObstacles_constraint(x, obs):
            obs_constraint = np.zeros(obs.obstacle_num)
            count = 0

            i = agent_id
            for j in range(obs.obstacle_num):
                dis = (obs.radius[j] + agent.agent[i].radius) - np.sqrt(
                    (x[0] - obs.pos[j][0]) ** 2.0 + (x[1] - obs.pos[j][1]) ** 2.0)
                obs_constraint = obs_constraint.at[count].set(dis)
                count += 1

            return obs_constraint

        def agentsize_constraint(x, t):

            num = agent.agent_num - 1
            agentsize_constraint = np.zeros(num)
            count = 0

            i = agent_id
            for j in range(agent.agent_num):
                if j != i:
                    x_j = agent.agent[j].pred_non_cooperated_traj[t, :]
                    dis = (agent.agent[i].radius + agent.agent[j].radius) - \
                          np.sqrt((x[0] - x_j[0]) ** 2.0 + (x[1] - x_j[1]) ** 2.0)
                    agentsize_constraint = agentsize_constraint.at[count].set(dis)
                    count += 1

            return agentsize_constraint

        num = agent.agent_num - 1

        if circle_obs is None:
            if boundary is None:  # no circle obstacle & no boundary limitation
                return np.where(t == N, np.zeros(2 + m * 2 + num),
                                np.concatenate((velocity_limit(x), control_limits(u), agentsize_constraint(x, t))))
            else:  # no circle obstacle & has boundary limitation
                return np.where(t == N, np.zeros(n * 2 + m * 2 + num),
                                np.concatenate((state_limit(x), control_limits(u), agentsize_constraint(x, t))))
        else:
            if boundary is None:  # has circle obstacle & no boundary limitation
                return np.where(t == N,
                                np.concatenate((np.zeros(m * 2 + num),
                                                cirObstacles_constraint(x, circle_obs))),
                                np.concatenate((control_limits(u), agentsize_constraint(x, t),
                                                cirObstacles_constraint(x, circle_obs))))
            else:  # has circle obstacle & has boundary limitation
                return np.where(t == N,
                                np.concatenate((np.zeros(n * 2 + m * 2 + num),
                                                cirObstacles_constraint(x, circle_obs))),
                                np.concatenate((state_limit(x), control_limits(u), agentsize_constraint(x, t),
                                                cirObstacles_constraint(x, circle_obs))))


    # initial state
    x0 = onp.zeros(n)

    x0[0] = agent.agent[agent_id].position[0]
    x0[1] = agent.agent[agent_id].position[1]
    x0[2] = agent.agent[agent_id].position[2]
    x0[3] = agent.agent[agent_id].position[3]

    sol = optimizers.constrained_ilqr(
        functools.partial(cost, **cost_args), dynamics, x0, U,
        equality_constraint=equality_constraint,
        inequality_constraint=inequality_constraint,
        constraints_threshold=constraints_threshold,
        maxiter_al=10)

    return sol
