# test.py
"""
Test : check
 1. jax.jit acceleration
 2. multi-thread work

Created on 2023/3/13
@author: Pin-Yun Hung
"""

import functools
from functools import partial
from jax.lib import xla_bridge
import time
import jax
import jax.numpy as np
from trajax import optimizers
from trajax.integrators import rk4
import concurrent.futures
import chex
import matplotlib.pylab as plt
from typing import Any

PyTree = Any
Array = jax.Array
Scalar = chex.Scalar


@partial(jax.jit, static_argnums=())
def testConstrainedCarSolve():
    T = 25
    dt = 0.1
    n = 3
    m = 2
    goal = np.array([1.0, 0.0, 0.0])

    def car(x, u, t):
      del t
      return np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])

    dynamics = rk4(car, dt=dt)

    cost_args = {
        'x_stage_cost': 0.1,
        'u_stage_cost': 1.0,
    }

    def cost(x, u, t, x_stage_cost, u_stage_cost):
      delta = x - goal
      stagewise_cost = 0.5 * x_stage_cost * np.dot(
          delta, delta) + 0.5 * u_stage_cost * np.dot(u, u)
      return np.where(t == T, 0.0, stagewise_cost)

    def equality_constraint(x, u, t):
      del u

      # maximum constraint dimension across time steps
      dim = 3

      def goal_constraint(x):
        err = x - goal
        return err

      return np.where(t == T, goal_constraint(x), np.zeros(dim))

    # obstacles
    obs1 = {'px': 0.5, 'py': 0.01, 'r': 0.25}

    # control limits
    u_lower = -1.0 * np.ones(m)
    u_upper = 1.0 * np.ones(m)

    def inequality_constraint(x, u, t):
        def obstacles(x):
            return np.array([obs1['r'] - np.sqrt((x[0] - obs1['px'])**2.0 + (x[1] - obs1['py'])**2.0)])

        def control_limits(u):
            return np.concatenate((u_lower - u, u - u_upper))

        return np.where(t == T, np.concatenate((np.zeros(2 * m), obstacles(x))),
                        np.concatenate((control_limits(u), obstacles(x))))

    x0 = np.zeros(n)
    U = np.zeros((T, m))

    constraints_threshold = 1.0e-3

    # start = time.perf_counter()
    sol = optimizers.constrained_ilqr(
        functools.partial(cost, **cost_args), dynamics, x0, U,
        equality_constraint=equality_constraint,
        inequality_constraint=inequality_constraint,
        constraints_threshold=constraints_threshold,
        maxiter_al=10)
    # end = time.perf_counter()
    # print(thread_id, "  Execute Time : {} sec".format((end - start)))

    # # test constraints
    # X = sol[0]
    # U = sol[1]
    # equality_constraints = sol[5]
    # inequality_constraints = sol[6]

    return sol


def test_multiThread(thread_num):

    # Without multi-thread
    start = time.perf_counter()
    for i in range(thread_num):
        testConstrainedCarSolve(i)
    end = time.perf_counter()
    print("Total Execute Time : {} sec".format((end - start)))

    # Using multi-thread
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=thread_num)

    start = time.perf_counter()
    results = [thread_pool.submit(testConstrainedCarSolve, i) for i in range(thread_num)]

    concurrent.futures.wait(results)
    end = time.perf_counter()
    print("Total Execute Time : {} sec".format((end - start)))


@partial(jax.jit, static_argnums=(4, 5))
def constrained_car_solver(
        U: Array,
        constraints_threshold: float,
        T: chex.Scalar,
        dt: chex.Scalar,
        n: int,
        m: int,
        goal: Array,
        cost_args: dict,
        obs: dict,
        input_limit: chex.Scalar) -> ...:

    x0 = np.zeros(n)

    def car(x: Array, u: Array, t: chex.Scalar) -> Array:
        del t
        return np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])

    dynamics = rk4(car, dt=dt)

    def cost(x: Array, u: Array, t: chex.Scalar,
             x_stage_cost: chex.Scalar, u_stage_cost: chex.Scalar):
        delta = x - goal
        stagewise_cost = 0.5 * x_stage_cost * np.dot(
          delta, delta) + 0.5 * u_stage_cost * np.dot(u, u)
        return np.where(t == T, 0.0, stagewise_cost)

    def equality_constraint(x: Array, u: Array, t: chex.Scalar):

        def goal_constraint(x: Array):
            err = x - goal
            return err

        return np.where(t == T, goal_constraint(x), np.zeros(n))

    def inequality_constraint(x: Array, u: Array, t: Scalar):
        def obstacles(x: Array) -> Array:
            return np.array([obs['r'] - np.sqrt((x[0] - obs['px'])**2.0 + (x[1] - obs['py'])**2.0)])

        def control_limits(u: Array) -> Array:
            u_lower = -input_limit * np.ones(m)
            u_upper = input_limit * np.ones(m)

            return np.concatenate((u_lower - u, u - u_upper))

        return np.where(t == T, np.concatenate((np.zeros(2 * m), obstacles(x))),
                      np.concatenate((control_limits(u), obstacles(x))))

    sol = optimizers.constrained_ilqr(
        functools.partial(cost, **cost_args), dynamics, x0, U,
        equality_constraint=equality_constraint,
        inequality_constraint=inequality_constraint,
        constraints_threshold=constraints_threshold,
        maxiter_al=10)

    return sol


if __name__ == '__main__':
    # check using cpu or gpu
    print("Using : ", xla_bridge.get_backend().platform)

    ##########################
    # Test Jit works
    ##########################
    print("\n----------- Check Accelerate -----------")
    start_time = time.time()
    sol = testConstrainedCarSolve()
    time_taken = time.time() - start_time
    print(f"Compile    obj={sol[8]:.3f}, time_taken={time_taken:.3f} (s)")


    X = sol[0]
    x1 = X[:, 0]
    y1 = X[:, 1]

    for i in range(5):
        start_time = time.time()

        sol = testConstrainedCarSolve()

        time_taken = time.time() - start_time
        print(f"iteration {i}    obj={sol[8]:.3f}, time_taken={time_taken:.3f} (s)")

    ##########################
    # Multi-thread Test
    ##########################
    print("\n----------- Check multi-thread works -----------")
    thread_num = 5
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=thread_num)

    start = time.perf_counter()
    results = [thread_pool.submit(testConstrainedCarSolve, ) for i in range(thread_num)]

    concurrent.futures.wait(results)
    end = time.perf_counter()
    print(f"Multi-Thread Execute Time {(end - start): .3f}    thread number = {thread_num}")

    ##################################
    # check the correction of solutions by plotting
    ##################################
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Left: compile, Right: accelerate')
    X = sol[0]
    x2 = X[:, 0]
    y2 = X[:, 1]
    ax1.plot(x1, y1)
    ax2.plot(x2, y2)
    plt.show()