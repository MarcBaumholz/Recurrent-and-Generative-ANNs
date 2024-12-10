#! /urs/bin/env python3


"""
In this file we generate 2D circular wave data expanding from a point source
outwards and reflecting at the boundaries of the simulation domain. Waves are
generate by using the two dimensional wave equation, a partial differential
equation which we solve using the finite difference method to discretize in
space and time. See Section 4.2 in https://arxiv.org/abs/1912.11141 for a
derivation of the solution.
"""

import argparse
import os

import numpy as np
import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def f(_x, _y, _varx, _vary, _a):
    """
    Function to set the initial activity of the field. We use the Gaussian bell
    curve to initialize the field smoothly.
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :param _varx: The variance in x-direction
    :param _vary: The variance in y-direction
    :param _a: The amplitude of the wave
    :return: The initial activity at (x, y)
    """
    x_part = ((_x-start_pt[0])**2)/(2*_varx)
    y_part = ((_y-start_pt[1])**2)/(2*_vary)
    return _a*np.exp(-(x_part+y_part))


def g(_x, _y, _varx, _vary, _a):
    """
    Function to determine the changes over time in the field
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :param _varx: The variance in x-direction
    :param _vary: The variance in y-direction
    :param _a: The amplitude of the wave
    :return: The changes over time in the field at (x, y)
    """
    # x_part = _x * f(_x, _y, _varx, _vary, _a)
    # y_part = _y * f(_x, _y, _varx, _vary, _a)
    # return (x_part + y_part) / 2.
    return 0.0


def u(_t, _x, _y):
    """
    Function to calculate the field activity in time step t at (x, y)
    :param _t: The current time step
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :param _c: The wave velocity
    :return: The field activity at position x, y in time step t.
    """

    # Compute changes in x- and y-direction
    dxxu = dxx_u(_t, _x, _y)
    dyyu = dyy_u(_t, _x, _y)

    # Get the activity at x and y in time step t
    u_t = field[t, _x, _y]

    # Catch initial condition, where there is no value of the field at time step
    # (t-1) yet
    if _t == 0:
        u_t_1 = dt_u(_x, _y)
        # u_t_1 = 0.0
    else:
        u_t_1 = field[_t-1, _x, _y]

    #c = velocity_ary[_x, _y]

    # Incorporate the changes in x- and y-direction and return the activity
    #return damp*((c*(dt**2))*(dxxu+dyyu)+2*u_t-u_t_1)
    return damp*(((c**2)*(dt ** 2))*(dxxu+dyyu)+2*u_t-u_t_1)


def dxx_u(_t, _x, _y):
    """
    The second derivative of u to x. Computes the lateral activity change in
    x-direction.
    Neuman Boundary conditions to prevent waves from reflecting at edges are
    taken from https://12000.org/my_notes/neumman_BC/Neumman_BC.htm
    :param _t: The current time step
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :return: Field activity in t, considering changes in x-direction
    """

    # (Neuman) Boundary condition at left end of the field
    if _x == 0:
        dx_left = 0.0

        # g_xy = g(_x, _y, wave_width_x, wave_width_y, amplitude)
        # u_right = field[_t, _x + 1, _y]
        #
        # if _y == 0:
        #     u_top = 0.0
        # else:
        #     u_top = field[_t, _x, _y - 1]
        #
        # if _y == height - 1:
        #     u_bot = 0.0
        # else:
        #     u_bot = field[_t, _x, _y + 1]
        #
        # u_xy = field[_t, _x, _y]
        #
        # dx_left = (1/4) * (2*u_right - 2*dx*g_xy + u_top + u_bot + dt**2*u_xy)
    else:
        dx_left = field[_t, int(_x-dx), _y]

    # Boundary condition at right end of the field
    if _x == width-1:
        dx_right = 0.0
    else:
        dx_right = field[_t, int(_x+dx), _y]

    # Calculate change in x-direction and return it
    ut_dx = dx_right - 2*field[_t, _x, _y] + dx_left

    return ut_dx/np.square(dx)


def dyy_u(_t, _x, _y):
    """
    The second derivative of u to y. Computes the lateral activity change in
    y-direction.
    :param _t: The current time step
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :return: Field activity in t, considering changes in y-direction
    """

    # Boundary condition at top end of the field
    if _y == 0:
        dy_above = 0.0
    else:
        dy_above = field[_t, _x, int(_y-dy)]

    # Boundary condition at bottom end of the field
    if _y == height-1:
        dy_below = 0.0
    else:
        dy_below = field[_t, _x, int(_y+dy)]

    # Calculate change in y-direction and return it
    ut_dy = dy_below - 2*field[_t, _x, _y] + dy_above

    return ut_dy/np.square(dy)


def dt_u(_x, _y):
    """
    First derivative of u to t, only required in the very first time step to
    compute u(-dt, x, y).
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :return: The value of the field at (t-1), x, y
    """
    return field[1, _x, _y] - 2*dt*g(_x, _y, wave_width_x, wave_width_y, amplitude)


def animate(_t):
    im.set_array(field[_t, :, :])
    return im


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="2D wave dynamics generator. Particular properties of the configuration can be overwritten, as "
                    "listed by the -h flag.")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Temporal step size of the solver.")
    parser.add_argument("--dy", type=float, default=1.0,
                        help="Spatial step size of the solver in y-direction")
    parser.add_argument("--dx", type=float, default=1.0,
                        help="Spatial step size of the solver in x-direction")
    parser.add_argument("--c", type=float, default=3.0,
                        help="Velocity of the wave (must satisfy CFL condition for numerical stability).")
    parser.add_argument("--wave-width-y", type=float, default=0.5,
                        help="Width of the wave in y-direction.")
    parser.add_argument("--wave-width-x", type=float, default=0.5,
                        help="Width of the wave in x-direction.")
    parser.add_argument("--amplitude", type=float, default=3.4,
                        help="Wave amplitude.")
    parser.add_argument("--damp", type=float, default=1.0,
                        help="Damp factor reducing the wave's energy over time.")
    parser.add_argument("--sequence-length", type=int, default=81,
                        help="Number of time steps in the solution")
    parser.add_argument("--skip-rate", type=int, default=1,
                        help="Take only every skip_rate's frame to simulate very fast waves beyond the c-parameter")
    parser.add_argument("--height", type=int, default=32,
                        help="Height of the simulation domain in pixels.")
    parser.add_argument("--width", type=int, default=32,
                        help="Width of the simulation domain in pixels.")
    parser.add_argument("--n-waves", type=int, default=1,
                        help="Number of wave initializations in a single simulation (not to be confused with the "\
                             "n-wamples argument).")
    parser.add_argument("--dataset-name", type=str, default="32x32_slow",
                        help="Name of the dataset (not to me confused with type of the dataset).")
    parser.add_argument("--dataset-type", type=str, default="train",
                        help="Type of the dataset, i.e, 'train', 'val', or 'test'.")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of samples that will be generated.")
    parser.add_argument("--visualize", action="store_true",
                        help="Whether to visualize the wave dynamics in an animation. When visualize=True, data will "\
                             "not be written to file.")
    run_args = parser.parse_args()

    dt = run_args.dt
    dy = run_args.dy
    dx = run_args.dx
    c = run_args.c
    amplitude = run_args.amplitude
    damp = run_args.damp
    height = run_args.height
    width = run_args.width
    wave_width_y = run_args.wave_width_y
    wave_width_x = run_args.wave_width_x

    dst_path = os.path.join("data", "numpy", run_args.dataset_name, run_args.dataset_type)
    velocity_ary = np.ones((run_args.width, run_args.height)) * run_args.c

    for file_no in tqdm.tqdm(range(run_args.n_samples), desc=f"Generating {run_args.dataset_type} samples"):

        # Calculate the number of simulation steps required to realize the chosen skip_rate
        simulation_steps = run_args.sequence_length*run_args.skip_rate

        # Initialize the wave field as two-dimensional zero-array
        field = np.zeros([simulation_steps, run_args.width, run_args.height])

        for wave in range(run_args.n_waves):
            # Generate a random point in the field where the impulse will be initialized
            start_pt = np.random.randint(0, run_args.width, 2)
            #start_pt = np.random.randint(run_args.width//2 - 8, run_args.width//2 + 8, 2)

            # Compute the initial field activity by applying a 2D gaussian around the start point
            for x in range(run_args.width):
                for y in range(run_args.height):
                    field[0, x, y] += f(
                        _x=x,
                        _y=y,
                        _varx=run_args.wave_width_x,
                        _vary=run_args.wave_width_y,
                        _a=run_args.amplitude
                    )

        # Iterate over all time steps to compute the activity at each position in the grid over all time steps
        for t in range(simulation_steps-1):

            # Iterate over all values in the field and update them
            for x in range(run_args.width):
                for y in range(run_args.height):
                    field[t+1, x, y] = u(_t=t, _x=x, _y=y)

            # Normalize the field activities to be at most 1 (or -1)
            # print(np.max(np.abs(field)))
            # field = field / np.max(np.abs(field))

        # Only take every skip_rate'th data point in time
        field = field[::run_args.skip_rate]

        if run_args.visualize:
            # plt.style.use("dark_background")
            # Plot the wave activity at one position
            fig, ax = plt.subplots(1, 1, figsize=[8, 2])
            ax.plot(range(len(field)), field[:, 5, 5])
            ax.set_xlabel("Time")
            ax.set_ylabel("Wave amplitude")
            ax.set_xlim([0, run_args.sequence_length-1])
            plt.tight_layout()
            plt.show()

            # Animate the overall wave
            fig, ax = plt.subplots(1, 1, figsize=[6, 6])
            im = ax.imshow(field[0], vmin=-run_args.amplitude/1.5, vmax=run_args.amplitude/1.5, cmap="Blues")
            anim = animation.FuncAnimation(fig,
                                           animate,
                                           frames=run_args.sequence_length,
                                           interval=200)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            print("Done.")
            exit()

        #
        # Write sample to file
        field = np.expand_dims(field, axis=1)  # Shape [T, C, H, W], where C=1

        # Check whether the directory to save the data exists and create it if not
        os.makedirs(dst_path, exist_ok=True)

        # Subselect only the 16x16 center field
        #half = width//2
        #dat_save = dat_save[:, half-8:half+8, half-8:half+8]

        # Write the data to file
        np.save(os.path.join(dst_path, run_args.dataset_type + "_" + str(file_no).zfill(5)), field)
