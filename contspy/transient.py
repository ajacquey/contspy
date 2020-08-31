import numpy as np
from termcolor import colored

from .outputs import initialize_output, write_trans_output
from .solvers import solve_Newton
from .spectral import applyBC, cheb


class Transient:
    def __init__(self, L=1.0, N=20, nvar=1, output_file_base=None):
        # Settings for spectral elements method
        self.L = L  # lengh of the domain
        self.N = N  # N must be even
        self.D, self.D2, self.x = cheb(self.N, self.L)
        # Boundary conditions
        self.D = applyBC(self.D)
        self.D2 = applyBC(self.D2)
        # Number of variables
        self.nvar = nvar

    def Res(self, u):
        """
        The residual to solve for the system
        """
        raise Exception("You need to implement this method in your child class!")

    def Jac(self, u):
        """
        The jacobian of the system
        """
        raise Exception("You need to implement this method in your child class!")

    def run(
        self,
        u0,
        step_size,
        max_steps,
        abs_tol=1.0e-08,
        rel_tol=1.0e-08,
        max_iters=200,
        filename=None,
        output_steps=False,
    ):
        """
        Transient solve
        INPUT
        u0: initial conditions for the solution
        step_size: initial time step
        max_steps: the maximum time steps to perform
        abs_tol: the absolute tolerance for the Newton solver
        rel_tol: the relative tolerance for the Newton solver
        max_iters: maximum number of iteration for the Newton solver
        """
        # INITIALIZATION
        k = 0  # iteration
        t = 0  # time
        output_fname, output_steps_fname = self.initial_step(
            u0, step_size, filename, output_steps
        )

        # Iteration
        k += 1
        t += step_size
        dt = step_size

        # MAIN LOOP FOR TIME
        while True:
            if k > max_steps:
                break

            newton_success = False
            # Loop in case solve fails
            while True:
                if newton_success:
                    break

                print()
                print(f"Time step {k}, time = {t:.3e}, dt = {dt:.3e}")

                u = self.u.copy()

                u, newton_success = solve_Newton(
                    lambda u: self.Res(u), lambda u: self.Jac(u), u
                )

                # Cut step size
                if not newton_success:
                    dt /= 2.0
                    self.dt = dt

            # Save current calues
            self.u = u
            self.time = t
            self.dt = dt

            # Output results
            write_trans_output(
                k,
                output_fname,
                output_steps_fname,
                self.x,
                self.u,
                self.time,
                self.nvar,
            )

            # Iteration
            k += 1
            t += dt

        print()
        print(colored("Transient simulation complete!", "green"))

        return None

    def initial_step(self, u0, dt, filename, output_steps):
        """
        Initialize transient solve
        INPUT
        u0: initial conditions of the solution
        dt: initial time step size
        filename: base file name for transient output
        output_steps: boolean to specify whether each step should be outputted
        OUTPUT
        output_fname: output file name (with path) for transient output
        """
        k0 = 0
        t0 = 0
        dt0 = 0
        print()
        print(f"Time step {k0}, time = {t0:.3e}, dt = {dt0:.3e}")

        # Get number of variables
        if len(u0) != (self.nvar * (self.N + 1)):
            raise Exception(
                "Size of the initial condition",
                len(u0),
                "does not match grid resolution and number of variables. Size should be ",
                self.nvar * (self.N + 1),
            )
        u0_vars = np.split(u0, self.nvar)
        for i in range(self.nvar):
            u0_vars[i] = applyBC(u0_vars[i])
        u0 = np.hstack(u0_vars)

        # Save in class
        self.u = u0
        self.time = 0.0
        self.dt = dt

        # Transient output: initial results
        if self.nvar > 1:
            headers_vars = ["u" + str(int(k)) + "_norm" for k in range(self.nvar)]
            headers_output = ["time"]
            headers_output[1:1] = headers_vars
        else:
            headers_output = ["time", "u_norm"]
        output_fname, output_steps_fname = initialize_output(
            filename, headers_output, output_steps
        )
        write_trans_output(
            0, output_fname, output_steps_fname, self.x, self.u, self.time, self.nvar
        )
        return output_fname, output_steps_fname
