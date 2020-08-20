import numpy as np
from termcolor import colored

from .outputs import initialize_output, write_output
from .solvers import solve_Newton
from .spectral import applyBC, cheb


class Continuation:
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

    def Res(self, u, lmbda):
        """
        The residual to solve for the system
    """
        raise Exception("You need to implement this method in your child class!")

    def dRes_dlmbda(self, u, lmbda):
        """
        The residual derivative wrt the lambda parameter
    """
        raise Exception("You need to implement this method in your child class!")

    def Jac(self, u, lmbda):
        """
        The jacobian of the system
    """
        raise Exception("You need to implement this method in your child class!")

    def run(
        self,
        u0,
        lmbda0,
        step_size,
        max_steps,
        abs_tol=1.0e-08,
        rel_tol=1.0e-08,
        max_iters=200,
        filename=None,
        output_steps=False,
    ):
        """
        Pseudo arc length continuation
        INPUT
        u0: initial guess for the solution
        lmbda0: initial guess for the bifurcation parameter
        step_size0: initial arc-length step
        max_steps: the maximum steps to perform
        abs_tol: the absolute tolerance for the Newton solver
        rel_tol: the relative tolerance for the Newton solver
        max_iters: maximum number of iteration for the Newton solver
    """
        # INITIALIZATION
        lmbda = lmbda0  # bifurcation parameter
        k = 0  # iteration
        s = 0  # arc length
        n_saddle = 0
        n_hopf = 0
        output_fname, output_steps_fname = self.initial_step(
            u0, lmbda0, step_size, filename, output_steps
        )

        # # Check regularity and stability of new point
        # eigvals = np.linalg.eigvals(self.Jac(u, lmbda))
        # if np.amax(eigvals.real) == 0:
        #     print("Zero Eigenvalue found!")
        # stability = np.amax(eigvals.real) < 0
        # oscillation = eigvals[np.argmax(eigvals.real)].imag != 0.0
        # saddle = False
        # hopf = False

        # Iteration
        k += 1
        s += step_size
        ds = step_size

        # MAIN LOOP FOR ARC LENGTH
        while True:
            if k > max_steps:
                break

            newton_success = False
            # Loop in case solve fails
            while True:
                if newton_success:
                    break

                print()
                print(f"Step {k}, s = {s:.3e}, ds = {ds:.3e}")

                (
                    newton_success,
                    u,
                    lmbda,
                    du_ds,
                    dlmbda_ds,
                    stability,
                    oscillation,
                    saddle,
                    hopf,
                ) = self.step(ds, abs_tol, rel_tol, max_iters)

                # Cut step size
                if not newton_success:
                    ds /= 2.0

            # Save current calues
            self.u = u
            self.lmbda = lmbda
            self.du_ds = du_ds
            self.dlmbda_ds = dlmbda_ds
            self.stability = stability

            # Output results
            write_output(
                k,
                output_fname,
                output_steps_fname,
                self.x,
                self.u,
                self.lmbda,
                self.nvar,
                stability,
                oscillation,
                saddle,
                hopf,
            )

            # Iteration
            k += 1
            s += ds
            if saddle:
                n_saddle += 1
            if hopf:
                n_hopf += 1

        print()
        print(colored("Arc-length continuation complete!", "green"))
        print(
            "Detected {} saddle points and {} Hopf bifurcation points.".format(
                n_saddle, n_hopf
            )
        )
        return None

    def tangent_predictor(self, u, lmbda, ds):
        """
    """
        # du/dlmbda
        delta = 1.0e-08
        eps1 = delta * (np.abs(lmbda) + delta)
        dRes_dlmbda = (self.Res(u, lmbda + eps1) - self.Res(u, lmbda)) / eps1
        du_dlmbda = np.linalg.solve(self.Jac(u, lmbda), -dRes_dlmbda)

        # dlmbda/ds
        if np.array_equal(u, self.u) and (lmbda == self.lmbda):  # First step
            r = 1.0 if ds > 0.0 else -1.0
        else:
            r = np.dot(du_dlmbda, u - self.u) + (lmbda - self.lmbda)
        dlmbda_ds = np.divide(
            r / np.abs(r), np.sqrt(1.0 + np.dot(du_dlmbda, du_dlmbda))
        )

        # du/ds
        du_ds = du_dlmbda * dlmbda_ds

        return du_ds, dlmbda_ds

    def initial_step(self, u0, lmbda, ds, filename, output_steps):
        """
        Calculate trivial solution with initial bifurcation parameter
        INPUT
        u0: initial guess of the solution
        lmbda: initial bifurcation parameter
        filename: base file name for continuation output
        output_steps: boolean to specify whether each step should be outputted
        OUTPUT
        u: initial solution
        output_fname: output file name (with path) for continuation output
    """
        k0 = 0
        s0 = 0
        ds0 = 0
        print()
        print(f"Step {k0}, s = {s0:.3e}, ds = {ds0:.3e}")

        # Get number of variables
        if len(u0) != (self.nvar * (self.N + 1)):
            raise Exception(
                "Size of the initial guess",
                len(u0),
                "does not match grid resolution and number of variables. Size should be ",
                self.nvar * (self.N + 1),
            )
        u0_vars = np.split(u0, self.nvar)
        for i in range(self.nvar):
            u0_vars[i] = applyBC(u0_vars[i])
        u0 = np.hstack(u0_vars)
        # Get initial solution
        u, _ = solve_Newton(
            lambda u: self.Res(u, lmbda), lambda u: self.Jac(u, lmbda), u0
        )

        # Save in class
        self.lmbda = lmbda
        self.u = u
        self.stability = True

        # Update tangents
        self.du_ds, self.dlmbda_ds = self.tangent_predictor(u, lmbda, ds)

        # Continuation output: initial results
        if self.nvar > 1:
            headers_vars = ["u" + str(int(k)) + "_norm" for k in range(self.nvar)]
            headers_output = [
                "lambda",
                "stability",
                "oscillation",
                "saddle",
                "hopf",
            ]
            headers_output[1:1] = headers_vars
        else:
            headers_output = [
                "lambda",
                "u_norm",
                "stability",
                "oscillation",
                "saddle",
                "hopf",
            ]
        output_fname, output_steps_fname = initialize_output(
            filename, headers_output, output_steps
        )
        write_output(
            0,
            output_fname,
            output_steps_fname,
            self.x,
            self.u,
            self.lmbda,
            self.nvar,
            True,
            False,
            False,
            False,
        )
        print("lmbda = {}, ||x|| = {}".format(self.lmbda, np.linalg.norm(self.u)))

        return output_fname, output_steps_fname

    def step(self, ds, abs_tol, rel_tol, max_iters):
        """
    """
        # Tangent predictor for next values
        u = self.u + self.du_ds * ds
        lmbda = self.lmbda + self.dlmbda_ds * ds

        # Arc length continuation
        # New values of u and lmbda
        u, lmbda, newton_success = self.arc_length_continuation(
            u, lmbda, ds, abs_tol, rel_tol, max_iters,
        )

        # Check regularity and stability of new point
        eigvals, eigvecs = np.linalg.eig(self.Jac(u, lmbda))
        stability = np.amax(eigvals.real) < 0
        oscillation = eigvals[np.argmax(eigvals.real)].imag != 0.0
        saddle = False
        hopf = False
        # Update tangents
        du_ds, dlmbda_ds = self.tangent_predictor(u, lmbda, ds)

        # Check Hopf point
        if self.stability != stability and oscillation:
            print(colored("Hopf point tracking algorithm!", "red"))
            u, lmbda, newton_success = self.hopf_point_locator(eigvals, eigvecs)

            # stability = False
            oscillation = True
            saddle = False
            hopf = True

            # # Update tangents
            # du_ds, dlmbda_ds = self.tangent_predictor(u, lmbda, ds)

        # Check saddle point
        elif self.dlmbda_ds * dlmbda_ds < 0.0:
            print(colored("Saddle point tracking algorithm!", "red"))
            u, lmbda, newton_success = self.saddle_point_locator()

            stability = False
            oscillation = False
            saddle = True
            hopf = False

            # # Update tangents
            # du_ds, dlmbda_ds = self.tangent_predictor(u, lmbda, ds)

        return (
            newton_success,
            u,
            lmbda,
            du_ds,
            dlmbda_ds,
            stability,
            oscillation,
            saddle,
            hopf,
        )

    def arc_length_continuation(
        self, u, lmbda, ds, abs_tol, rel_tol, max_iters,
    ):
        """
        Arc length continuation. Implementation based on:
        http://www.cs.sandia.gov/LOCA/loca1.1_book.pdf
        INPUT

        OUTPUT
    """
        # Iterations
        num_newton_steps = 0
        newton_success = False

        # Residuals and Jacobian
        J = self.Jac(u, lmbda)
        Ru = self.Res(u, lmbda)
        Rlmbda = (
            np.dot(self.du_ds, u - self.u) + (lmbda - self.lmbda) * self.dlmbda_ds - ds
        )

        Rnrm = np.sqrt(np.linalg.norm(Ru) ** 2 + Rlmbda ** 2)
        Rnrm0 = Rnrm

        print(f"{num_newton_steps} Nonlinear |R| = ", colored(f"{Rnrm:.6e}", "green"))

        while num_newton_steps < max_iters:
            # Convergence criteria
            if Rnrm < abs_tol or np.divide(Rnrm, Rnrm0) < rel_tol:
                if Rnrm < abs_tol:
                    reason = "ABS_TOL"
                else:
                    reason = "REL_TOL"
                print(
                    f"Solve converged due to {reason} after {num_newton_steps} iterations."
                )
                print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(u)))
                print(colored("Solve Converged!", "green"))
                newton_success = True
                return u, lmbda, newton_success

            # Update solution
            # Solve
            #
            #  (J,     dR/dlmbda) (du    )  =  -(Ru)
            #  (du/ds, dlmbda/ds) (dlmbda)      (Rlmbda)
            #

            # dR/dlmbda
            delta = 1.0e-06
            eps1 = delta * (np.abs(lmbda) + delta)
            Resb = (self.Res(u, lmbda + eps1) - Ru) / eps1

            # Temporary vectors
            a = np.linalg.solve(J, -Ru)
            b = np.linalg.solve(J, -Resb)

            # Update values
            dlmbda = -np.divide(
                Rlmbda + np.dot(self.du_ds, a), self.dlmbda_ds + np.dot(self.du_ds, b)
            )
            du = a + dlmbda * b

            u += du
            lmbda += dlmbda
            num_newton_steps += 1

            # Update residuals
            Ru = self.Res(u, lmbda)
            Rlmbda = (
                np.dot(self.du_ds, u - self.u)
                + (lmbda - self.lmbda) * self.dlmbda_ds
                - ds
            )
            # Update jacobien
            J = self.Jac(u, lmbda)

            Rnrm_old = Rnrm
            Rnrm = np.sqrt(np.linalg.norm(Ru) ** 2 + Rlmbda ** 2)
            if Rnrm < 0.8 * Rnrm_old:
                color = "green"
            elif Rnrm > Rnrm_old:
                color = "red"
            else:
                color = "yellow"

            print(f"{num_newton_steps} Nonlinear |R| = ", colored(f"{Rnrm:.6e}", color))

        # Here, no solutions have been found after max_iters iterations
        print("Solve diverged due to MAX_ITER.")
        print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(u)))
        print(colored("Solve did NOT converge!", "red"))
        return u, lmbda, newton_success

    def saddle_point_locator(self, abs_tol=1.0e-04, rel_tol=1.0e-04, max_iters=1000):
        # Initialization
        u = self.u.copy()
        lmbda = self.lmbda.copy()

        # Iterations
        num_newton_steps = 0
        newton_success = False

        # Residuals and Jacobian
        J = self.Jac(u, lmbda)
        Ru = self.Res(u, lmbda)
        # dR/dlmbda
        delta = 1.0e-06
        eps1 = delta * (np.abs(lmbda) + delta)
        Resb = (self.Res(u, lmbda + eps1) - Ru) / eps1
        b = np.linalg.solve(J, -Resb)
        v = b / np.linalg.norm(b)
        phi = b / np.linalg.norm(b)
        Rv = np.dot(J, v)
        Rlmbda = np.dot(phi, v) - 1.0

        Rnrm = np.sqrt(np.linalg.norm(Ru) ** 2 + np.linalg.norm(Rv) ** 2 + Rlmbda ** 2)
        Rnrm0 = Rnrm

        print(f"{num_newton_steps} Nonlinear |R| = ", colored(f"{Rnrm:.6e}", "green"))

        while num_newton_steps < max_iters:
            # Convergence criteria
            if Rnrm < abs_tol or np.divide(Rnrm, Rnrm0) < rel_tol:
                if Rnrm < abs_tol:
                    reason = "ABS_TOL"
                else:
                    reason = "REL_TOL"
                print(
                    f"Solve converged due to {reason} after {num_newton_steps} iterations."
                )
                print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(u)))
                print(colored("Solve Converged!", "green"))
                newton_success = True
                return u, lmbda, newton_success

            # Update solution
            # Solve
            #
            #  (J,      0,   dR/dlmbda )  (du)       -(Ru)
            #  (dJv/dx, J,   dJv/dlmbda)  (dv)     = -(Jv)
            #  (0,      phi, 0         )  (dlmbda)   -(phi v - 1)
            #

            # Temporary vectors
            # Estimate jacobian with 1st order finite differences
            delta = 1.0e-06
            a = np.linalg.solve(J, -Ru)

            eps1 = delta * (np.abs(lmbda) + delta)
            Resb = (self.Res(u, lmbda + eps1) - Ru) / eps1
            b = np.linalg.solve(J, -Resb)

            eps2 = delta * (np.linalg.norm(u) / np.linalg.norm(a) + delta)
            Resc = (np.dot(self.Jac(u + eps2 * a, lmbda), v) - np.dot(J, v)) / eps2
            c = np.linalg.solve(J, -Resc)

            eps3 = delta * (np.linalg.norm(u) / np.linalg.norm(b) + delta)
            Resd = (
                np.dot(self.Jac(u + eps3 * b, lmbda), v) / eps3
                + np.dot(self.Jac(u, lmbda + eps1), v) / eps1
                - np.dot(J, v) * (1.0 / eps3 + 1.0 / eps1)
            )
            d = np.linalg.solve(J, -Resd)

            # Update values
            dlmbda = (1.0 - np.dot(phi, c)) / np.dot(phi, d)
            du = a + dlmbda * b
            dv = c + dlmbda * d - v

            u += du
            v += dv
            lmbda += dlmbda
            num_newton_steps += 1

            # Update residuals and Jacobian
            J = self.Jac(u, lmbda)
            Ru = self.Res(u, lmbda)
            Rv = np.dot(J, v)
            Rlmbda = np.dot(phi, v) - 1.0

            Rnrm_old = Rnrm
            Rnrm = np.sqrt(
                np.linalg.norm(Ru) ** 2 + np.linalg.norm(Rv) ** 2 + Rlmbda ** 2
            )
            if Rnrm < 0.8 * Rnrm_old:
                color = "green"
            elif Rnrm > Rnrm_old:
                color = "red"
            else:
                color = "yellow"

            print(f"{num_newton_steps} Nonlinear |R| = ", colored(f"{Rnrm:.6e}", color))

        # Here, no solutions have been found after max_iters iterations
        print("Solve diverged due to MAX_ITER.")
        print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(u)))
        print(colored("Solve did NOT converge!", "red"))
        return u, lmbda, newton_success

    def hopf_point_locator(
        self,
        eigenvals_current,
        eigenvecs_current,
        abs_tol=1.0e-04,
        rel_tol=1.0e-04,
        max_iters=1000,
    ):
        # Initialization
        u = self.u.copy()
        lmbda = self.lmbda.copy()

        # Iterations
        num_newton_steps = 0
        newton_success = False

        # Residuals and Jacobian
        J = self.Jac(u, lmbda)
        ind = np.argmax(eigenvals_current.real)
        Ru = self.Res(u, lmbda)
        # dR/dlmbda
        delta = 1.0e-06
        eps1 = delta * (np.abs(lmbda) + delta)
        Resb = (self.Res(u, lmbda + eps1) - Ru) / eps1
        b = np.linalg.solve(J, -Resb)
        v = eigenvecs_current[:, ind].real
        phi = b / np.linalg.norm(b)
        w = eigenvecs_current[:, ind].imag
        omega = eigenvals_current[ind].imag
        Rv = (
            np.dot(J, v) + omega * w
        )  # here I assume that B is the identity matrix (neglect thermal pressurization)
        Rw = (
            np.dot(J, w) - omega * v
        )  # here I assume that B is the identity matrix (neglect thermal pressurization)
        Romega = np.dot(phi, v) - 1.0
        Rlmbda = np.dot(phi, w)

        Rnrm = np.sqrt(
            np.linalg.norm(Ru) ** 2
            + np.linalg.norm(Rv) ** 2
            + np.linalg.norm(Rw) ** 2
            + Romega ** 2
            + Rlmbda ** 2
        )
        Rnrm0 = Rnrm

        print(f"{num_newton_steps} Nonlinear |R| = ", colored(f"{Rnrm:.6e}", "green"))

        while num_newton_steps < max_iters:
            # Convergence criteria
            if Rnrm < abs_tol or np.divide(Rnrm, Rnrm0) < rel_tol:
                if Rnrm < abs_tol:
                    reason = "ABS_TOL"
                else:
                    reason = "REL_TOL"
                print(
                    f"Solve converged due to {reason} after {num_newton_steps} iterations."
                )
                print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(u)))
                print(colored("Solve Converged!", "green"))
                newton_success = True
                return u, lmbda, newton_success

            # Update solution
            # Solve
            #
            #  (J,      0,        0,       0,    dR/dlmbda )  (du)       -(Ru)
            #  (dJv/du, J,        omega B, B w,  dJv/dlmbda)  (dv)       -(Jv + omega B w)
            #  (dJw/du, -omega B, J,       -B v, dJw/dlmbda)  (dw)     = -(Jw - omega B v)
            #  (0,      phi,      0,       0,    0         )  (domega)   -(phi v - 1)
            #  (0,      0,        phi,     0,    0         )  (dlmbda)   -(phi w)
            #

            # Temporary vectors
            # Estimate jacobian with 1st order finite differences
            delta = 1.0e-06
            a = np.linalg.solve(J, -Ru)

            eps1 = delta * (np.abs(lmbda) + delta)
            Resb = (self.Res(u, lmbda + eps1) - Ru) / eps1
            b = np.linalg.solve(J, -Resb)

            J_hopf = np.block(
                [[J, omega * np.identity(Ru.size)], [-omega * np.identity(Ru.size), J]]
            )
            Rescd = np.concatenate((-w, v))
            sol_cd = np.linalg.solve(J_hopf, -Rescd)
            sol_cd = np.split(sol_cd, 2)
            c = sol_cd[0]
            d = sol_cd[1]

            eps2 = delta * (np.linalg.norm(u) / np.linalg.norm(a) + delta)
            Rese = (np.dot(self.Jac(u + eps2 * a, lmbda), v) - np.dot(J, v)) / eps2
            Resf = (np.dot(self.Jac(u + eps2 * a, lmbda), w) - np.dot(J, w)) / eps2
            Resef = np.concatenate((Rese, Resf))
            sol_ef = np.linalg.solve(J_hopf, -Resef)
            sol_ef = np.split(sol_ef, 2)
            e = sol_ef[0]
            f = sol_ef[1]

            eps3 = delta * (np.linalg.norm(u) / np.linalg.norm(b) + delta)
            Resg = (
                np.dot(self.Jac(u + eps3 * b, lmbda), v) / eps3
                + np.dot(self.Jac(u, lmbda + eps1), v) / eps1
                - np.dot(J, v) * (1.0 / eps3 + 1.0 / eps1)
            )
            Resh = (
                np.dot(self.Jac(u + eps3 * b, lmbda), w) / eps3
                + np.dot(self.Jac(u, lmbda + eps1), w) / eps1
                - np.dot(J, w) * (1.0 / eps3 + 1.0 / eps1)
            )
            Resgh = np.concatenate((Resg, Resh))
            sol_gh = np.linalg.solve(J_hopf, -Resgh)
            sol_gh = np.split(sol_gh, 2)
            g = sol_gh[0]
            h = sol_gh[1]

            # Update values
            dlmbda = (
                np.dot(phi, c) * np.dot(phi, f)
                - np.dot(phi, e) * np.dot(phi, d)
                + np.dot(phi, d)
            ) / (np.dot(phi, d) * np.dot(phi, g) - np.dot(phi, c) * np.dot(phi, h))
            domega = (dlmbda * np.dot(phi, h) + np.dot(phi, f)) / np.dot(phi, d)
            du = a + dlmbda * b
            dv = e + dlmbda * g - domega * c - v
            dw = f + dlmbda * h - domega * d - w

            u += du
            v += dv
            w += dw
            omega += domega
            lmbda += dlmbda
            num_newton_steps += 1

            # Update residuals and jacobian
            J = self.Jac(u, lmbda)
            Ru = self.Res(u, lmbda)
            Rv = (
                np.dot(J, v) + omega * w
            )  # here I assume that B is the identity matrix (neglect thermal pressurization)
            Rw = (
                np.dot(J, w) - omega * v
            )  # here I assume that B is the identity matrix (neglect thermal pressurization)
            Romega = np.dot(phi, v) - 1.0
            Rlmbda = np.dot(phi, w)

            Rnrm_old = Rnrm
            Rnrm = np.sqrt(
                np.linalg.norm(Ru) ** 2
                + np.linalg.norm(Rv) ** 2
                + np.linalg.norm(Rw) ** 2
                + Romega ** 2
                + Rlmbda ** 2
            )
            if Rnrm < 0.8 * Rnrm_old:
                color = "green"
            elif Rnrm > Rnrm_old:
                color = "red"
            else:
                color = "yellow"

            print(f"{num_newton_steps} Nonlinear |R| = ", colored(f"{Rnrm:.6e}", color))

        # Here, no solutions have been found after max_iters iterations
        print("Solve diverged due to MAX_ITER.")
        print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(u)))
        print(colored("Solve did NOT converge!", "red"))
        return u, lmbda, newton_success
