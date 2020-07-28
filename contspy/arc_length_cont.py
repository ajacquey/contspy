import numpy as np
from termcolor import colored

from .solvers import solve_Newton


def pseudo_arclength_cont(
    problem,
    x0,
    lmbda0,
    step_size0,
    max_steps,
    callback,
    newton_abs_tol=1.0e-10,
    newton_rel_tol=1.0e-10,
    newton_max_iters=200,
):
    """
    Pseudo arc length continuation
    INPUT
    problem: the problem class
    x0: initial guess for the solution
    lmbda0: initial guess for the bifurcation parameter
    step_size0: initial arc-length step
    max_steps: the maximum steps to perform
    callback: custom function to call after each step
  """

    # Initialize
    lmbda = lmbda0
    k = 0
    s = 0
    lmbda_turning = []
    lmbda_hopf = []

    # Get initial solution
    x, _ = solve_Newton(
        lambda x: problem.Res(x, lmbda), lambda x: problem.Jac(x, lmbda), x0
    )

    # Arclength step size
    ds = step_size0

    # Tangent predictor for first step
    dx_dlmbda = np.linalg.solve(problem.Jac(x, lmbda), -problem.dRes_dlmbda(x, lmbda))
    dlmbda_ds = 1.0 if step_size0 > 0.0 else -1.0
    dx_ds = dx_dlmbda * dlmbda_ds

    # Correct with norm??
    nrm = np.sqrt(np.dot(dx_ds, dx_ds) + dlmbda_ds ** 2)
    dx_ds /= nrm
    dlmbda_ds /= nrm

    # Check regularity and stability of new point
    eigvals = np.linalg.eigvals(problem.Jac(x, lmbda))
    if np.amax(eigvals.real) == 0:
        print("Zero Eigenvalue found!")
    stability = np.amax(eigvals.real) < 0
    oscillation = eigvals[np.argmax(eigvals.real)].imag != 0.0

    # Save current values
    x_current = x
    lmbda_current = lmbda
    dx_ds_current = dx_ds
    dlmbda_ds_current = dlmbda_ds
    stability_current = stability

    # Function to plot or write data
    callback(k, lmbda, x, stability, oscillation)

    # Iterations
    k = 1
    s = ds

    while True:
        if k > max_steps:
            break

        print()
        print(f"Step {k}, s = {s:.3e}, ds = {ds:.3e}")

        # Predictor
        x = x_current + dx_ds_current * ds
        lmbda = lmbda_current + dlmbda_ds_current * ds

        # Newton corrector
        # New values of x and lmbda
        x, lmbda, newton_success = Newton_corrector(
            problem,
            x,
            lmbda,
            x_current,
            lmbda_current,
            dx_ds,
            dlmbda_ds,
            ds,
            abs_tol=newton_abs_tol,
            rel_tol=newton_rel_tol,
        )

        # Check regularity and stability of new point
        eigvals, eigvecs = np.linalg.eig(problem.Jac(x, lmbda))
        stability = np.amax(eigvals.real) < 0
        oscillation = eigvals[np.argmax(eigvals.real)].imag != 0.0
        # print(np.amax(eigvals.real))
        # print(eigvals[np.argmax(eigvals.real)].imag)

        # Check if we passed a bifurcation point
        if stability != stability_current:
            if ~oscillation:
                print(colored("Turning point tracking algorithm!", "red"))
                x, lmbda, newton_success = turning_point_locator(
                    problem, x_current, lmbda_current
                )
                lmbda_turning.append(lmbda)
            else:
                print(colored("Hopf point tracking algorithm!", "red"))
                x, lmbda, newton_success = hopf_point_locator(
                    problem, x_current, lmbda_current, eigvals, eigvecs
                )
                lmbda_hopf.append(lmbda)

        # Approximate dlmbda/ds and du/ds for the next predictor step
        dx_dlmbda = np.linalg.solve(
            problem.Jac(x, lmbda), -problem.dRes_dlmbda(x, lmbda)
        )
        # Make sure the sign of dlambda_ds is correct
        r = np.dot(dx_dlmbda, x - x_current) + (lmbda - lmbda_current)
        dlmbda_ds = 1.0 if r > 0.0 else -1.0
        dx_ds = dx_dlmbda * dlmbda_ds

        # Correct with norm??
        nrm = np.sqrt(np.dot(dx_ds, dx_ds) + dlmbda_ds ** 2)
        dx_ds /= nrm
        dlmbda_ds /= nrm

        # Save current calues
        x_current = x
        lmbda_current = lmbda
        dx_ds_current = dx_ds
        dlmbda_ds_current = dlmbda_ds
        stability_current = stability

        # Function to plot or write data
        callback(k, lmbda, x, stability, oscillation)

        # Iteration
        k += 1
        s += ds

    print()
    print(colored("Arc-length continuation complete!", "green"))
    print(
        "Detected {} turning points and {} Hopf bifurcation points.".format(
            len(lmbda_turning), len(lmbda_hopf)
        )
    )
    return np.array(lmbda_turning), np.array(lmbda_hopf)


def Newton_corrector(
    problem,
    x,
    lmbda,
    x_current,
    lmbda_current,
    dx_ds,
    dlmbda_ds,
    ds,
    abs_tol=1.0e-10,
    rel_tol=1.0e-10,
    max_iters=1000,
):

    # Iterations
    num_newton_steps = 0
    newton_success = False

    # Residuals
    Rx = problem.Res(x, lmbda)
    n = np.dot(x - x_current, dx_ds) + (lmbda - lmbda_current) * dlmbda_ds - ds

    Rnrm = np.sqrt(np.linalg.norm(Rx) ** 2 + n ** 2)
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
            print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(x)))
            print(colored("Solve Converged!", "green"))
            newton_success = True
            return x, lmbda, newton_success

        # Update solution
        # Solve
        #
        #  (J,     dR/dlmbda) (dx    )  =  -(R)
        #  (dx/ds, dlmbda/ds) (dlmbda)      (n)
        #
        # Temporary vectors
        a = np.linalg.solve(problem.Jac(x, lmbda), -problem.Res(x, lmbda))
        b = np.linalg.solve(problem.Jac(x, lmbda), -problem.dRes_dlmbda(x, lmbda))

        # Update values
        dlmbda = -np.divide(n + np.dot(dx_ds, a), dlmbda_ds + np.dot(dx_ds, b))
        dx = a + dlmbda * b

        x += dx
        lmbda += dlmbda
        num_newton_steps += 1

        Rx = problem.Res(x, lmbda)
        n = np.dot(x - x_current, dx_ds) + (lmbda - lmbda_current) * dlmbda_ds - ds

        Rnrm_old = Rnrm
        Rnrm = np.sqrt(np.linalg.norm(Rx, ord=2) ** 2 + n ** 2)
        if Rnrm < 0.8 * Rnrm_old:
            color = "green"
        elif Rnrm > Rnrm_old:
            color = "red"
        else:
            color = "yellow"

        print(f"{num_newton_steps} Nonlinear |R| = ", colored(f"{Rnrm:.6e}", color))

    # Here, no solutions have been found after max_iters iterations
    print("Solve diverged due to MAX_ITER.")
    print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(x)))
    print(colored("Solve did NOT converge!", "red"))
    return x, lmbda, newton_success


def turning_point_locator(
    problem, x_current, lmbda_current, abs_tol=1.0e-04, rel_tol=1.0e-04, max_iters=1000
):
    # Initialization
    x = x_current.copy()
    lmbda = lmbda_current.copy()

    # Iterations
    num_newton_steps = 0
    newton_success = False

    # Residuals and Jacobian
    J = problem.Jac(x, lmbda)
    Rx = problem.Res(x, lmbda)
    b = np.linalg.solve(problem.Jac(x, lmbda), -problem.dRes_dlmbda(x, lmbda))
    y = b / np.linalg.norm(b)
    phi = b / np.linalg.norm(b)
    Ry = np.dot(problem.Jac(x, lmbda), y)
    Rlmbda = np.dot(phi, y) - 1.0

    Rnrm = np.sqrt(np.linalg.norm(Rx) ** 2 + np.linalg.norm(Ry) ** 2 + Rlmbda ** 2)
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
            print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(x)))
            print(colored("Solve Converged!", "green"))
            newton_success = True
            return x, lmbda, newton_success

        # Update solution
        # Solve
        #
        #  (J,      0,   dR/dlmbda )  (dx)       -(R)
        #  (dJy/dx, J,   dJy/dlmbda)  (dy)     = -(Jy)
        #  (0,      phi, 0         )  (dlmbda)   -(phi y - 1)
        #

        # Temporary vectors
        # Estimate jacobian with 1st order finite differences
        delta = 1.0e-06
        a = np.linalg.solve(J, -Rx)

        eps1 = delta * (np.abs(lmbda) + delta)
        Resb = (problem.Res(x, lmbda + eps1) - Rx) / eps1
        b = np.linalg.solve(J, -Resb)

        eps2 = delta * (np.linalg.norm(x) / np.linalg.norm(a) + delta)
        Resc = (np.dot(problem.Jac(x + eps2 * a, lmbda), y) - np.dot(J, y)) / eps2
        c = np.linalg.solve(J, -Resc)

        eps3 = delta * (np.linalg.norm(x) / np.linalg.norm(b) + delta)
        Resd = (
            np.dot(problem.Jac(x + eps3 * b, lmbda), y) / eps3
            + np.dot(problem.Jac(x, lmbda + eps1), y) / eps1
            - np.dot(J, y) * (1.0 / eps3 + 1.0 / eps1)
        )
        d = np.linalg.solve(J, -Resd)

        # Update values
        dlmbda = (1.0 - np.dot(phi, c)) / np.dot(phi, d)
        dx = a + dlmbda * b
        dy = c + dlmbda * d - y

        x += dx
        y += dy
        lmbda += dlmbda
        num_newton_steps += 1

        # Update residuals and Jacobian
        J = problem.Jac(x, lmbda)
        Rx = problem.Res(x, lmbda)
        Ry = np.dot(J, y)
        Rlmbda = np.dot(phi, y) - 1.0

        Rnrm_old = Rnrm
        Rnrm = np.sqrt(np.linalg.norm(Rx) ** 2 + np.linalg.norm(Ry) ** 2 + Rlmbda ** 2)
        if Rnrm < 0.8 * Rnrm_old:
            color = "green"
        elif Rnrm > Rnrm_old:
            color = "red"
        else:
            color = "yellow"

        print(f"{num_newton_steps} Nonlinear |R| = ", colored(f"{Rnrm:.6e}", color))

    # Here, no solutions have been found after max_iters iterations
    print("Solve diverged due to MAX_ITER.")
    print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(x)))
    print(colored("Solve did NOT converge!", "red"))
    return x, lmbda, newton_success


def hopf_point_locator(
    problem,
    x_current,
    lmbda_current,
    eigenvals_current,
    eigenvecs_current,
    abs_tol=1.0e-04,
    rel_tol=1.0e-04,
    max_iters=1000,
):
    # Initialization
    x = x_current.copy()
    lmbda = lmbda_current.copy()

    # Iterations
    num_newton_steps = 0
    newton_success = False

    # Residuals and Jacobian
    J = problem.Jac(x, lmbda)
    ind = np.argmax(eigenvals_current.real)
    Rx = problem.Res(x, lmbda)
    b = np.linalg.solve(problem.Jac(x, lmbda), -problem.dRes_dlmbda(x, lmbda))
    y = eigenvecs_current[:, ind].real
    phi = b / np.linalg.norm(b)
    z = eigenvecs_current[:, ind].imag
    omega = eigenvals_current[ind].imag
    Ry = (
        np.dot(problem.Jac(x, lmbda), y) + omega * z
    )  # here I assume that B is the identity matrix (neglect thermal pressurization)
    Rz = (
        np.dot(problem.Jac(x, lmbda), z) - omega * y
    )  # here I assume that B is the identity matrix (neglect thermal pressurization)
    Romega = np.dot(phi, y) - 1.0
    Rlmbda = np.dot(phi, z)

    Rnrm = np.sqrt(
        np.linalg.norm(Rx) ** 2
        + np.linalg.norm(Ry) ** 2
        + np.linalg.norm(Rz) ** 2
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
            print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(x)))
            print(colored("Solve Converged!", "green"))
            newton_success = True
            return x, lmbda, newton_success

        # Update solution
        # Solve
        #
        #  (J,      0,        0,       0,    dR/dlmbda )  (dx)       -(R)
        #  (dJy/dx, J,        omega B, B z,  dJy/dlmbda)  (dy)       -(Jy + omega B z)
        #  (dJz/dx, -omega B, J,       -B y, dJz/dlmbda)  (dz)     = -(Jz - omega B y)
        #  (0,      phi,      0,       0,    0         )  (domega)   -(phi y - 1)
        #  (0,      0,        phi,     0,    0         )  (dlmbda)   -(phi z)
        #

        # Temporary vectors
        # Estimate jacobian with 1st order finite differences
        delta = 1.0e-06
        a = np.linalg.solve(J, -Rx)

        eps1 = delta * (np.abs(lmbda) + delta)
        Resb = (problem.Res(x, lmbda + eps1) - Rx) / eps1
        b = np.linalg.solve(J, -Resb)

        J_hopf = np.block(
            [[J, omega * np.identity(Rx.size)], [-omega * np.identity(Rx.size), J]]
        )
        Rescd = np.concatenate((-z, y))
        sol_cd = np.linalg.solve(J_hopf, -Rescd)
        sol_cd = np.split(sol_cd, 2)
        c = sol_cd[0]
        d = sol_cd[1]

        eps2 = delta * (np.linalg.norm(x) / np.linalg.norm(a) + delta)
        Rese = (
            np.dot(problem.Jac(x + eps2 * a, lmbda), y)
            - np.dot(problem.Jac(x, lmbda), y)
        ) / eps2
        Resf = (
            np.dot(problem.Jac(x + eps2 * a, lmbda), z)
            - np.dot(problem.Jac(x, lmbda), z)
        ) / eps2
        Resef = np.concatenate((Rese, Resf))
        sol_ef = np.linalg.solve(J_hopf, -Resef)
        sol_ef = np.split(sol_ef, 2)
        e = sol_ef[0]
        f = sol_ef[1]

        eps3 = delta * (np.linalg.norm(x) / np.linalg.norm(b) + delta)
        Resg = (
            np.dot(problem.Jac(x + eps3 * b, lmbda), y) / eps3
            + np.dot(problem.Jac(x, lmbda + eps1), y) / eps1
            - np.dot(J, y) * (1.0 / eps3 + 1.0 / eps1)
        )
        Resh = (
            np.dot(problem.Jac(x + eps3 * b, lmbda), z) / eps3
            + np.dot(problem.Jac(x, lmbda + eps1), z) / eps1
            - np.dot(J, z) * (1.0 / eps3 + 1.0 / eps1)
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
        dx = a + dlmbda * b
        dy = e + dlmbda * g - domega * c - y
        dz = f + dlmbda * h - domega * d - z

        x += dx
        y += dy
        z += dz
        omega += domega
        lmbda += dlmbda
        num_newton_steps += 1

        # Update residuals and jacobian
        J = problem.Jac(x, lmbda)
        Rx = problem.Res(x, lmbda)
        Ry = (
            np.dot(J, y) + omega * z
        )  # here I assume that B is the identity matrix (neglect thermal pressurization)
        Rz = (
            np.dot(J, z) - omega * y
        )  # here I assume that B is the identity matrix (neglect thermal pressurization)
        Romega = np.dot(phi, y) - 1.0
        Rlmbda = np.dot(phi, z)

        Rnrm_old = Rnrm
        Rnrm = np.sqrt(
            np.linalg.norm(Rx) ** 2
            + np.linalg.norm(Ry) ** 2
            + np.linalg.norm(Rz) ** 2
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
    print("lmbda = {}, ||x|| = {}".format(lmbda, np.linalg.norm(x)))
    print(colored("Solve did NOT converge!", "red"))
    return x, lmbda, newton_success
