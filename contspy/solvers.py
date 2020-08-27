import numpy as np
from termcolor import colored


def solve_Newton(
    R, J, u0, abs_tol=1.0e-10, rel_tol=1.0e-10, max_iters=200, debug=False
):
    """
    Solve nonlinear system R=0 by Newton's method.
    J is the Jacobian of R. Both R and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < abs_tol or ||F|| / ||F0|| < rel_tol.
  """

    # Initial Residuals
    u = u0
    Ru = R(u)
    Ju = J(u)
    Rnrm = np.linalg.norm(Ru)
    Rnrm0 = Rnrm

    # Iterations
    num_newton_steps = 0
    newton_success = False

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
            print(colored("Solve Converged!", "green"))
            newton_success = True
            return u, newton_success

        # Solve linear system
        du = np.linalg.solve(Ju, -Ru)

        # Update solution
        u += du
        num_newton_steps += 1

        # Update residuals and jacobian
        Ru = R(u)
        Ju = J(u)

        # Print residuals
        Rnrm_old = Rnrm
        Rnrm = np.linalg.norm(Ru)  # l2 norm of vector
        if Rnrm < 0.8 * Rnrm_old:
            color = "green"
        elif Rnrm > Rnrm_old:
            color = "red"
        else:
            color = "yellow"

        print(f"{num_newton_steps} Nonlinear |R| = ", colored(f"{Rnrm:.6e}", color))

    # Here, no solutions have been found after max_iters iterations
    print("Solve diverged due to MAX_ITER.")
    print(colored("Solve did NOT converge!", "red"))
    return u, newton_success
