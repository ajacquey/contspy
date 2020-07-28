import numpy as np


def solve_Newton(
    R, J, x0, abs_tol=1.0e-10, rel_tol=1.0e-10, max_iters=200, debug=False
):
    """
    Solve nonlinear system R=0 by Newton's method.
    J is the Jacobian of R. Both R and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < abs_tol or ||F|| / ||F0|| < rel_tol.
  """

    # Initial Residuals
    x = x0
    Rx = R(x)
    Rnrm = np.linalg.norm(Rx)
    Rnrm0 = Rnrm

    # Iteration
    k = 0
    while k < max_iters:
        # Convergence criteria
        if Rnrm < abs_tol or np.divide(Rnrm, Rnrm0) < rel_tol:
            return x, k

        # Update solution
        dx = np.linalg.solve(J(x), -Rx)
        x += dx
        Rx = R(x)
        Rnrm = np.linalg.norm(Rx)  # l2 norm of vector
        k += 1

    # Here, no solutions have been found after max_iters iterations
    raise Exception(
        "No solution has been found after ", k, " iterations in the Newton solve."
    )
