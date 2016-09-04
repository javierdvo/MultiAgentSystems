import numpy as np
# import scipy as sp

"""
Copyright 2012, Michael Andersen, DIKU. michael (at) diku (dot) dk
"""


def ip_lcp(A, b, x0, max_iter=0,
           tol_rel=0.00001, tol_abs=np.finfo(np.float64).eps*10,
           solver='central_trajectory', profile=True):
    """
    Interior point solver for lcp using central trajectory and merit
    reduction (not implemnted).

    :Parameters:
    A: The LCP matrix
    b: The LCP matrix
    x0: Initial guess
    max_iter: Maximum number of iterations used to solve the lcp.
    tol_rel: Relative tolerence, break if the error changes less
        than this between two iterations.
    tol_abs: Absolute tolerence, break if the error drops below
        this.
    solver: Chooses which subsolver for solving the subsystem, 
        choices should contain merit reduction (not
        implemented) and central trajectory (working).
    profile: whether or not to profile while running, log
        convergence and perhaps other properties.

    :Returns:
    x: the solution
    err: the error
    iterate: number of iterations
    flag: return flag
    convergence: the error for each iteration
    msg: exit message for return flag

    ##### Explaination of how the Interior Point method works #####
    ## From the original ip_lcp.m by Kenny Erleben

    Given  A,b then the LCP is defined by

            A x + b >= 0
                  x >= 0
        x^T(A x + b) = 0

    We rewrite this using the slack variable y = Ax + b as

          (Ax - y + b)_i = 0
                 x_i y_i = 0
               x_i, y_i >= 0

    Idea is to solve this iteratively while keeping x_i^k, y_i^k > 0. To
    ensure invariant we introduce a postive value gamma such that

          (Ax - y + b)_i = 0
         x_i y_i - gamma = 0
                x_i, y_i > 0

    Here gamma is a relaxed complimentarity measure (more on this below).
    Now we introduce the Kojima mapping which we use to reformulate the
    relaxed LCP formulation above

                           |  A x - y + b        |
           F(x,y:gamma) =  |  M W e  - gamma e   | = 0

    where M and W are diagonal matrices with M_ii = x_i and W_ii = y_i and
    e is the nD vector of ones. In the k'th iteration we pick gamma to be

             gamma^k = sigma (x^ k)^T (y^k)/n

    where n is the dimenstion of the problem and the relaxation (centering)
    parameter is 0 < sigma < 1. In the k'th iteration of our framework we
    solve F(x,y: gamma^k)=0 approximately. The solutions of the Kojima map
    for all positive values of gamma define the central path which is a
    trajectory that leads to the solution of the LCP as gamma tends to zero.
    Using Taylor series expansion we define the Newton equation

            |  A    -I  | | dx^k |   | - A x^k + y^k - b     |
            |           | |      | = |                       |
            |  W^k  M^k | | dy^k |   | - M^k W^k e + gamma e |

    Having solved for the Newton direction ( dx^k, dy^k ) we do the Newton
    update

        x^(k+1)  = x^k  + tau^k dx^k
        y^(k+1)  = y^k  + tau^k dy^k

    where tau^k is the step length. The step length is chosen as the
    maximum positive value such that

        x^(k+1)_i > 0 and y^(k+1)_i > 0 for all i    (**)

    That is we choose tau^k = min( tau^x, tau^y ) where

        tau^x = max{ 0 < tau  < 1 |  x^k + tau dx^k >= (1-alpha) x^k }
        tau^y = max{ 0 < tau  < 1 |  y^k + tau dy^k >= (1-alpha) y^k }

    Here 0 < alpha < 1 controls how far we back away from the maximum step
    for which the conditions (**) are statisfied.

    To accelerate the convergence we make alpha approach one as the iterates
    approach the solution
    """

    # Human readable dictionary of exit messages
    msg = {1: 'preprocessing',  # flag = 1
           2: 'iterating',      # flag = 2
           3: 'relative',       # flag = 3
           4: 'absolute',       # flag = 4
           5: 'stagnation',     # flag = 5
           6: 'local minima',   # flag = 6
           7: 'nondescent',     # flag = 7
           8: 'maxlimit',       # flag = 8
           }

    # This sould use shape instead of size (as size gives the number of elements)
    N = np.size(b)
    flag = 1

    if max_iter == 0:
        max_iter = N

    # Setup the things we need to compute the solution.
    y0 = np.dot(A,x0)+b
    e  = np.ones((N,1))
    I  = np.identity(N)

    eps = np.finfo(np.float64).eps

    if np.dot(x0.T,y0) <= 0:
        W = np.diag(y0[:,0])
        M = np.diag(x0[:,0])
        # Jtmp1 = [A | -I]
        Jtmp1 = np.hstack((A,-I))
        # Jtmp2 = [W | M]
        Jtmp2 = np.hstack((W,M))
        #     | Jtmp1 |
        # J = | ----- |
        #     | Jtmp2 |
        J = np.vstack((Jtmp1,Jtmp2))
        #      | A*x0-y0+b |
        # Fk = | --------- |
        #      | M*W*e - 1 |
        Fk = np.vstack((np.dot(A,x0) - y0 + b,np.dot(np.dot(M,W),e)-1))

        # Inverse condition number of J
        rcond = 1/np.linalg.cond(J)

        # Is the problem well conditioned, if not use pinv.
        if rcond < 2*eps:
            p = np.dot(-np.linalg.pinv(J),Fk)
        else:
            p = -np.linalg.solve(J,Fk)

        dx = p[:N]
        dy = p[N:]

        x0 = np.maximum(1, np.abs(x0 + dx))
        y0 = np.maximum(1, np.abs(y0 + dy))

    # Should use np.infty.
    old_err = 1e20
    # Copy over x0 and y0
    x = x0[:]
    y = y0[:]
    # Iterate
    iterate = 1
    # Track convergence rate (if profile is true)
    convergence = np.zeros(max_iter+1)
    
    flag = 2

    grad_steps = 0

    while iterate <= max_iter: # We start with 1 not zero as we are
                               # supposed to.

        # Merit value, complementarity measure, error.
        err = np.dot(x.T, y)

        if err <= 0:
            print(">>> ERROR x'*y = %2.5f" % (err))
            return

        if profile:
            convergence[iterate-1] = err

        if np.abs(err - old_err)/np.abs(old_err) < tol_rel:
            flag = 3
            break

        if err < tol_abs:
            flag = 4
            break
        
        sigma = np.float64(1)/iterate # Goes towards zero as we iterate
        gamma = sigma*np.dot(x.T,y)/N # Should go towards zero as we iterate
        
        ##### Setup the Newton system ###############

        # if 'central_trajectory' == str.lower(solver):
        W = np.diag(y[:,0])
        M = np.diag(x[:,0])

        Jtmp1 = np.hstack((A,-I))
        Jtmp2 = np.hstack((W, M))
        J = np.vstack((Jtmp1,Jtmp2))
        Fk = np.vstack((np.dot(A,x) - y + b,
                        np.dot(np.dot(M,W),e)-gamma*e))
        
        ##### Solve the Newton system ###############

        p = -np.linalg.solve(J,Fk)
        # linalg.inv is very slow.
        # p = np.dot(-np.linalg.inv(J), Fk)

        # elif 'schur_complement':

           # Needs to be implmented

        # elif 'potential_reduction':

           # Needs to be implmented

        # else:
        #     print ">>> ERROR: Unknown subsolver: {}".format(solver)

        dx = p[:N]
        dy = p[N:]

        # Controls how much we step back from the maximum step
        # length. Such that we still respect the conditions
        # x^{(k+1)}_i > and y^{(k+1)}_i > 0. 0 < alpha < 1
        alpha = np.float64(iterate)/(max_iter+1.0) # Goes towards one as we iterate

        xx = ((1-alpha)*x - x)/dx
        yy = ((1-alpha)*y - y)/dy

        if not np.size(xx[dx < 0]) == 0:
            taux = np.minimum( 1, np.min(xx[dx < 0]))
        else:
            taux = 1

        if not np.size(yy[dy < 0]) == 0:
            tauy = np.minimum(1, np.min(yy[dy < 0]))
        else:
            tauy = 1

        # Verify that the step length is valid.
        tau = np.min([taux, tauy])

        if tau < 0 or tau > 1:
            print(">>> ERROR tau = %2.20f" % tau)
            return
        
        ### Take the Newton step ###############
        x = x + taux*dx
        y = y + tauy*dy

        # Verify that the x and y vector are still valid. x > 0 and y
        # > 0.
        if np.min(x) <= 0:
            print(">>> ERROR Min x = %2.20f" % (np.min(x)))
            return

        if np.min(y) <= 0:
            print(">>> ERROR Min y = %2.20f" % (np.min(y)))
            return

        iterate = iterate + 1
    
    if iterate >= max_iter:
        iterate -= 1
        flag = 8
        
    # Return the solution, err, iterates, exit flag, convergence and
    # human readable message.
    return (x, err, iterate, flag, convergence[:iterate], msg[flag])
