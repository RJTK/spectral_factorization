import numpy as np
import cmath

from math import sqrt
from numpy import poly, roots, polymul
from scipy import stats
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import patches

try:
    from progressbar import ProgressBar, Bar, ETA, Percentage
    prog_bar = True
except ImportError:
    prog_bar = False

try:
    from tabulate import tabulate
    tab = True
except ImportError:
    tab = False

from spectral_factorization import spectral_factorization,\
    FactorizationError

def random_arma(p, q, k = 1, z_radius = 1, p_radius = 0.75):
  '''
  Returns a random ARMA(p, q) filter.  The parameters p and q define
  the order of the filter where p is the number of AR coefficients
  (poles) and q is the number of MA coefficients (zeros).  k is the
  gain of the filter.  The z_radius and p_radius paramters specify the
  maximum magnitude of the zeros and poles resp.  In order for the
  filter to be stable, we should have p_radius < 1.  The poles and
  zeros will be placed uniformly at random inside a disc of the
  specified radius.

  We also force the coefficients to be real.  This is done by ensuring
  that for every complex pole or zero, it's recipricol conjugate is
  also present.  If p and q are even, then all the poles/zeros could
  be complex.  But if p or q is odd, then one of the poles and or
  zeros will be purely real.

  The filter must be causal.  That is, we assert p >= q.

  Finally, note that in order to generate complex numbers uniformly
  over the disc we can't generate R and theta uniformly then transform
  them.  This will give a distribution concentrated near (0, 0).  We
  need to generate u uniformly [0, 1] then take R = sqrt(u).  This can
  be seen by starting with a uniform joint distribution f(x, y) =
  1/pi, then applying a transform to (r, theta) with x = rcos(theta),
  y = rsin(theta), calculating the distributions of r and theta, then
  applying inverse transform sampling.
  '''
  assert(p >= q), 'System is not causal'
  P = []
  Z = []
  for i in range(p % 2):
    pi_r = stats.uniform.rvs(loc = -p_radius, scale = p_radius)
    P.append(pi_r)
    
  for i in range((p - (p % 2)) / 2):
    pi_r = sqrt(stats.uniform.rvs(loc = 0, scale = p_radius))
    pi_ang = stats.uniform.rvs(loc = -np.pi, scale = np.pi)
    P.append(cmath.rect(pi_r, pi_ang))
    P.append(cmath.rect(pi_r, -pi_ang))

  for i in range(q % 2):
    zi_r = stats.uniform.rvs(loc = -z_radius, scale = z_radius)
    Z.append(zi_r)

  for i in range((q - (q % 2)) / 2):
    zi_r = stats.uniform.rvs(loc = 0, scale = z_radius)
    zi_ang = stats.uniform.rvs(loc = -np.pi, scale = np.pi)
    Z.append(cmath.rect(zi_r, zi_ang))
    Z.append(cmath.rect(zi_r, -zi_ang))

  b, a = signal.zpk2tf(Z, P, k)

  return b, a

#------------------------------------------------
def test1():
    '''
    Runs the spectral factorization routine on a few trivial examples
    and plots the results.
    '''
    #Specify zeros of the filter which produces the PSD
    z1 = 0.5
    z2 = -0.5 + 0.75j
    z3 = 0.99 + 0.01j
    H1_z, P1_z = [z1], [z1, 1/z1]
    H2_z, P2_z = [z1, z2, z2.conjugate()], [z1, 1/z1, z2, 1/z2,
                                            z2.conjugate(),
                                            1/z2.conjugate()]
    H3_z, P3_z = [z3, z3.conjugate()], [z3, 1/z3, z3.conjugate(),
                                        1/z3.conjugate()]

    for H_z, P_z in [(H1_z, P1_z), (H2_z, P2_z), (H3_z, P3_z)]:
        P = poly(H_z)
        P = polymul(P[::-1], P) #Creates a valid PSD polynomial

        #Pass in only the causal portion
        Q, s = spectral_factorization(P[len(P)/2:])
        Q_zeros = roots(s*Q)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        uc = patches.Circle((0, 0), radius = 1, fill = False,
                            color = 'black', ls = 'dashed')
        ax.add_patch(uc)
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.plot([z.real for z in P_z], [z.imag for z in P_z],
                 'b+', markersize = 12, mew = 3)
        plt.plot([z.real for z in Q_zeros], [z.imag for z in Q_zeros],
                 'rx', markersize = 12, mew = 3)

        plt.gca().set_aspect('equal')  
        plt.show()
    return

def test2():
    '''
    Performs a number of randomized tests with visualizations
    '''
    N = 10 #Number of tests
    p = 7 #Number of roots
    np.random.seed(1)
    for i in range(N):
        b, H = random_arma(p, 0, p_radius = 1) #Random filter

        P = polymul(H[::-1], H) #Creates a valid PSD polynomial

        #Pass in only the causal portion
        Q, s = spectral_factorization(P[len(P)/2:])
        Q_zeros = roots(s*Q)
        P_zeros = roots(P)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        uc = patches.Circle((0, 0), radius = 1, fill = False,
                            color = 'black', ls = 'dashed')
        ax.add_patch(uc)
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.plot([z.real for z in P_zeros], [z.imag for z in P_zeros],
                 'b+', markersize = 12, mew = 3)
        plt.plot([z.real for z in Q_zeros], [z.imag for z in Q_zeros],
                 'rx', markersize = 12, mew = 3)

        plt.gca().set_aspect('equal')  
        plt.show()
    
    return

def test3():
    '''
    Performs randomized tests with less visualization.
    '''
    N = 100 #Number of tests
    p = [3, 7, 11, 16, 21, 35, 70, 141, 211] #Number of roots
    np.random.seed(1)
    #Tolerances for our checks
    tols = [1./(10**i) for i in range(1, 10)]
    factorization_fails = np.array([0]*len(p))

    for p_index, num_poles in enumerate(p):
        if prog_bar:
            pbar = ProgressBar(widgets = ('p = %d ' % num_poles, 
                                          Bar(), Percentage(),
                                          ETA()), maxval = N).start()
        num_fails = np.array([0]*len(tols))
        for trial_num in range(N):
            if prog_bar:
                pbar.update(trial_num)
            b, H = random_arma(num_poles, 0, p_radius = 1) #Random filter

            #Creates a valid PSD polynomial
            #Only the causal portion is needed
            P = polymul(H[::-1], H)

            try:
                Q, s = spectral_factorization(P[len(P)/2:])
            except FactorizationError:
                factorization_fails[p_index] += 1
                num_fails += 1
                continue

            Q_zeros = np.sort(roots(s*Q))
            P_zeros = roots(P)
            P_minphase_zeros = np.sort([z for z in P_zeros if abs(z) < 1])
            
            for (atol_i, atol) in enumerate(tols):
                if not np.allclose(Q_zeros, P_minphase_zeros,
                                   rtol = 0, atol = atol):
                    num_fails[atol_i] += 1

        try:
            total_fails = np.vstack((total_fails, num_fails))

        except NameError:
            total_fails = num_fails

        if prog_bar:
            pbar.finish()

    total_fails = total_fails / float(N) #Make everything a ratio
    if tab:
        print 'Fraction of time (over %d trials) accuracy not within... '\
            '(oo indicates a failure to return a valid factorization)' % N
        headers = ['p', 'oo'] + tols
        table = []
        for i, pi in enumerate(p):
            row_i = [pi] + [factorization_fails[i]] + list(total_fails[i])
            table.append(row_i)
        print tabulate(table, headers = headers)
        results_file = open('test_results.txt', 'w')
        results_file.write('Fraction of time (over %d trials) accuracy '\
                           'not within...\n(oo indicates a failure to '\
                           'return a valid factorization)\n' % N)
        results_file.write(tabulate(table, headers = headers))
        results_file.write('\n')
        results_file.close()
    return

if __name__ == '__main__':
    #test1()
    #test2()
    test3()
