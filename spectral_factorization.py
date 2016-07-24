import numpy as np
from scipy import linalg as spla

class FactorizationError(RuntimeError):
    '''
    An exception raised when the spectral factorization fails.
    '''
    pass

def spectral_factorization(P, extra_zeros = 300):
    '''Performs a spectral factorization of the power spectrum P using
    the method of Bauer.  See the reference

    P should be an array containing the coefficients of the causal part
    of the power spectrum, in order from highest power to lowester.
    That is,

    P = p_0 + p_1*z^-1 + ... + p_m*z^-m

    We return a tuple (Q, s) where Q is the first monic spectral factor
    with the same high to low ordering as P.  And s is the scaling
    term.  The total factorization is

    P = s^2*Q(z)*Q*(1/z*)

    In order to factor a rational power spectrum, feed in the numerator
    and denominator separately to this function.

    The extra_zeros parameter defines how large of a companion matrix we
    construct.  As it gets larger the estimate should be more accurate,
    however there are numerical conditioning issues as it increases.  In
    order to be safe, this method is called recursively with decreasing
    extra_zeros if the factorization fails.  This is slow, and there are
    certainly better methods, but this works.

    See the paper below for references.

    A. H. Sayed and T. Kailath, “A survey of spectral factorization
    methods,” Numerical Linear Algebra with Applications, vol. 8,
    no. 6-7, pp. 467–496, 2001. [Online]. Available:
    http://dx.doi.org/10.1002/nla.250
    '''
    #This recursion ensures we will get a valid Q(z)
    #As extra_zeros increases, T may not be positive definite,
    #and even if it is, sometimes the roots of Q will still be
    #outside the unit circle.  This is a total hack to ensure
    #that the function always returns something valid.  But
    #also note that it usually succeeds and these problems are
    #rare.
    def spectral_factorization_rec(P, extra_zeros):
        m = len(P)
        if extra_zeros < 0:
            raise FactorizationError('Failed to produce '\
                                     'a valid factorization')

        #First we need to first find extra_zeros so that T is PSD
        T = spla.toeplitz(np.append(P, np.zeros(extra_zeros)))
        try:
            L, D = LDU(T)
        except spla.LinAlgError: #T not PSD
            return spectral_factorization_rec(P, extra_zeros - 30)

        Q = L[-1,:][::-1][:m] #Last m elements of bottom row
        s = D[-1] #Last element of D
        if not np.all([abs(z) < 1 for z in np.roots(s*Q)]):
            #Roots must be inside unit circle
            return spectral_factorization_rec(P, extra_zeros - 30)
        else:
            return Q, s

    #----FUNCTION ENTRY POINT----
    return spectral_factorization_rec(P, extra_zeros)

def LDU(A):
    '''
    Computes the LDU decomposition of A.  We return L and D.  CARE:
    D is returned as a vector.  So A != dot(L, D), L.T), but A = dot(L,
    diag(D), L.T).  Note that A must be positive semidefinite.

    This method isn't native to numpy or scipy, so I've rolled my own.
    LDU is a faster decomposition than Cholesky since it avoids
    computing square roots, there is recent discussion on the numpy
    board for merging in LDU.
    '''
    #This can be generalized for complex A
    assert np.allclose(A.imag, 0)
    A = A.real
    L = spla.cholesky(A, lower = True)
    D = np.diag(L)
    L = L/D
    D = D**2
    LDU = np.dot(np.dot(L, np.diag(D)), L.T)
    assert np.allclose(A, LDU)
    assert np.allclose(LDU.imag, 0)
    assert np.allclose(L.imag, 0)
    assert np.allclose(D.imag, 0)
    return L.real, D.real
