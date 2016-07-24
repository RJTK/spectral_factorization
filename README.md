# spectral_factorization
Code for performing power spectral density factorizations

The power spectral density of a random process can be factorized into minimum and maximum phase factors, along with a scalar.

P(z) = s^2 Q(z) Q\*(1/z\*)

The attached .pdf documents "spectral_factorization_theory.pdf" as well as "spectral_factorization_example.pdf"
provide more information.  I pulled these references from here: (http://ece-research.unm.edu/bsanthan/ece539/)

Given a PSD P(z), the spectral factorization recovers the factor Q(z) and the scaling term s.  This factorization has
a wide variety of applications.  For example, filtering white noise with the filter sQ(z) will produce a random process
with PSD P(z).

Performing this factorization is rather difficult since polynomial root finding is numerically ill-conditioned.  It
is not reliable to simply find the roots of P(z) with a generic root finding algorithm and then pick out the roots
inside the unit circle to obtain Q(z).

The code in this repo implements the method of Bauer for performing a PSD factorization.  More information can be found here:

A. H. Sayed and T. Kailath, “A survey of spectral factorization methods,”
Numerical Linear Algebra with Applications, vol. 8, no. 6-7, pp. 467–496,
2001. [Online]. Available: http://dx.doi.org/10.1002/nla.250

The test.py file provides some tests and examples.
