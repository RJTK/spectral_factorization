# spectral_factorization
Code for performing power spectral density factorizations

The power spectral density of a random process can be factorized into minimum and maximum phase factors, along with a scalar.

P(z) = s^2 Q(z) Q\*(1/z\*)

The attached .pdf documents "spectral_factorization_theory.pdf" as well as "spectral_factorization_example.pdf"
provide more information.  I pulled these references from here: (http://ece-research.unm.edu/bsanthan/ece539/)

Given a PSD P(z), the spectral factorization recovers the factor Q(z) and the scaling term s.  This factorization has
a wide variety of applications.  For example, filtering white noise with the filter sQ(z) will produce a random process
with PSD P(z).  Here is an example image.

![alt tag](https://raw.githubusercontent.com/RJTK/spectral_factorization/master/example_factorization.png)

Performing this factorization is rather difficult since polynomial root finding is numerically ill-conditioned.  It
is not reliable to simply find the roots of P(z) with a generic root finding algorithm and then pick out the roots
inside the unit circle to obtain Q(z).

The test file attempts to quantify the reliability.  The table shows the fraction of tests (over 100 trials) that found every zero accurately to within some absolute tolerance.  Sometimes we fail to return a valid factorization at all, indicated by the number in the "oo" (for infinity) column.  The implementation is fairly reliable for polynomials with up to about 15 roots.

```
Fraction of time (over 100 trials) accuracy not within...
(oo indicates a failure to return a valid factorization)
  p    oo    0.1    0.01    0.001    0.0001    1e-05    1e-06    1e-07    1e-08    1e-09
---  ----  -----  ------  -------  --------  -------  -------  -------  -------  -------
  3     0   0       0.01     0.02      0.03     0.05     0.06     0.08     0.08     0.1
  7     0   0       0        0.05      0.09     0.11     0.13     0.13     0.13     0.16
 11     0   0       0.02     0.04      0.13     0.15     0.19     0.25     0.27     0.33
 16     0   0.01    0.04     0.15      0.26     0.34     0.41     0.46     0.5      0.58
 21     1   0.05    0.09     0.19      0.26     0.37     0.44     0.48     0.52     0.61
 35     7   0.11    0.18     0.33      0.44     0.51     0.55     0.61     0.7      0.82
 70    29   0.5     0.57     0.72      0.8      0.9      0.98     0.99     1        1
141    69   0.93    0.93     1         1        1        1        1        1        1
211    74   0.98    1        1         1        1        1        1        1        1
```

The code in this repo implements the method of Bauer for performing a PSD factorization.  More information can be found here:

A. H. Sayed and T. Kailath, “A survey of spectral factorization methods,”
Numerical Linear Algebra with Applications, vol. 8, no. 6-7, pp. 467–496,
2001. [Online]. Available: http://dx.doi.org/10.1002/nla.250

The test.py file provides some tests and examples.

Some other nice references here: http://stanford.edu/~boyd/papers/magdes.html
