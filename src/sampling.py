from scipy import integrate
from scipy.optimize import root_scalar
import numpy as np

def samples_from_pdf(pdf, N, x0, x1=None, args=None, cdf=None, range_integral_value=None):
    """
    Return samples of a probability distribution having the same distance
    measured in units of the corresponding cumulative distribution function
    (cdf).

    To calculate the integral of the pdf in the target bounds, there are three methods available.
        - cdf callback (being the integral of the pdf), then it is used to calculate alpha (needs x1)
        - range_integral_value to give alpha directly if it is known (x1 is ignored in this case)
        - numerical integration of the pdf (default, needs x1)

    :param pdf: The probability density function of the requested distribution
    :type pdf: callable

    :param N: Number of samples to return
    :type N: int

    :param x0: First sample point
    :type x0: float

    :param x1: Last sample point, optional, see above when required.
    :type x1: float, default None

    :param args: Additional arguments for the pdf callable, optional.
    :type args: tuple, default None

    :param cdf: The cdf (integral of pdf), optional.
    :type cdf: callable, default None

    :param range_integral_value: Value of the integral of pdf from x0 to x1. Optional.
    :type range_integral_value: float, default None

    """
    if args is None:
        args = ()

    if range_integral_value is not None:
        alpha = range_integral_value
    if cdf is not None:
        if x1 is None:
            raise ValueError("Need x1 for calculating the integral with the given cdf.")
        
        alpha = cdf(x1, *args) - cdf(x0, *args)
    else:
        if x1 is None:
            raise ValueError("Need x1 for calculating the numerical integral of the pdf.")

        # numerical integral
        alpha = integrate.quad(pdf, x0, x1, args=args)[0]

    def disteq(eps, lastx, N, pdf, alpha, args):
        # trapezoidal rule, giving integral of pdf from lastx to lastx + eps
        deltacdf = (pdf(lastx, *args) + pdf(lastx + eps, *args)) * eps / 2
        return deltacdf - alpha/N
    
    def distderiv(eps, lastx, N, pdf, alpha, args):
        return pdf(lastx + eps, *args)

    pdf_v = np.vectorize(pdf)

    lastx = x0
    samples = []
    for _ in range(N):
        eps = root_scalar(disteq, x0=0, fprime=distderiv, args=(lastx, N, pdf, alpha, args))
        newx = lastx + eps.root
        samples.append(newx)
        lastx = newx
        
    samples = np.array(samples)

    # if alpha == 0:
    #     return samples, None
    # else:
    return samples, 1 / alpha
