import math
import numpy as np
import matplotlib.pyplot as plt


"""
Study into kernel density estimation.

Kernel density estimation is a method for estimating a probabiliy density function based on sampled data.
Each data point adds a contribution to the overall function. The expectation is that for data drawn from
a normally distributed random variable, the estimate will approach a normal distribution as more samples
are taken.

For each data point, draw a normal curve centred at that point with a given standard deviation (discussed
later). The sum of all of these normal curves is our PDF estimate.

The value of the standard deviation is closely related to the bandwidth of the estimation process.
Setting a small standard deviation will lead to small overlaps between the PDFs drawn from the data points,
and so the resulting estimate function will be spiky. High standard deviation will lead to large overlaps
between the PDFs, so the estimate function will be smoother.

The individual PDFs are called the kernels. The form of the kernel can be chosen
"""


def normal_pdf_at_x(x, h, stdev=2.25, samples=1000):
    """
    Produce a normal pdf with the given standard deviation, centred at x.
    """
    rng_min = -10
    rng_max = 25.1
    interval = (rng_max - rng_min) / samples
    rng = np.arange(rng_min, rng_max, interval)
    f = 1/stdev/h/math.sqrt(2*np.pi)*np.exp(-0.5 * ((rng-x)/stdev)**2)
    return rng, f


if __name__ == "__main__":
    ## Bandwidth parameter, not yet fully understood or properly implemented.
    h = 0.7

    ## Create a random sampling of data.
    data = [0.1, 0.3, 0.5, -1.5, 0.5, -1.1, 8, 2.4, 5.4, 10.4, 10.5, 2.5, 2.1, 3.1]

    ## These will contain the f(x) pairs to be plotted.
    X = list()
    F = list()

    ## For each value in the dataset, produce a normal curve centred at that value.
    for value in data:
        x, f = normal_pdf_at_x(value, h)
        X.append(x)
        F.append(f)

    ## Draw the data points on the x axis.
    plt.scatter(data, [0]*len(data))

    ## Draw each normal curve.
    for i in range(len(X)):
        plt.plot(X[i], F[i], "--", linewidth=1)

    ## Draw the sum of all the normal curves. This is our estimated PDF.
    FF = F[0]
    for f in F[1:]:
        FF += f
    plt.plot(X[0], FF/len(data)/h, linewidth=5)

    plt.show()
