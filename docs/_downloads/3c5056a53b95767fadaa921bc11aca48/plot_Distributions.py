"""_summary_
"""
from matplotlib import pyplot as plt
import numpy as np
from geobipy import Distribution, DataArray, RectilinearMesh2D, Histogram, get_prng

#%%
# Create a histogram so we can test joint probabilities
x = DataArray(np.linspace(-4.0, 4.0, 100), 'Variable 1')
y = DataArray(np.linspace(-4.0, 4.0, 105), 'Variable 2')

mesh = RectilinearMesh2D(x_edges=x, y_edges=y)
# Instantiate
H = Histogram(mesh)

#%%
# Update the histogram counts
H.update(np.random.randn(1000000), np.random.randn(1000000))
plt.figure()
H.plot()

prng = get_prng()
#%%
# dist = Distribution('uniform', 0.0, 1.0, prng=prng)

# plt.figure()
# dist.plot_pdf()
# dist.plot_pdf(log=True)

# p = H.compute_probability(dist, axis=0)

# #%%
# dist = Distribution('normal', 0.0, 1.0, prng=prng)

# plt.figure()
# dist.plot_pdf()
# dist.plot_pdf(log=True)

# print(H.compute_probability(dist, axis=0))

#%%
# Multivariate normal
dist = Distribution('mvnormal', [-2, 0, 2], np.diag([1.0, 1.0, 1.0]), prng=prng)

plt.figure()
dist.plot_pdf()

p = H.compute_probability(dist, axis=0)
plt.figure()
p.plot()

#%%
dist = Distribution('lognormal', 1.0, 1.0, linearSpace=True, prng=prng)

plt.figure()
dist.plot_pdf()
dist.plot_pdf(log=True)


dist = Distribution('mvlognormal', np.r_[1.0, 2.0, 3.0], np.r_[1.0, 1.0, 1.0], prng=prng)

plt.figure()
dist.plot_pdf()

plt.show()