.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Statistics_plot_StatArray.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Statistics_plot_StatArray.py:


StatArray Class
----------------

Extends the numpy ndarray class to add extra attributes such as names, and
units, and allows us to attach statistical descriptors of the array.
The direct extension to numpy maintains speed and functionality of numpy arrays.


.. code-block:: default

    from geobipy import StatArray
    from geobipy import Histogram1D
    import numpy as np
    import matplotlib.pyplot as plt
    import h5py
    from geobipy import hdfRead









Instantiating a new StatArray class
+++++++++++++++++++++++++++++++++++

The StatArray can take any numpy function that returns an array as an input.
The name and units of the variable can be assigned to the StatArray.


.. code-block:: default


    Density = StatArray(np.random.randn(1), name="Density", units="$\frac{g}{cc}$")
    Density.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Name: Density
         Units: $\frac{g}{cc}$
         Shape: (1,)
         Values: [-1.78582282]





Attaching Prior and Proposal Distributions to a StatArray
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The StatArray class has been built so that we may easily 
attach not only names and units, but statistical distributions too.  
We won't go into too much detail about the different distribution 
classes here so check out the :ref:`Distribution Class` for a better description.

Two types of distributions can be attached to the StatArray.

* Prior Distribution
    The prior represents how the user believes the variable should 
    behave from a statistical standpoint.  
    The values of the variable can be evaluated against the attached prior, 
    to determine how likely they are to have occured https://en.wikipedia.org/wiki/Prior_probability

* Proposal Distribution
    The proposal describes a probability distribution from which to 
    sample when we wish to perturb the variable 
    https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm


.. code-block:: default


    # Obtain an instantiation of a random number generator. 
    # This is optional, but is an important consideration for parallel programming.
    prng = np.random.RandomState()
    Density.setPrior('Uniform', -2.0, 2.0, prng=prng)








We can also attach a proposal distribution


.. code-block:: default

    Density.setProposal('Normal', 0.0, 1.0, prng=prng)
    Density.summary()
    print("Class type of the prior: ",type(Density.prior))
    print("Class type of the proposal: ",type(Density.proposal))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Name: Density
         Units: $\frac{g}{cc}$
         Shape: (1,)
         Values: [-1.78582282]
    Prior: 
         Uniform Distribution: 
      Min: :-2.0
      Max: :2.0
    Proposal: 
    Normal Distribution: 
        Mean: :0.0
    Variance: :1.0

    Class type of the prior:  <class 'geobipy.src.classes.statistics.UniformDistribution.Uniform'>
    Class type of the proposal:  <class 'geobipy.src.classes.statistics.NormalDistribution.Normal'>




The values in the variable can be evaluated against the prior.
In this case, we have 3 elements in the variable, and a univariate Normal for the prior. 
Therefore each element is evaluated to get 3 probabilities, one for each element.


.. code-block:: default

    print(Density.probability(log=False))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.25




The univariate proposal distribution can generate random samples from itself.


.. code-block:: default

    print(Density.propose())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    -0.18202958893220814




From a sampling stand point we can either sample using only the proposal
Or we can only generate samples that simultaneously satisfy the prior. 


.. code-block:: default

    print(Density.propose(relative=True))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [-1.33762956]




We can perturb the variable by drawing from the attached proposal distribution.


.. code-block:: default


    Density.perturb()
    Density.summary()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Name: Density
         Units: $\frac{g}{cc}$
         Shape: (1,)
         Values: [-1.20100922]
    Prior: 
         Uniform Distribution: 
      Min: :-2.0
      Max: :2.0
    Proposal: 
    Normal Distribution: 
        Mean: :0.0
    Variance: :1.0





Attaching a Histogram to capture the posterior distribution
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The StatArray can perturb itself, evaluate its current probability given its priors
and a histogram can be attached to capture its posterior distribution.
As an example, lets create a Histogram class with bins generated from the prior.


.. code-block:: default

    bins = Density.prior.bins()
    post = Histogram1D(bins=bins)








Attach the histogram


.. code-block:: default

    Density.setPosterior(post)








In an iterative sense, we can propose and evaluate new values, and update the posterior


.. code-block:: default

    for i in range(1000):
        Density.perturb()
        p = Density.probability(log=False)

        if p > 0.0: # This is a simple example!
            Density.updatePosterior()









.. code-block:: default

    plt.figure()
    Density.summaryPlot()




.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_001.png
    :class: sphx-glr-single-img





Attach a multivariate normal distribution as the prior and proposal
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Attach the multivariate prior


.. code-block:: default


    mean = np.random.randn(Density.size)
    variance = np.ones(Density.size)
    Density.setPrior('MvNormal', mean, variance, prng=prng)









Since the prior is multivariate, the appropriate equations are used to
evaluate the probability for all elements in the StatArray.
This produces a single probability.


.. code-block:: default


    print(Density.probability(log=False))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.1300835201082144




Attach the multivariate proposal


.. code-block:: default


    mean = np.random.randn(Density.size)
    variance = np.ones(Density.size)
    Density.setProposal('MvNormal', mean, variance, prng=prng)









Perturb the variables using the multivariate proposal.


.. code-block:: default


    Density.perturb()
    Density.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Name: Density
         Units: $\frac{g}{cc}$
         Shape: (1,)
         Values: [-0.38547039]
    Prior: 
         MV Normal Distribution: 
        Mean: [1.35036815]
        Variance: [1.]
    Proposal: 
    MV Normal Distribution: 
        Mean: [-0.70953006]
        Variance: [1.]
    Posterior: 
    <class 'geobipy.src.classes.statistics.Histogram1D.Histogram1D'>
    Bins: 
    Cell Centres 
    Name: 
         Units: 
         Shape: (100,)
         Values: [-1.98 -1.94 -1.9  ...  1.9   1.94  1.98]
    Cell EdgesName: 
         Units: 
         Shape: (101,)
         Values: [-2.   -1.96 -1.92 ...  1.92  1.96  2.  ]
    Counts:
    Name: Frequency
         Units: 
         Shape: (100,)
         Values: [3 2 1 ... 4 1 0]
    Values are logged to base None
    Relative to: None




Basic manipulation
++++++++++++++++++

The StatArray contains other functions to perform basic array manipulations

These routines essentially wrap around numpy functions, 
but the result will have the same name and units, 
and if any prior or proposal are set, those will be carried through too.

1D example
__________


.. code-block:: default


    x = StatArray(-np.cumsum(np.arange(10.0)))
    print(x)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ -0.  -1.  -3. ... -28. -36. -45.]





.. code-block:: default



    print(x.insert(i=[0, 9], values=[999.0, 999.0]))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [999.  -0.  -1. ... -36. 999. -45.]





.. code-block:: default



    print(x.prepend(999.0))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [999.  -0.  -1. ... -28. -36. -45.]





.. code-block:: default



    print(x.prepend([998.0, 999.0]))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [998. 999.  -0. ... -28. -36. -45.]





.. code-block:: default



    print(x.append([998.0, 999.0]))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ -0.  -1.  -3. ... -45. 998. 999.]





.. code-block:: default



    print(x.resize(14))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [-0. -1. -3. ... -1. -3. -6.]





.. code-block:: default



    print(x.delete([5,8]))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ -0.  -1.  -3. ... -21. -28. -45.]





.. code-block:: default



    print(x.edges())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [  0.5  -0.5  -2.  ... -32.  -40.5 -49.5]





.. code-block:: default



    print(x.internalEdges())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ -0.5  -2.   -4.5 ... -24.5 -32.  -40.5]





.. code-block:: default



    print(x.firstNonZero())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1





.. code-block:: default



    print(x.lastNonZero())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    10





.. code-block:: default



    print(x.abs())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ 0.  1.  3. ... 28. 36. 45.]




2D example
__________


.. code-block:: default


    x = StatArray(np.asarray([[0, -2, 3],[3, 0, -1],[1, 2, 0]]))
    print(x)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[ 0 -2  3]
     [ 3  0 -1]
     [ 1  2  0]]





.. code-block:: default



    print(x.insert(i=0, values=4))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[ 4  4  4]
     [ 0 -2  3]
     [ 3  0 -1]
     [ 1  2  0]]





.. code-block:: default



    print(x.insert(i=[2, 3], values=5, axis=1))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[ 0 -2  5  3  5]
     [ 3  0  5 -1  5]
     [ 1  2  5  0  5]]





.. code-block:: default



    print(x.insert(i=2, values=[10, 11, 12], axis=1))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[ 0 -2 10  3]
     [ 3  0 11 -1]
     [ 1  2 12  0]]





.. code-block:: default



    print(x.prepend(999))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[999 999 999]
     [  0  -2   3]
     [  3   0  -1]
     [  1   2   0]]





.. code-block:: default



    print(x.prepend([999, 998, 997], axis=1))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[999   0  -2   3]
     [998   3   0  -1]
     [997   1   2   0]]





.. code-block:: default



    print(x.append([[999, 998, 997]]))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[  0  -2   3]
     [  3   0  -1]
     [  1   2   0]
     [999 998 997]]





.. code-block:: default



    print(x.resize([5,5]))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[ 0 -2  3  3  0]
     [-1  1  2  0  0]
     [-2  3  3  0 -1]
     [ 1  2  0  0 -2]
     [ 3  3  0 -1  1]]





.. code-block:: default



    print(x.delete(5))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ 0 -2  3 ...  1  2  0]





.. code-block:: default



    print(x.delete(2, axis=0))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[ 0 -2  3]
     [ 3  0 -1]]





.. code-block:: default



    print(x.firstNonZero(axis=0))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [1 0 0]





.. code-block:: default



    print(x.lastNonZero(axis=0))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [3 3 2]





.. code-block:: default



    print(x.firstNonZero(axis=1))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [1 0 0]





.. code-block:: default



    print(x.lastNonZero(axis=1))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [3 3 2]





.. code-block:: default



    print(x.abs())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[0 2 3]
     [3 0 1]
     [1 2 0]]




Plotting
++++++++

We can easily plot the StatArray with its built in plotting functions.
All plotting functions can take matplotlib keywords


.. code-block:: default


    # The simplest is to just plot the array

    Density = StatArray(np.random.randn(100),name="Density",units="$\frac{g}{cc}$")
    Time = StatArray(np.linspace(0, 100, Density.size), name='Time', units='s')
    Depth = StatArray(np.random.exponential(size=Density.size), name='Depth', units='m')










.. code-block:: default



    plt.figure()
    _ = Density.plot(linewidth=0.5, marker='x', markersize=1.0)




.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_002.png
    :class: sphx-glr-single-img





We can quickly plot a bar graph.


.. code-block:: default


    plt.figure()
    _ = Density.bar()





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_003.png
    :class: sphx-glr-single-img





We can scatter the contents of the StatArray if it is 1D


.. code-block:: default


    plt.figure()
    _ = Density.scatter(alpha=0.7)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_004.png
    :class: sphx-glr-single-img





Histogram Equalization
______________________

A neat trick with colourmaps is histogram equalization.
This approach forces all colours in the images to have an equal weight.
This distorts the colour bar, but can really highlight the lower and higher
ends of whatever you are plotting. Just add the equalize keyword!


.. code-block:: default


    plt.figure()
    _ = Density.scatter(alpha=0.7, equalize=True)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_005.png
    :class: sphx-glr-single-img





Take the log base(x) of the data

We can also take the data to a log, log10, log2, or a custom number!


.. code-block:: default


    plt.figure()
    _ = Density.scatter(alpha=0.7,edgecolor='k',log='e') # could also use log='e', log=2, log=x) where x is the base you require




.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_006.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Values <= 0.0 have been masked before taking their log




X and Y axes

We can specify the x axis of the scatter plot.


.. code-block:: default



    plt.figure()
    _ = Density.scatter(x=Time, alpha=0.7, edgecolor='k')





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_007.png
    :class: sphx-glr-single-img





Notice that I never specified the y axis, so the y axis defaulted to the values in the StatArray. 
In this case, any operations applied to the colours, are also applied to the y axis, e.g. log=10.  
When I take the values of Density to log base 10, because I do not specify the y plotting locations, those locations are similarly affected.

I can however force the y co-ordinates by specifying it as input. 
In the second subplot I explicitly plot distance on the y axis. 
In the first subplot, the y axis is the same as the colourbar.


.. code-block:: default



    plt.figure()
    ax1 = plt.subplot(211)
    Density.scatter(x=Time, alpha=0.7, edgecolor='k', log=10)
    plt.subplot(212, sharex=ax1)
    _ = Density.scatter(x=Time, y=Depth, alpha=0.7, edgecolor='k', log=10)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_008.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Values <= 0.0 have been masked before taking their log
    Values <= 0.0 have been masked before taking their log




Point sizes

Since the plotting functions take matplotlib keywords, I can also specify the size of each points.


.. code-block:: default



    s = np.ceil(100*(np.abs(np.random.randn(Density.size))))
    plt.figure()
    plt.tight_layout()
    ax1 = plt.subplot(211)
    Density.scatter(x=Time, y=Depth, s=s, alpha=0.7,edgecolor='k', sizeLegend=2)
    plt.subplot(212, sharex=ax1)
    #Density.scatter(x=Time, y=Depth, s=s, alpha=0.7,edgecolor='k', sizeLegend=[1.0, 100, 200, 300])
    v = np.abs(Density)+1.0
    _ = Density.scatter(x=Time, y=Depth, s=s, alpha=0.7,edgecolor='k', sizeLegend=[1.0, 100, 200, 300], log=10)







.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_009.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Values <= 0.0 have been masked before taking their log




Of course we can still take the log, or equalize the colour histogram


.. code-block:: default


    plt.figure()
    _ = Density.scatter(x=Time, y=Depth, s=s, alpha=0.7,edgecolor='k',equalize=True,log=10)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_010.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Values <= 0.0 have been masked before taking their log




Typically pcolor only works with 2D arrays. The StatArray has a pcolor method that will pcolor a 1D array


.. code-block:: default


    plt.figure()
    plt.subplot(221)
    Density.pcolor()
    plt.subplot(222)
    Density.pcolor(y=Time)
    plt.subplot(223)
    Density.pcolor(y=Time, flipY=True)
    plt.subplot(224)
    _ = Density.pcolor(y=Time, log=10, equalize=True)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_011.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Values <= 0.0 have been masked before taking their log




We can add grid lines, and add opacity to each element in the pcolor image

This is useful if the colour values need to be scaled by another variable e.g. variance.


.. code-block:: default



    plt.figure()
    plt.subplot(121)
    Density.pcolor(grid=True, cmap='jet')
    plt.subplot(122)
    a = np.linspace(1.0, 0.0, Density.size)
    _ = Density.pcolor(grid=True, alpha=a, cmap='jet')





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_012.png
    :class: sphx-glr-single-img





We can plot a histogram of the StatArray


.. code-block:: default


    plt.figure()
    _ = Density.hist(100)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_013.png
    :class: sphx-glr-single-img





We can write the StatArray to a HDF5 file.  HDF5 files are binary files that can include compression.  They allow quick and easy access to parts of the file, and can also be written to and read from in parallel!


.. code-block:: default


    with h5py.File('1Dtest.h5','w') as f:
        Density.toHdf(f,'test')









We can then read the StatArray from the file
Here x is a new variable, that is read in from the hdf5 file we just wrote.


.. code-block:: default


    x = hdfRead.readKeyFromFiles('1Dtest.h5','/','test')
    print('x has the same values as Density? ',np.all(x == Density))
    x[2] = 5.0 # Change one of the values in x
    print('x has its own memory allocated (not a reference/pointer)? ',np.all(x == Density) == False)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    x has the same values as Density?  True
    x has its own memory allocated (not a reference/pointer)?  True




We can also define a 2D array


.. code-block:: default


    Density = StatArray(np.random.randn(50,100),"Density","$\frac{g}{cc}$")
    Density.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Name: Density
         Units: $\frac{g}{cc}$
         Shape: (50, 100)
         Values: [[ 0.93086274 -0.23452262 -0.39548999 ... -1.77457573 -0.59190863
       0.38720698]
     [ 0.90731836  1.31181291 -0.65440119 ... -1.14402084  0.9666266
       0.35062989]
     [-0.00475169  1.05844704 -0.58677181 ... -0.42828661  0.13430066
       0.12910916]
     ...
     [ 1.95783267 -0.58488051 -0.45933394 ...  0.38304206 -0.34271281
      -0.11256415]
     [-0.12176328  2.10456355 -0.10022661 ... -1.39656847 -0.54014559
       1.20310483]
     [-1.0321643   0.49540948 -0.44440532 ...  2.30997649 -0.06236417
      -0.17748218]]





The StatArray Class's functions work whether it is 1D or 2D

We can still do a histogram


.. code-block:: default


    plt.figure()
    _ = Density.hist()





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_014.png
    :class: sphx-glr-single-img





And we can use pcolor to plot the 2D array


.. code-block:: default


    plt.figure()
    _ = Density.pcolor()





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_015.png
    :class: sphx-glr-single-img





The StatArray comes with extra plotting options

Here we specify the x and y axes for the 2D array using two other 1D StatArrays


.. code-block:: default


    plt.figure()
    x = StatArray(np.arange(101),name='x Axis',units = 'mm')
    y = StatArray(np.arange(51),name='y Axis',units = 'elephants')
    _ = Density.pcolor(x=x, y=y)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_016.png
    :class: sphx-glr-single-img





We can plot using a log10 scale, in this case, we have values that are less
than or equal to 0.0.  Plotting with the log option will by default mask any
of those values, and will let you know that it has done so!


.. code-block:: default


    plt.figure()
    _ = Density.pcolor(x=x,y=y,log=2)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_017.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Values <= 0.0 have been masked before taking their log




A neat trick with colourmaps is histogram equalization.
This approach forces all colours in the image to have an equal amount.
This distorts the colours, but can really highlight the lower and higher
ends of whatever you are plotting


.. code-block:: default


    plt.figure()
    _ = Density.pcolor(x=x, y=y, equalize=True)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_018.png
    :class: sphx-glr-single-img





We can equalize the log10 plot too :)


.. code-block:: default


    plt.figure()
    _ = Density.pcolor(x=x,y=y,equalize=True, log=10)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_019.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Values <= 0.0 have been masked before taking their log




We can add opacity to each pixel in the image


.. code-block:: default


    a = StatArray(np.random.random(Density.shape), 'Opacity from 0.0 to 1.0')










.. code-block:: default



    plt.figure()
    ax1 = plt.subplot(131)
    ax = Density.pcolor(x=x, y=y, flipY=True, linewidth=0.1, noColorbar=True)
    plt.subplot(132, sharex=ax1, sharey=ax1)
    ax = Density.pcolor(x=x, y=y, alpha=a, flipY=True, linewidth=0.1, noColorbar=True)
    plt.subplot(133, sharex=ax1, sharey=ax1)
    _ = a.pcolor(x=x, y=y, flipY=True)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_020.png
    :class: sphx-glr-single-img





If the array potentially has a lot of white space around the edges, we can trim the image


.. code-block:: default


    Density[:10, :] = 0.0
    Density[-10:, :] = 0.0
    Density[:, :10] = 0.0
    Density[:, -10:] = 0.0
    plt.figure()
    plt.subplot(121)
    Density.pcolor()
    plt.subplot(122)
    _ = Density.pcolor(trim=0.0)





.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_021.png
    :class: sphx-glr-single-img





Create a stacked area plot of a 2D StatArray


.. code-block:: default


    A = StatArray(np.abs(np.random.randn(13,100)), name='Variable', units="units")
    x = StatArray(np.arange(100),name='x Axis',units = 'mm')
    plt.figure()
    ax1 = plt.subplot(211)
    A.stackedAreaPlot(x=x, axis=1)
    plt.subplot(212, sharex=ax1)
    _ = A.stackedAreaPlot(x=x, i=np.s_[[1,3,4],:], axis=1, labels=['a','b','c'])



.. image:: /examples/Statistics/images/sphx_glr_plot_StatArray_022.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  3.509 seconds)


.. _sphx_glr_download_examples_Statistics_plot_StatArray.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_StatArray.py <plot_StatArray.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_StatArray.ipynb <plot_StatArray.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
