""" @NormalDistribution
Module defining a normal distribution with statistical procedures
"""
from numpy import array, exp, linspace, log, size, squeeze, hstack
from .baseDistribution import baseDistribution
from scipy.stats import chi2
from ...base import plotting as cP
from ..core.DataArray import DataArray


class ChiSquared(baseDistribution):
    """Univariate normal distribution

    Normal(mean, variance)

    Parameters
    ----------
    mean : numpy.float
        The mean of the distribution
    variance : numpy.float
        The variance of the distribution

    """
    def __init__(self, df, prng=None, **kwargs):
        """Instantiate a Normal distribution """
        # assert size(mean) == 1, 'Univariate Normal mean must have size = 1'
        # assert size(variance) == 1, 'Univariate Normal variance must have size = 1'
        super().__init__(prng)
        self.df = df

    @property
    def address(self):
        return hstack([hex(id(self)), hex(id(self.df))])

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @property
    def log(self):
        return False

    @property
    def ndim(self):
        return 1

    @property
    def multivariate(self):
        return False

    @property
    def variance(self):
        return 2 * self.df

    def cdf(self, x):
        """ For a realization x, compute the probability """
        return DataArray(chi2.cdf(x=x, df=self.df), "Cumulative Density")


    def __deepcopy__(self, memo={}):
        """Create a deepcopy

        Returns
        -------
        out
            Normal

        """
        return ChiSquared(self.df, self.prng)


    def derivative(self, x, moment):
        assert 0 <= moment <= 1, ValueError("Must have 0 <= moment < 2")


        if moment == 0:
            return ((x - self._mean) / self.variance) * self.probability(x)
        else:
            return (0.5 / self.variance**2.0) * ((x - self._mean)**2.0 - self.variance) * self.probability(x)


    def rng(self, size=1):
        """Generate random numbers

        Parameters
        ----------
        N : int or sequence of ints
            Number of samples to generate

        Returns
        -------
        out
            numpy.ndarray

        """
        size = (size, self.ndim)
        values = squeeze(self.prng.chisquare(df=self.df, size=size))
        return exp(values) if self.log else values

    def plot_pdf(self, log=False, **kwargs):
        bins = self.bins()
        t = r"$\tilde{\Chi}^{2}(k="+str(self.df)+")$"

        return cP.plot(bins, self.probability(bins, log=log), label=t, **kwargs)

    def ppf(self, alpha):
        return chi2.ppf(alpha, df=self.df)

    def probability(self, x, log):
        """ For a realization x, compute the probability """

        if self.log:
            x = log(x)

        if log:
            return DataArray(chi2.logpdf(df=self.df, x=x), "Probability Density")
        else:
            return DataArray(chi2.pdf(df=self.df, x=x), "Probability Density")

    @property
    def summary(self):
        msg =  "{}\n".format(type(self).__name__)
        msg += '    Degrees of freedom:{}\n'.format(self.df)
        return msg

#    def hdfName(self):
#        """ Create the group name for an HDF file """
#        return('Distribution("Normal",0.0,1.0)')
#
#    def toHdf(self, h5obj, myName):
#        """ Write the object to an HDF file """
#        grp = h5obj.create_group(myName)
#        grp.attrs["repr"] = self.hdfName()
#        grp.create_dataset('mean', data=self.mean)
#        grp.create_dataset('variance', data=self.variance)
#
#    def createHdf(self, parent, myName):
#        """ Create the hdf group metadata in file """
#        grp = parent.create_group(myName)
#        grp.attrs["repr"] = self.hdfName()
#        grp.create_dataset('mean', (1,), dtype=self.mean.dtype)
#        grp.create_dataset('variance', (1,), dtype=self.variance.dtype)
#
#    def writeHdf(self, parent, myName, create=True):
#        """ Write the StatArray to an HDF object
#        parent: Upper hdf file or group
#        myName: object hdf name. Assumes createHdf has already been called
#        """
#        # create a new group inside h5obj
#        if (create):
#            self.createHdf(parent, myName)
#
#        grp = parent.get(myName)
#        writeNumpy(self.mean,grp,'mean')
#        writeNumpy(self.variance,grp,'variance')
#
#    def fromHdf(self, h5grp):
#        """ Reads the Uniform Distribution from an HDF group """
#        T1 = array(h5grp.get('mean'))
#        T2 = array(h5grp.get('variance'))
#        return MvNormal(T1, T2)

    def bins(self, nBins = 99, nStd=4.0):
        """ Discretizes a range given the mean and variance of the distribution """
        return DataArray(linspace(0.5, nStd*self.df, nBins+1))
