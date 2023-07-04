from decimal import Decimal
from scipy.stats import chi2

class ChiSquared:
    @staticmethod
    def cdf(x: Decimal, deg_free: int) -> Decimal:
        """
        :param x: Decimal, value of random variable
        :param deg_free: int, number of degrees of freedom
        :return: Decimal, value cdf of distribution
        """
        return chi2.cdf(x, deg_free)