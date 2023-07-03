from decimal import Decimal
from scipy.stats import norm


class Normal:
    @staticmethod
    def cdf(mean: Decimal, sd: Decimal, x: Decimal):
        """
        :param mean: Decimal, mean of normal distribution
        :param sd:   Decimal, standard deviation
        :param x:    Decimal, value of random variable
        :return:
        """
        return norm.cdf(x, mean, sd)
