from decimal import Decimal
from scipy.stats import poisson
import math

class Poisson:
    @staticmethod
    def pmf(rate: Decimal, x: int) -> Decimal:
        """
        :param rate: Decimal, rate parameter
        :param x:    int, number of events in time frame
        :return:
        """
        return Decimal(poisson.pmf(x, rate))

    @staticmethod
    def cdf(rate: Decimal, x: int) -> Decimal:
        """
        :param rate: Decimal, rate parameter
        :param x:    int, maximum number of events in time frame
        :return:     Decimal, probability of x or fewer events occuring in time frame
        """
        res = 0
        for _ in range(0, x + 1):
            res += Poisson.pmf(rate, _)
        return res

    @staticmethod
    def mean(rate: Decimal) -> Decimal:
        return rate

    @staticmethod
    def variance(rate) -> Decimal:
        return rate

    @staticmethod
    def skewness(rate) -> Decimal:
        return 1 / math.sqrt(rate)
