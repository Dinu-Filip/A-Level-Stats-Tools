from decimal import *
from decimal import Decimal
import math

class Binomial:
    @staticmethod
    def pmf(n: int, p: Decimal, x: int) -> Decimal:
        """
        :param n:  int, total number of trials
        :param p:  Decimal, probability of success
        :param x:  int, number of successful trials
        :return:   Decimal, value of pmf at particular x
        """
        if x > n:
            raise ValueError("The number of successes must be less than or equal to the total number of trials")
        return Decimal(math.comb(n, x)) * Decimal(pow(p, x)) * Decimal(pow(1 - p, n - x))

    @staticmethod
    def cdf(n: int, p: Decimal, lower_x: int, upper_x: int) -> Decimal:
        """
        :param n:  int, total number of trials
        :param p:  Decimal, probability of success
        :param lower_x: int, lower limit of interval
        :param upper_x: int, upper limit of interval
        :return:   Decimal, value of cdf between limits
        """
        if upper_x > n:
            raise ValueError("The number of successes must be less than or equal to the total number of trials")
        elif upper_x <= lower_x:
            raise ValueError("The upper limit must be greater than the lower limit")
        res = 0
        for _ in range(0, upper_x + 1):
            res += Binomial.pmf(n, p, _)
        for _ in range(0, lower_x + 1):
            res -= Binomial.pmf(n, p, _)
        return res

    @staticmethod
    def mean(n: int, p: Decimal) -> Decimal:
        """
        :param n:  int, total number of trials
        :param p:  Decimal, probability of success
        :return:   Decimal, mean of binomial distribution
        """
        return n * p

    @staticmethod
    def variance(n: int, p: Decimal) -> Decimal:
        """
        :param n: int, total number of trials
        :param p: Decimal, probability of success
        :return:  Decimal, variance of binomial distribution
        """
        return n * p * (1 - p)

    @staticmethod
    def skewness(n: int, p: Decimal) -> Decimal:
        """
        :param n: int, total number of trials
        :param p: Decimal, probability of success
        :return:  Decimal, skewness of binomial distribution
        """
        standard_dev = math.sqrt(n * p * (1 - p))
        return (1 - 2 * p) / standard_dev

