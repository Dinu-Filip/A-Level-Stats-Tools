import math
from decimal import Decimal

class Geometric:
    @staticmethod
    def pmf(p: Decimal, x: int) -> Decimal:
        """
        :param p: Decimal, probability of success
        :param x: int, index of first successful trial
        :return:  Decimal, pmf of geometric distribution
        """
        return p * pow(1 - p, x - 1)

    @staticmethod
    def cdf(p: Decimal, x: int) -> Decimal:
        """
        :param p: Decimal, probability of success
        :param x: int, index of first successful trial
        :return:  Decimal, cdf of geometric distribution
        """
        return 1 - pow(1 - p, x)

    @staticmethod
    def mean(p: Decimal) -> Decimal:
        """
        :param p: Decimal, probability of success
        :return:
        """
        return 1 / p

    @staticmethod
    def variance(p: Decimal):
        """
        :param p: Decimal, probability of success
        :return:
        """
        return (1 - p) / (p ** 2)

    @staticmethod
    def skewness(p: Decimal):
        """
        :param p: Decimal, probability of success
        :return:
        """
        return (2 - p) / (math.sqrt(1 - p))