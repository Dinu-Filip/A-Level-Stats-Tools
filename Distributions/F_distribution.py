from decimal import Decimal
from scipy.stats import f


class FDistribution:
    @staticmethod
    def cdf(x: Decimal, df_1: int, df_2: int) -> Decimal:
        """
        :param x: Decimal, value of random variable
        :param df_1: int, number of degrees of freedom of first chi-square distribution
        :param df_2: int, number of degrees of freedom of second chi-square distribution
        :return: Decimal, value of cdf
        """
        return f.cdf(x, df_1, df_2)
