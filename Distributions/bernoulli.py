from decimal import Decimal


class Bernoulli:
    @staticmethod
    def pmf(p: Decimal, x: int) -> Decimal:
        """
        :param p:  Decimal, probability of success
        :param x:  Integer, 1 or 0
        :return:   Decimal, probability of outcome being x
        """
        if p < 0:
            raise ValueError("The probability of success cannot be less than 0")
        if x == 0:
            return 1 - p
        elif x == 1:
            return p
        else:
            raise ValueError("The outcome must be either 0 or 1")

    @staticmethod
    def mean(p: Decimal) -> Decimal:
        """
        :param p:  Decimal, probability of success
        :return:   Decimal, mean of distribution
        """
        return p

    @staticmethod
    def variance(p: Decimal) -> Decimal:
        """
        :param p:  Decimal, probability of success
        :return:   Decimal, variance of distribution
        """
        return p * (1 - p)
