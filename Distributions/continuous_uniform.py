from decimal import Decimal


class ContinuousUniform:
    @staticmethod
    def cdf(a: Decimal, b: Decimal, x: Decimal) -> Decimal:
        """
        :param a: Decimal, lower limit
        :param b: Decimal, upper limit
        :param x: Decimal
        :return:
        """
        return (x - a) / (b - a)

    @staticmethod
    def mean(a: Decimal, b: Decimal) -> Decimal:
        """
        :param a: Decimal, lower limit
        :param b: Decimal, upper limit
        :return:
        """
        return (a + b) / 2

    @staticmethod
    def variance(a: Decimal, b: Decimal) -> Decimal:
        """
        :param a: Decimal, lower limit
        :param b: Decimal, upper limit
        :return:
        """
        return pow(b - a, 2) / 12
