from decimal import Decimal

class Exponential:
    @staticmethod
    def cdf(rate: Decimal, x: Decimal) -> Decimal:
        """
        :param rate: Decimal, number of events per time frame
        :param x:    Decimal, time up to event
        :return:
        """
        return 1 - Decimal.exp(-1 * rate * x)

    @staticmethod
    def mean(rate: Decimal) -> Decimal:
        """
        :param rate: Decimal, number of events per time frame
        :return:
        """
        return 1 / rate

    @staticmethod
    def variance(rate: Decimal) -> Decimal:
        """
        :param rate: Decimal, number of events per time frame
        :return:
        """
        return 1 / pow(rate, 2)

    @staticmethod
    def skewness() -> int:
        return 2