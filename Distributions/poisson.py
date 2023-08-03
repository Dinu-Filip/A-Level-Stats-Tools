from decimal import Decimal
from scipy.stats import poisson
from distribution_templates import DistributionTemplates
import math

class Poisson:
    poisson_pmf = """P(X = x) = \\frac{e^{-\\lambda} \\lambda^x}{x!}"""
    poisson_cdf = """P(X \\leq x) = \\sum_{i = 1}^{x} \\frac{e^{-\\lambda} \\lambda^i}{i!}"""
    poisson_mean = """E(X) = \\lambda"""
    poisson_variance = """\\sigma^2 = \\lambda"""
    poisson_skewness = """\\gamma = \\frac{1}{\\sqrt{\\lambda}}"""

    @staticmethod
    def pmf(rate: Decimal, x: int, dp: int) -> dict:
        """
        :param dp: int, number of decimal places to round to
        :param rate: Decimal, rate parameter
        :param x:    int, number of events in time frame
        :return:
        """
        res = round(poisson.pmf(x, rate), dp)
        params = {
            "rate, \\lambda": rate,
            "x": x
        }
        method = f"""
        Substituting into the equation:
        P(X = {x}) = \\frac{{e^{-1 * rate} {rate}^{x}}}{{{x}!}} = {res}
        """
        return {"res": res, "full_method": DistributionTemplates.method_template("Poisson",
                                                                          "probability mass function, PMF",
                                                                                 Poisson.poisson_pmf,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def cdf(rate: Decimal, lower_x: Decimal, upper_x: Decimal, dp: int) -> dict:
        """
        :param lower_x: Decimal, lower limit of interval
        :param upper_x: Decimal, upper limit of interval
        :param dp: int, number of decimal places to round to
        :param rate: Decimal, rate parameter
        :param x:    int, maximum number of events in time frame
        :return:     Decimal, probability of x or fewer events occuring in time frame
        """
        upper_cdf = 0
        for _ in range(0, upper_x + 1):
            upper_cdf += Decimal(poisson.pmf(_, rate))
        lower_cdf = 0
        for _ in range(0, lower_x):
            lower_cdf += Decimal(poisson.pmf(_, rate))
        res = round(upper_cdf - lower_cdf, dp)
        params = {
            "rate, \\lambda": rate,
            "x_1": lower_x,
            "x_2": upper_x
        }
        method = f"""
        We know that P(a \\leq X \\leq b) = P(X \\leq b) - P(X < a)
        Substituting into the formula:
        P({lower_x} \\leq X \\leq {upper_x}) &= P(X \\leq {upper_x}) - P(X < {lower_x})
        &= \\sum_{{i = 1}}^{{{upper_x}}} P(X = i) - \\sum_{{i = 1}}^{{{lower_x - 1}}} P(X = i)
        &= {upper_cdf} - {lower_cdf}
        &= {res}
        """
        return {"res": res, "full_method": DistributionTemplates.method_template("Poisson",
                                                                          "cumulative distribution function",
                                                                                 Poisson.poisson_cdf,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def mean(rate: Decimal, dp: int) -> dict:
        """
        :param dp: int, number of decimal places to round to
        :param rate: Decimal, rate parameter
        :return:
        """
        res = round(rate, dp)
        params = {
            "rate, \\lambda": rate
        }
        method = f"""
        Substituting into the formula:
        E(X) = {rate}
        """
        return {"res": res, "full_method": DistributionTemplates.method_template("Poisson",
                                                                          "mean",
                                                                                 Poisson.poisson_mean,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def variance(rate: Decimal, dp: int) -> dict:
        """
        :param dp: int, number of decimal places to round to
        :param rate: Decimal, rate parameter
        :return:
        """
        res = round(rate, dp)
        params = {
            "rate, \\lambda": rate
        }
        method = f"""
        Substituting into the formula:
        \\sigma^2 - {rate}
        """
        return {"res": res, "full_method": DistributionTemplates.method_template("Poisson",
                                                                          "variance",
                                                                                 Poisson.poisson_variance,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def skewness(rate: Decimal, dp: int) -> dict:
        """
        :param rate: Decimal, rate parameter
        :param dp: int, number of decimal places to round to
        :return:
        """
        res = round(1 / math.sqrt(rate), dp)
        params = {
            "rate, \\lambda": rate
        }
        method = f"""
        Substituting into the formula:
        \\gamma = \\frac{{1}}{{\\sqrt{{{rate}}}}} = {res}
        """
        return {"res": res, "full_method": DistributionTemplates.method_template("Poisson",
                                                                          "skewness",
                                                                                 Poisson.poisson_skewness,
                                                                                 params,
                                                                                 method)}
