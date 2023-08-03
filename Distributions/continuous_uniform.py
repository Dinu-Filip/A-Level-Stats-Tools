from decimal import Decimal
from distribution_templates import DistributionTemplates


class ContinuousUniform:
    uniform_cdf = r"P(X \leq x) = \frac{x - a}{x - b}"
    uniform_pdf = r"f(x) = \frac{1}{b- a}, \ a < x < b"
    uniform_mean = r"\mu = \frac{a+b}{2}"
    uniform_variance = r"\sigma^2 = \frac{(b-a)^2}{12}"

    @staticmethod
    def cdf(a: Decimal, b: Decimal, x: Decimal, dp: int) -> Decimal:
        """
        :param a: Decimal, lower limit
        :param b: Decimal, upper limit
        :param x: Decimal
        :return:
        """
        params = {
            "a": a,
            "b": b,
            "x": x
        }
        res = round((x - a) / (b - a), dp)
        method = r"""Substituting into the formula:
        P(X \leq #x#) = \frac{x - #a#}{x - #b#} = #res#"""
        return {"res": res, "method": DistributionTemplates.method_template("continuous uniform",
                                                                            "cumulative distribution function",
                                                                            ContinuousUniform.uniform_cdf,
                                                                            params,
                                                                            method)}

    @staticmethod
    def mean(a: Decimal, b: Decimal, dp: int) -> Decimal:
        """
        :param a: Decimal, lower limit
        :param b: Decimal, upper limit
        :return:
        """
        res = round((a + b) / 2, dp)
        params = {
            "a": a,
            "b": b
        }
        method = r"""Substituting into the formula:
        \mu = \frac{#a# + #b#}{2} = #res#"""
        return {"res": res, "method": DistributionTemplates.method_template("continuous uniform",
                                                                            "mean",
                                                                            ContinuousUniform.uniform_mean,
                                                                            params,
                                                                            method)}

    @staticmethod
    def variance(a: Decimal, b: Decimal, dp: int) -> Decimal:
        """
        :param a: Decimal, lower limit
        :param b: Decimal, upper limit
        :return:
        """
        res = round(pow(b - a, 2) / 12, dp)
        params = {
            "a": a,
            "b": b
        }
        method = r"""Substituting into the formula:
        \sigma^2 = \frac{(#b# - #a#)^2}{12} = #res#"""
        return {"res": res, "method": DistributionTemplates.method_template("continuous uniform",
                                                                            "variance",
                                                                            ContinuousUniform.uniform_variance,
                                                                            params,
                                                                            method)}
