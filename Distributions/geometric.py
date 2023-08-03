import math
from decimal import Decimal
from distribution_templates import DistributionTemplates


class Geometric:
    geometric_pmf = """P(X = x) = p(1 - p)^{x-1}"""
    geometric_cdf = """P(X \\leq x) = 1 - (1 - p)^x"""
    geometric_mean = """E(X) = \\frac{1}{p}"""
    geometric_variance = """\\sigma^2 = \\frac{1 - p}{p^2}"""
    geometric_skewness = """\\gamma = \\frac{2 - p}{\\sqrt{1 - p}}"""

    @staticmethod
    def pmf(p: Decimal, x: int, dp: int) -> dict:
        """
        :param dp: int, number of decimal places to round to
        :param p: Decimal, probability of success
        :param x: int, index of first successful trial
        :return:  Decimal, pmf of geometric distribution
        """
        res = round(p * pow(1 - p, x - 1), dp)
        params = {
            "p": p,
            "x": x
        }
        method = f"""
        Substituting into the formula:
        P(X = {x}) = {p} \\times (1 - {p})^{{{x - 1}}} = {res}
        """
        return {"res": res, "full_method": DistributionTemplates.method_template("geometric",
                                                                          "probability mass function, (PMF)",
                                                                                 Geometric.geometric_pmf,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def cdf(p: Decimal, dp: int, lower_x: Decimal, upper_x: Decimal) -> dict:
        """
        :param dp: int, number of decimal places to round to
        :param p: Decimal, probability of success
        :param x: int, index of first successful trial
        :return:  Decimal, cdf of geometric distribution
        """
        upper_cdf = round(1 - pow(1 - p, upper_x), dp)
        lower_cdf = round(1 - pow(1 - p, lower_x - 1), dp)
        res = upper_cdf - lower_cdf

        params = {
            "p": p,
            "x_1": lower_x,
            "x_2": upper_x
        }
        method = f"""
        We know that P(a \\leq X \\leq b) = P(X \\leq b) - P(X < a)
        So P({lower_x} \\leq X \\leq {upper_x}) &= P(X \\leq {upper_x}) - P(X < {lower_x})
        &= \\sum_{{i = 1}}^{{{upper_x}}} P(X = i) - \\sum_{{i = 1}}^{{{lower_x - 1}}} P(X = i)
        &= {upper_cdf} - {lower_cdf}
        &= {res}
        """
        return {"res": res, "full_method": DistributionTemplates.method_template("geometric",
                                                                          "cumulative distribution function, CDF",
                                                                                 Geometric.geometric_cdf,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def mean(p: Decimal, dp: int) -> dict:
        """
        :param dp: int, number of decimal places to round to
        :param p: Decimal, probability of success
        :return:
        """
        res = round(1 / p, dp)
        params = {
            "p": p,
        }
        method = f"""
        Substituting into the formula:
        E(X) = \\frac{{1}}{{{p}}} = {res}
        """
        return {"res": res, "full_method": DistributionTemplates.method_template("geometric",
                                                                          "mean",
                                                                                 Geometric.geometric_mean,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def variance(p: Decimal, dp: int) -> dict:
        """
        :param dp: int, number of decimal places to round to
        :param p: Decimal, probability of success
        :return:
        """
        res = round((1 - p) / (p ** 2), dp)
        params = {
            "p": p
        }
        method = f"""
        Substituting into the formula:
        \\sigma^2 = \\frac{{1 - {p}}}{{{p}^2}} = {res} 
        """
        return {"res": res, "full_method": DistributionTemplates.method_template("geometric",
                                                                          "variance",
                                                                                 Geometric.geometric_variance,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def skewness(p: Decimal, dp: int) -> dict:
        """
        :param dp: int. number of decimal places to round to
        :param p: Decimal, probability of success
        :return:
        """
        res = round((2 - p) / math.sqrt(1 - p), dp)
        params = {
            "p": p
        }
        method = f"""
        Substituting into the formula:
        \\gamma = \\frac{{2 - {p}}}{{\\sqrt{{1 - {p}}}}} = {res}
        """
        return {"res": res, "full_method": DistributionTemplates.method_template("geometric",
                                                                          "skewness",
                                                                                 Geometric.geometric_skewness,
                                                                                 params,
                                                                                 method)}
