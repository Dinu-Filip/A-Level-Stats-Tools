from decimal import Decimal
from scipy.stats import chi2
from Distributions.distribution_templates import DistributionTemplates
import math


class ChiSquared:
    chi_squared_mean = r"E(X) = k"
    chi_squared_variance = r"\sigma^2 = 2k"
    chi_squared_skewness = r"\gamma = \sqrt{\frac{8}{k}}"

    @staticmethod
    def cdf_calc(lower_x: float, upper_x: float, deg_free: int, dp: int) -> Decimal:
        """
        :param x: Decimal, value of random variable
        :param deg_free: int, number of degrees of freedom
        :return: Decimal, value cdf of distribution
        """
        return round(chi2.cdf(upper_x, deg_free) - chi2.cdf(lower_x, deg_free), dp)

    @staticmethod
    def cdf(lower_x: str, upper_x: str, deg_free: str, dp: str) -> dict:
        if upper_x == "":
            res = round(1 - ChiSquared.cdf_calc(0, float(lower_x), float(deg_free), int(dp) + 2), int(dp))
        else:
            if lower_x == "":
                lower_x = 0
            res = ChiSquared.cdf_calc(float(lower_x), float(upper_x), int(deg_free), int(dp))

        method = r"""The formula for the CDF of the chi-squared distribution uses the gamma and incomplete gamma 
        functions, which are outside the scope of the A Level specification. """
        return {"res": str(res), "method": method}

    @staticmethod
    def mean(deg_free: str):
        params = {"k": deg_free}
        method = r"""Substituting into the formula: E(X) = #k# = #k#"""
        return {"res": deg_free, "method": DistributionTemplates.method_template("chi-squared",
                                                                                 "mean",
                                                                                 ChiSquared.chi_squared_mean,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def variance(deg_free: str):
        params = {"k": deg_free, "res": str(2 * int(deg_free))}
        method = r"""Substituting into the formula: \sigma^2 = 2 \times #k# = #res#"""
        return {"res": str(2 * int(deg_free)), "method": DistributionTemplates.method_template("chi-squared",
                                                                                               "variance",
                                                                                               ChiSquared.chi_squared_variance,
                                                                                               params,
                                                                                               method)}

    @staticmethod
    def skewness(deg_free: str, dp: str):
        res = str(round(math.sqrt(8 / int(deg_free)), int(dp)))
        params = {"k": deg_free, "res": res}
        method = r"""Substituting into the formula: \gamma = \sqrt{\frac{8}{#k#} = #res#"""
        return {"res": res, "method": DistributionTemplates.method_template("chi-squared",
                                                                            "skewness",
                                                                            ChiSquared.chi_squared_skewness,
                                                                            params,
                                                                            method)}

    @staticmethod
    def inverse_chi_squared(deg_free: str, P: str, dp: str) -> dict:
        res = str(round(chi2.ppf(Decimal(P), df=int(deg_free)), int(dp)))
        return {"res": res}

    @staticmethod
    def chi_squared_points(deg_free: str) -> dict:
        min_x = round(chi2.ppf(0.999, df=int(deg_free)), 2)
        max_x = round(chi2.ppf(0.001, df=int(deg_free)), 2)
        points = {}
        for x in range(min_x, max_x, (max_x - min_x) / 100):
            points[x] = chi2.pdf(x, df=int(deg_free))
        return points


if __name__ == "__main__":
    print(ChiSquared.cdf("2", "6", "2", "5"))
