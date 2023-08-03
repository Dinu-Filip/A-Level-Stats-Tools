from decimal import Decimal
from distribution_templates import DistributionTemplates


class Bernoulli:
    pmf_formula = r"P(X=x)=p^x(1-p)^{1-x}"
    mean_formula = r"p"
    var_formula = r"p(1-p)"

    @staticmethod
    def pmf(p: Decimal, x: int, dp: int) -> dict:
        """
        :param dp: int, number of decimal places
        :param p:  Decimal, probability of success
        :param x:  Integer, 1 or 0
        :return:   Decimal, probability of outcome being x
        """
        if x == 0:
            res = round(1 - p, dp)
        else:
            res = round(p, dp)
        params = {"p": str(p), "x": str(x), "res": res}
        method = r"""P(X = #x#) = #p#^#x#(1 - #p#)^{1-#x#} = #res#"""
        return {"res": res,
                "method": DistributionTemplates.method_template("Bernoulli",
                                                                "probability mass function, (PMF)",
                                                                Bernoulli.pmf_formula,
                                                                params,
                                                                method)}

    @staticmethod
    def mean(p: Decimal, dp: int) -> dict:
        """
        :param dp: int, number of decimal places
        :param p:  Decimal, probability of success
        :return:   Decimal, mean of distribution
        """
        res = round(p, dp)
        params = {"p": str(p), "res": res}
        method = r"""E(X) = p = #res#"""
        return {"res": res,
                "method": DistributionTemplates.method_template("Bernoulli",
                                                                "mean",
                                                                Bernoulli.mean_formula,
                                                                params,
                                                                method)}

    @staticmethod
    def variance(p: Decimal, dp: int) -> dict:
        """
        :param dp: int, number of decimal places to round
        :param p:  Decimal, probability of success
        :return:   Decimal, variance of distribution
        """
        res = round(p * (1 - p), dp)
        params = {"p": str(p), "res": res}
        method = r"""\sigma^2 = {#p#}(1-#p#}) = #res#"""
        return {"res": res,
                "method": DistributionTemplates.method_template("Bernoulli",
                                                                "variance",
                                                                Bernoulli.var_formula,
                                                                params,
                                                                method)}
