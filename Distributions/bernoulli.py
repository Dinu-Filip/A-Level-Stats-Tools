from decimal import Decimal
from distribution_templates import DistributionTemplates


class Bernoulli:
    pmf_formula = "P(X=x)=p^x(1-p)^{1-x}"
    mean_formula = "p"
    var_formula = "p(1-p)"

    @staticmethod
    def pmf(p: Decimal, x: int, dp: int) -> dict:
        """
        :param dp: int, number of decimal places
        :param p:  Decimal, probability of success
        :param x:  Integer, 1 or 0
        :return:   Decimal, probability of outcome being x
        """
        res = None
        if x == 0:
            res = round(1 - p, dp)
        elif x == 1:
            res = round(p, dp)
        else:
            raise ValueError("The outcome must be either 0 or 1")
        params = {"p": str(p), "x": str(x)}
        method = f"""\\begin{{align*}}
                    P(X = {x}) &= {p}^{x}(1 - {p})^{{1-{x}}} \\ &= {res}
                    \\end{{align*}}"""
        return {"res": res,
                "method": DistributionTemplates.discrete("Bernoulli",
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
        params = {"p": str(p)}
        method = f"""\\begin{{align*}}
                    E(X) = p = {res}
                    \\end{{align*}}"""
        return {"res": res,
                "method": DistributionTemplates.discrete("Bernoulli",
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
        params = {"p": str(p)}
        method = f"""\\begin{{align*}}
                    \sigma^2 = {p}(1-{p}) = {res}
                    \\end{{align*}}"""
        return {"res": res,
                "method": DistributionTemplates.discrete("Bernoulli",
                                                             "variance",
                                                             Bernoulli.var_formula,
                                                             params,
                                                             method)}

