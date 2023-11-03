from decimal import Decimal
import math
from Distributions.distribution_templates import DistributionTemplates
from scipy.stats import binom


class Binomial:
    pmf_formula = r"P(X = x) = \binom{n}{x} p^x (1-p)^{n-x}"
    cdf_formula = r"P(X \leq x) = \sum_{i = 1}^{x} P(X=x)"
    mean_formula = r"E(X) = n \times p"
    var_formula = r"\sigma^2 = np(1-p)"
    skew_formula = r"\gamma = \frac{1 - 2p}{\sqrt{np(1-p)}}"

    @staticmethod
    def pmf_calc(n: int, p: Decimal, x: int, dp: int) -> Decimal:
        #
        # pmf_calc used to calculate raw pmf value
        # Used by cdf and graphing algorithms in frontend
        #
        return round(binom.pmf(x, n=n, p=float(p)), int(dp))

    @staticmethod
    def pmf(n_val: str, p_val: str, x_val: str, dp: str) -> dict:
        comb = str(math.comb(int(n_val), int(x_val)))
        res = str(Binomial.pmf_calc(int(n_val), Decimal(p_val), int(x_val), int(dp)))
        params = {
            "n": n_val,
            "p": p_val,
            "x": x_val,
            "comb": comb,
            "res": res
        }
        method = r"""First calculate the binomial coefficient: $\binom{#n#}{#x#} = \frac{#n#!}{#x#!(#n# - #x#)!} = #comb#$. 
Then substitute into the equation: $P(X = #x#) = \binom{#n#}{#x#} #p#^#x# (1-#p#)^{#n#-#x#} = #res#$ """
        return {"res": res, "method": DistributionTemplates.method_template("binomial",
                                                                            "probability mass function",
                                                                            Binomial.pmf_formula,
                                                                            params,
                                                                            method)}

    @staticmethod
    def cdf_calc(n: int, p: Decimal, lower_x: int, upper_x: int, dp: int):
        res = 0
        for _ in range(0, upper_x + 1):
            res += Binomial.pmf_calc(n, p, _, dp + 2)
        for _ in range(0, lower_x):
            res -= Binomial.pmf_calc(n, p, _, dp + 2)
        return round(res, dp)

    @staticmethod
    def cdf(n_val: str, p_val: str, lower_x_val: str, upper_x_val: str, dp: str) -> dict:
        """
        :param dp: int, number of decimal places to round to
        :param n:  int, total number of trials
        :param p:  Decimal, probability of success
        :param lower_x: int, lower limit of interval
        :param upper_x: int, upper limit of interval
        :return:   Decimal, value of cdf between limits
        """
        if lower_x_val == "":
            lower_x_val = "0"
        elif upper_x_val == "":
            upper_x_val = n_val
        res = str(Binomial.cdf_calc(int(n_val), Decimal(p_val), int(lower_x_val), int(upper_x_val), int(dp)))
        params = {"n": n_val,
                  "p": p_val,
                  "x_1": lower_x_val,
                  "x_2": upper_x_val,
                  "res": res}
        method = r"""We know that $P(X=x) = \binom{n}{x} p^x (1-p)^x$. Using the general formula for the cumulative distribution of a discrete random variable: \n $P(#x_1# \leq x \leq #x_2#) = P(X \leq #x_2#) - P(X \leq #x_1#) = \sum_{i=0}^{#x_1#} P(X=i) - \sum_{i=0}^{#x_2# - 1} P(X=i) = #res#$ """
        return {"res": res, "method": DistributionTemplates.method_template("binomial",
                                                                                 "cumulative distribution function, "
                                                                                 "(CDF)",
                                                                                 Binomial.cdf_formula,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def inverse_binomial(n: str, p: str, P: str, dp: str):
        res = str(int(round(binom.ppf(float(P), n=int(n), p=float(p)), int(dp))))
        return {"res": res}

    @staticmethod
    def mean(n_val: str, p_val: str, dp: str) -> dict:
        """
        :param n:  int, total number of trials
        :param p:  Decimal, probability of success
        :return:   Decimal, mean of binomial distribution
        """
        n = int(n_val)
        p = Decimal(p_val)
        res = str(round(n * p, int(dp)))
        params = {
            "n": n_val,
            "p": p_val,
            "res": res
        }
        method = r"""Substituting into the formula:
$E(x) = #n# \times #p# = #res#$"""
        return {"res": res, "method": DistributionTemplates.method_template("binomial",
                                                                                 "mean",
                                                                                 Binomial.mean_formula,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def variance(n_val: str, p_val: str, dp: str) -> dict:
        """
        :param dp: int, number of decimal places
        :param n: int, total number of trials
        :param p: Decimal, probability of success
        :return:  Decimal, variance of binomial distribution
        """
        n = int(n_val)
        p = float(p_val)
        res = str(round(n * p * (1 - p), int(dp)))
        params = {
            "n": n_val,
            "p": p_val,
            "res": res
        }
        method = r"""Substituting into the formula:
$\sigma^2 = #n# \times #p# \times (1 - #p#) = #res# $"""
        return {"res": res, "method": DistributionTemplates.method_template("binomial",
                                                                                 "variance",
                                                                                 Binomial.var_formula,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def skewness(n_val: str, p_val: str, dp: str) -> dict:
        """
        :param dp: int, number of decimal places to round to
        :param n: int, total number of trials
        :param p: Decimal, probability of success
        :return:  Decimal, skewness of binomial distribution
        """
        n = int(n_val)
        p = float(p_val)
        standard_dev = str(round(math.sqrt(n * p * (1 - p)), int(dp)))
        res = str(round((1 - 2 * p) / float(standard_dev), int(dp)))
        params = {
            "n": n_val,
            "p": p_val,
            "res": res,
            "sigma": standard_dev
        }
        method = r"""First calculate the standard deviation which is given by $\sigma = \sqrt{np(1-p)} = \sqrt{#n# \times #p# \times (1 - #p#)} = #sigma#$
Then substitute into formula:
$\gamma = \frac{1 - 2 * #p#}{#sigma#} = {#res#}$
        """
        return {"res": res, "method": DistributionTemplates.method_template("binomial",
                                                                                 "skewness",
                                                                                 Binomial.skew_formula,
                                                                                 params,
                                                                                 method)}

    @staticmethod
    def binomial_bars(n: str, p: str) -> dict:
        points = {}
        for x in range(0, int(n) + 1):
            points[str(x)] = str(Binomial.pmf_calc(int(n), Decimal(p), x, 3))
        return points


if __name__ == "__main__":
    print(Binomial.pmf("50", "0.032", "2", "5"))
