from decimal import *
from decimal import Decimal
import math
from distribution_templates import DistributionTemplates


class Binomial:
    binomial_pmf = """P(X = x) = \\binom{n}{x} p^x (1-p)^{n-x}"""
    binomial_cdf = """P(X \\leq x) = \\sum_{{i = 1}}^{{x}} P(X=x)"""
    binomial_mean = """E(X) = n \\times p"""
    binomial_variance = """\\sigma^2 = np(1-p)"""
    binomial_skewness = """\\gamma = \\frac{1 - 2p}{\\sqrt{np(1-p)}}"""

    @staticmethod
    def pmf(n: int, p: Decimal, x: int) -> Decimal:
        #
        # pmf used to calculate raw pmf value
        # Used by cdf and graphing algorithms in frontend
        #
        return Decimal(math.comb(n, x)) * Decimal(pow(p, x)) * Decimal(pow(1 - p, n - x))

    @staticmethod
    def pmf_method(n: int, p: Decimal, x: int, dp: int) -> dict:
        """
        :param dp: int, number of decimal places to round to
        :param n:  int, total number of trials
        :param p:  Decimal, probability of success
        :param x:  int, number of successful trials
        :return:   Decimal, value of pmf at particular x
        """
        if x > n:
            raise ValueError("The number of successes must be less than or equal to the total number of trials")
        binom_val = math.comb(n, x)
        res = round(Binomial.pmf(n, p, x), dp)
        params = {
            "n": n,
            "p": p,
            "x": x
        }
        method = f"""First find the binomial coefficient using the formula 
                 \\(\\binom{{{n}}}{{{x}}} = \frac{{{n}!}}{{{x}!({n}-{x})!}}\\) or using Pascal's triangle\n
                 \\(\binom{{{n}}}{{{x}}} = {math.comb(n, x)} \\) so
                 \\begin{{align*}}
                 P(X = {x}) &= \\binom{{{n}}}{{{x}}} \\times {p}^{x} \\times (1-{p})^{{{n}-{x}}} \\\\
                 &= {binom_val} \\times \\ {p}^{x} \\ ({1 - p})^{{{n - x}}} \\\\
                 &= {res}
                 \end{{align*}}
                 """
        return {"res": res, "full_method": DistributionTemplates.discrete("Binomial",
                                                                          "probability mass function, (PMF)",
                                                                          Binomial.binomial_pmf,
                                                                          params,
                                                                          method)}

    @staticmethod
    def cdf(n: int, p: Decimal, lower_x: int, upper_x: int, dp: int) -> Decimal:
        """
        :param dp: int, number of decimal places to round to
        :param n:  int, total number of trials
        :param p:  Decimal, probability of success
        :param lower_x: int, lower limit of interval
        :param upper_x: int, upper limit of interval
        :return:   Decimal, value of cdf between limits
        """
        if upper_x > n:
            raise ValueError("The number of successes must be less than or equal to the total number of trials")
        elif upper_x <= lower_x:
            raise ValueError("The upper limit must be greater than the lower limit")
        res = 0
        for _ in range(0, upper_x + 1):
            res += Binomial.pmf(n, p, _)
        for _ in range(0, lower_x):
            res -= Binomial.pmf(n, p, _)
        res = round(res, dp)
        params = {"n": n,
                  "p": p,
                  "x_1": lower_x,
                  "x_2": upper_x}
        method = f"""We know that \\(P(X \\leq x) = \\sum_{{i = 1}}^{{x}} P(X=x) 
        = \\sum_{{i = 1}}^{{x}} \\binom{{n}}{{i}}p^i(1-p)^{{n-i}} \\) and that \\(P(a \\leq X \\leq b) 
        = P(X \\leq b) - P(X < a) \\).
        Substituting \\( a = x_1 \\) and \\( b = x_2 \\):
        \\(P({{{lower_x}}} \\leq X \\leq {{{upper_x}}}) &= P(X \\leq {{{upper_x}}}) - P(X < {{{lower_x}}})\\
        &= \\sum_{{i = 1}}^{{{upper_x}}} P(X = i) - \\sum_{{i = 1}}^{{{lower_x} - 1}} P(X = i)\\
        &= {res} \\)
        """
        return {"res": res, "full_method": DistributionTemplates.discrete("binomial",
                                                                          "cumulative distribution function, (CDF)",
                                                                          Binomial.binomial_cdf,
                                                                          params,
                                                                          method)}

    @staticmethod
    def mean(n: int, p: Decimal, dp: int) -> Decimal:
        """
        :param n:  int, total number of trials
        :param p:  Decimal, probability of success
        :return:   Decimal, mean of binomial distribution
        """
        res = round(n * p, dp)
        params = {
            "n": n,
            "p": p
        }
        method = f"""
        Substituting into the formula:
        \\(E(x) = n \\times p = {n} \times {p} = {res}\\)
        """
        return {"res": res, "full_method": DistributionTemplates.discrete("binomial",
                                                                          "mean",
                                                                          Binomial.binomial_mean,
                                                                          params,
                                                                          method)}

    @staticmethod
    def variance(n: int, p: Decimal, dp: int) -> Decimal:
        """
        :param dp: int, number of decimal places
        :param n: int, total number of trials
        :param p: Decimal, probability of success
        :return:  Decimal, variance of binomial distribution
        """
        res = round(n * p * (1 - p), dp)
        params = {
            "n": n,
            "p": p
        }
        method = f"""
        Substituting into the formula:
        \\(\\sigma^2 = {n} \\times {p} \times {1 - p} = {res} \\) 
        """
        return {"res": res, "full_method": DistributionTemplates.discrete("binomial",
                                                                          "variance",
                                                                          Binomial.binomial_variance,
                                                                          params,
                                                                          method)}

    @staticmethod
    def skewness(n: int, p: Decimal, dp: int) -> Decimal:
        """
        :param dp: int, number of decimal places to round to
        :param n: int, total number of trials
        :param p: Decimal, probability of success
        :return:  Decimal, skewness of binomial distribution
        """
        standard_dev = math.sqrt(n * p * (1 - p))
        res = round((1 - 2 * p) / standard_dev, dp)
        params = {
            "n": n,
            "p": p
        }
        method = f"""
        First calculate the standard deviation which is given by \\(\sigma = \\sqrt{{np(1-p)}} 
        = \\sqrt{{{n}}} \\times {p} \\times {1 - p}}} = {standard_dev} \\)
        Then substitute into formula:
        \\( \\gamma = \\frac{{1 - 2p}}{{\\sigma}} \\ &= \\frac{{{1 - 2 * p}}}{{{standard_dev}}} \\ &= {res} \\)
        """
        return {"res": res, "full_method": DistributionTemplates.discrete("binomial",
                                                                          "skewness",
                                                                          Binomial.binomial_skewness,
                                                                          params,
                                                                          method)}
