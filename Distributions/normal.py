from scipy.stats import norm
from Distributions.distribution_templates import DistributionTemplates
import numpy as np

class Normal:
    cdf_formula = r"P(X \leq x) = \frac{1}{\sigma \sqrt{2 \pi}} \int_{-\infty}^{x} e^{\frac{-(t-\mu)^2}{2 \sigma^2}} dt"

    @staticmethod
    def cdf_calc(mean: float, sd: float, lower_x: float, upper_x: float, dp: int):
        res = norm.cdf(upper_x, mean, sd) - norm.cdf(lower_x, mean, sd)
        return round(res, dp)

    @staticmethod
    def cdf(mean_val: str, sd_val: str, lower_x_val: str, upper_x_val: str, dp: str) -> dict:
        """
        :param mean: Decimal, mean of normal distribution
        :param sd:   Decimal, standard deviation
        :param x:    Decimal, value of random variable
        :return:
        """
        params = {"mu": mean_val,
                  "sigma": sd_val}
        #
        # Allows the user to optionally leave either the lower bound or the upper bound blank
        # to calculate unbounded probabilities
        #
        if lower_x_val == "":
            res = str(round(norm.cdf(float(upper_x_val), float(mean_val), float(sd_val)), int(dp)))
            params["res"] = res
            params["x"] = upper_x_val
            method = r"""Substituting into the equation we have $P(X \leq #x#) = #res#$
Since the integral in the normal CDF is not analytically tractable, it must be evaluated numerically.
                    """
        elif upper_x_val == "":
            res = str(round(1 - norm.cdf(float(lower_x_val), float(mean_val), float(sd_val)), int(dp)))
            params["res"] = res
            params["x"] = lower_x_val
            method = r"""We know that in general for continuous random variables:
$P(X \geq x) = 1 - P(X \leq x)$
So substituting into the equation we have $P(X \geq x) = 1 - P(X \leq #x#) = #res#$
Since the integral in the normal CDF is not analytically tractable, it must be evaluated numerically."""
        else:
            res = str(Normal.cdf_calc(float(mean_val), float(sd_val), float(lower_x_val), float(upper_x_val), int(dp)))
            params["res"] = res
            params["x_1"] = lower_x_val
            params["x_2"] = upper_x_val
            method = r"""We know that in general for continuous random variables:
$P(a \leq X \leq b) = P(X \leq b) - P(X \leq a)$
So substituting into the equation we have
$P(#x_1# \leq X \leq #x_2#) = P(X \leq #x_2#) - P(X \leq #x_1#) = #res#$
Since the integral in the normal CDF is not analytically tractable, it must be evaluated numerically."""

        return {"res": res, "method": DistributionTemplates.method_template("normal",
                                                                            "cumulative distribution function",
                                                                            Normal.cdf_formula,
                                                                            params,
                                                                            method)}

    @staticmethod
    def inverse_normal(mean: str, sd: str, P: str, dp: str):
        res = str(round(norm.ppf(float(P), loc=float(mean), scale=float(sd)), int(dp)))
        return {"res": res}

    @staticmethod
    def normal_points(mean: str, sd: str, lower: str | None, upper: str | None) -> dict:
        mean = float(mean)
        sd = float(sd)
        min_x = mean - 5 * sd
        max_x = mean + 5 * sd
        points = {}
        #
        # Calculates 100 evenly spaced values to represent the points for the graph
        #
        point_vals = list(np.linspace(min_x, max_x, 100))
        if lower:
            point_vals.append(float(lower))
        if upper:
            point_vals.append(float(upper))
        point_vals.sort()
        for val in point_vals:
            points[str(val)] = str(norm.pdf(val, mean, sd))
        return points


if __name__ == "__main__":
    print(Normal.cdf("40", "2", "35", "45", "8"))
