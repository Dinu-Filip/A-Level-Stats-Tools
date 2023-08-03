import math
from Distributions.distribution_templates import DistributionTemplates

class BivariateAnalysis:
    standard_dev_formula = r"\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2} = \sqrt{(\frac{1}{N} \sum_{i=1}^N x_i^2) - (\frac{1}{N} \sum_{i=1}^N x_i)^2}"
    pmcc_formula = r"\rho = \frac{S_{xy}}{\sqrt{S_{xx} S_{yy}}"
    s_xx_formula = r"S_{xx} = \sum (x_i - \bar{x})^2 = \sum x_i^2 - \frac{(\sum x_i)^2}{n}"
    s_xy_formula = r"S_{xy} = \sum (x_i - \bar{x})(y_i - \bar{y}) = \sum x_i y_i - \frac{\sum x_i \sum y_i}{n}"

    @staticmethod
    def mean(vals: list, dp: str) -> dict:
        vals = [float(num) for num in vals]
        return {"res": round(sum(vals) / len(vals), int(dp))}

    @staticmethod
    def sum(vals: list, dp: str) -> dict:
        vals = [float(num) for num in vals]
        return {"res": round(sum(vals), int(dp))}

    @staticmethod
    def square_sum(vals: list, dp: str) -> dict:
        vals = [float(num) for num in vals]
        return {"res": round(sum([num ** 2 for num in vals]), int(dp))}

    @staticmethod
    def standard_dev(vals: list, dp: str):
        vals = [float(num) for num in vals]
        square_sum = sum([num ** 2 for num in vals])
        mean = sum(vals) / len(vals)
        res = round(math.sqrt(square_sum / len(vals) - mean ** 2), int(dp))
        length = len(vals)
        params = {"square_sum": square_sum, "mean": mean, "res": res, "formula": BivariateAnalysis.standard_dev_formula, "len": length}
        method = r"""The formula for the standard deviation of $x_0, x_1, \dots, x_N$ is #formula#
We have $\sum x^2 = #square_sum# $ and $\bar{x} = #mean#$.
Substituting into the formula:
$\sigma = \sqrt{\frac{#square_sum#}{#len#}} - (#mean#)^2} = #res#"""
        return {"res": res, "method": DistributionTemplates.replace_vals(params, method)}

    @staticmethod
    def pmcc(x_vals: list, y_vals: list, dp: str):
        x_vals = [float(num) for num in x_vals]
        y_vals = [float(num) for num in y_vals]
        length = len(x_vals)
        product_sum = sum([x_vals[i] * y_vals[i] for i in range(len(x_vals))])
        x_sum = sum(x_vals)
        y_sum = sum(y_vals)
        x_square_sum = sum([num ** 2 for num in x_vals])
        y_square_sum = sum([num ** 2 for num in y_vals])
        s_xx = x_square_sum - ((x_sum ** 2) / length)
        s_yy = y_square_sum - ((y_sum ** 2) / length)
        s_xy = product_sum - ((x_sum * y_sum) / length)
        res = round(s_xy / math.sqrt(s_xx * s_yy), int(dp))
        params = {
            "formula": BivariateAnalysis.pmcc_formula,
            "s_xx_formula": BivariateAnalysis.s_xx_formula,
            "s_xy_formula": BivariateAnalysis.s_xy_formula,
            "x_sum": x_sum,
            "y_sum": y_sum,
            "x_square_sum": x_square_sum,
            "y_square_sum": y_square_sum,
            "product_sum": product_sum,
            "len": length,
            "res": res
        }
        method = r"""The formula for the Pearson's correlation coefficient is $#formula#$, where $ S_{xx} = #s_xx_formula# $ and $ S_{xy} = #s_xy_formula# $
We have $ \sum x = #x_sum#, \sum y = #y_sum#, \sum x^2 = #x_square_sum#, \sum y^2 = #y_square_sum#, \sum xy = #product_sum# $ and $ n = #len#.
Substituting into the formula for $ S_xx $:
$ S_xx = #x_square_sum# - \frac{(#x_sum#)^2}{#len#} = #s_xx#
Similarly for $ S_yy $:
$ S_yy = #y_square_sum# - \frac{(#y_sum#)^2}{#len#} = #s_yy#
Substituting into the formula for $ S_xy $:
$ S_xy = #product_sum# - \frac{#x_sum# \times #y_sum#}{#len#}
Now to calculate the product moment correlation coefficient:
\rho = \frac{#s_xy#}{\sqrt{#s_xx# \times #s_yy#}} = #res#"""
        return {"res": res, method: DistributionTemplates.replace_vals(params, method)}

    @staticmethod
    