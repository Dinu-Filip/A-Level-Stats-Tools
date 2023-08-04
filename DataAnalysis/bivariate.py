import math
from Distributions.distribution_templates import DistributionTemplates
from scipy.stats import linregress


class BivariateAnalysis:
    standard_dev_formula = r"\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2} = \sqrt{(\frac{1}{N} \sum_{i=1}^N x_i^2) - (\frac{1}{N} \sum_{i=1}^N x_i)^2}"
    pmcc_formula = r"\rho = \frac{S_{xy}}{\sqrt{S_{xx} S_{yy}}}"
    s_xx_formula = r"S_{xx} = \sum (x_i - \bar{x})^2 = \sum x_i^2 - \frac{(\sum x_i)^2}{n}"
    s_xy_formula = r"S_{xy} = \sum (x_i - \bar{x})(y_i - \bar{y}) = \sum x_i y_i - \frac{\sum x_i \sum y_i}{n}"

    @staticmethod
    def mean(vals: list, dp: str) -> dict:
        vals = [float(num) for num in vals]
        return {"res": str(round(sum(vals) / len(vals), int(dp)))}

    @staticmethod
    def sum(vals: list, dp: str) -> dict:
        vals = [float(num) for num in vals]
        return {"res": str(round(sum(vals), int(dp)))}

    @staticmethod
    def square_sum(vals: list, dp: str) -> dict:
        vals = [float(num) for num in vals]
        return {"res": str(round(sum([num ** 2 for num in vals]), int(dp)))}

    @staticmethod
    def standard_dev(vals: list, dp: str):
        vals = [float(num) for num in vals]
        square_sum = round(sum([num ** 2 for num in vals]), int(dp) + 2)
        mean = round(sum(vals) / len(vals), int(dp) + 2)
        res = round(math.sqrt(square_sum / len(vals) - mean ** 2), int(dp))
        length = len(vals)
        params = {"square_sum": str(square_sum), "mean": str(mean), "res": str(res), "formula": BivariateAnalysis.standard_dev_formula,
                  "len": str(length)}
        method = r"""The formula for the standard deviation of $x_0, x_1, \dots, x_N$ is $#formula#$
We have $\sum x^2 = #square_sum# $ and $\bar{x} = #mean#$.
Substituting into the formula:
$\sigma = \sqrt{\frac{#square_sum#}{#len#} - (#mean#)^2} = #res#$"""
        return {"res": str(res), "method": DistributionTemplates.replace_vals(params, method)}

    @staticmethod
    def pmcc(x_vals: list, y_vals: list, dp: str):
        x_vals = [float(num) for num in x_vals]
        y_vals = [float(num) for num in y_vals]
        length = len(x_vals)
        product_sum = round(sum([x_vals[i] * y_vals[i] for i in range(len(x_vals))]), int(dp) + 2)
        x_sum = round(sum(x_vals), int(dp) + 2)
        y_sum = round(sum(y_vals), int(dp) + 2)
        x_square_sum = round(sum([num ** 2 for num in x_vals]), int(dp) + 2)
        y_square_sum = round(sum([num ** 2 for num in y_vals]), int(dp) + 2)
        s_xx = round(x_square_sum - ((x_sum ** 2) / length), int(dp) + 2)
        s_yy = round(y_square_sum - ((y_sum ** 2) / length), int(dp) + 2)
        s_xy = round(product_sum - ((x_sum * y_sum) / length), int(dp) + 2)
        res = round(s_xy / math.sqrt(s_xx * s_yy), int(dp))
        params = {
            "formula": BivariateAnalysis.pmcc_formula,
            "s_xx_formula": BivariateAnalysis.s_xx_formula,
            "s_xy_formula": BivariateAnalysis.s_xy_formula,
            "x_sum": str(x_sum),
            "y_sum": str(y_sum),
            "x_square_sum": str(x_square_sum),
            "y_square_sum": str(y_square_sum),
            "product_sum": str(product_sum),
            "s_xx": str(s_xx),
            "s_yy": str(s_yy),
            "s_xy": str(s_xy),
            "len": str(length),
            "res": str(res)
        }
        method = r"""The formula for the Pearson's correlation coefficient is $#formula#$, where $ S_{xx} = #s_xx_formula# $ and $ S_{xy} = #s_xy_formula# $
We have $ \sum x = #x_sum#, \sum y = #y_sum#, \sum x^2 = #x_square_sum#, \sum y^2 = #y_square_sum#, \sum xy = #product_sum# $ and $ n = #len# $.
Substituting into the formula for $ S_xx $:
$ S_xx = #x_square_sum# - \frac{(#x_sum#)^2}{#len#} = #s_xx# $
Similarly for $ S_yy $:
$ S_yy = #y_square_sum# - \frac{(#y_sum#)^2}{#len#} = #s_yy# $
Substituting into the formula for $ S_xy $:
$ S_xy = #product_sum# - \frac{#x_sum# \times #y_sum#}{#len#} $
Now to calculate the product moment correlation coefficient:
$ \rho = \frac{#s_xy#}{\sqrt{#s_xx# \times #s_yy#}} = #res#$"""
        return {"res": str(res), "method": DistributionTemplates.replace_vals(params, method)}

    @staticmethod
    def regress_slope(x_vals: list, y_vals: list, dp: str):
        x_vals = [float(num) for num in x_vals]
        y_vals = [float(num) for num in y_vals]
        res = round(linregress(x_vals, y_vals).slope, int(dp))
        return {"res": str(res)}

    @staticmethod
    def regress_intercept(x_vals: list, y_vals: list, dp: str):
        x_vals = [float(num) for num in x_vals]
        y_vals = [float(num) for num in y_vals]
        res = round(linregress(x_vals, y_vals).intercept, int(dp))
        return {"res": str(res)}

    @staticmethod
    def generate_results(x_vals: list, y_vals: list, dp: str):
        result = {"x_sum": BivariateAnalysis.sum(x_vals, dp), "y_sum": BivariateAnalysis.sum(y_vals, dp),
                  "x_mean": BivariateAnalysis.mean(x_vals, dp), "y_mean": BivariateAnalysis.mean(y_vals, dp),
                  "x_square_sum": BivariateAnalysis.square_sum(x_vals, dp),
                  "y_square_sum": BivariateAnalysis.square_sum(y_vals, dp),
                  "x_sd": BivariateAnalysis.standard_dev(x_vals, dp),
                  "y_sd": BivariateAnalysis.standard_dev(y_vals, dp), "rho": BivariateAnalysis.pmcc(x_vals, y_vals, dp),
                  "regress_slope": BivariateAnalysis.regress_slope(x_vals, y_vals, dp),
                  "regress_intercept": BivariateAnalysis.regress_intercept(x_vals, y_vals, dp)}
        return result

if __name__ == "__main__":
    print(BivariateAnalysis.generate_results(["3", "5", "6", "8", "9", "11"], ["1.04", "1.49", "1.79", "2.58", "3.1", "4.463"], "4"))