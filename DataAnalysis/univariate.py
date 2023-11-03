import math
from Distributions.distribution_templates import DistributionTemplates


class UnivariateAnalysis:
    standard_dev_formula = (r"\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2} = \sqrt{(\frac{1}{N} \sum_{"
                            r"i=1}^N x_i^2) - (\frac{1}{N} \sum_{i=1}^N x_i)^2}")

    @staticmethod
    def mean(vals: list, freq_vals: list | None, dp: str) -> dict:
        if not freq_vals:
            return {"res": str(round(sum(vals) / len(vals), int(dp)))}
        else:
            total = 0
            for i in range(len(vals)):
                total += vals[i] * freq_vals[i]
            return {"res": str(round(total / sum(freq_vals), int(dp)))}

    @staticmethod
    def sum(vals: list, freq_vals: list | None, dp: str) -> dict:
        if not freq_vals:
            return {"res": str(round(sum(vals), int(dp)))}
        else:
            total = 0
            for i in range(len(vals)):
                total += vals[i] * freq_vals[i]
            return {"res": str(round(total, int(dp)))}

    @staticmethod
    def square_sum(vals: list, freq_vals: list | None, dp: str) -> dict:
        if not freq_vals:
            return {"res": str(round(sum([num ** 2 for num in vals]), int(dp)))}
        else:
            total = 0
            for i in range(len(vals)):
                total += (vals[i] ** 2) * freq_vals[i]
            return {"res": str(round(total, int(dp)))}

    @staticmethod
    def standard_dev(vals: list, freq_vals: list | None, dp: str):
        #
        # Calculates sd differently depending on whether frequency values or raw data is inputted
        #
        if not freq_vals:
            square_sum = round(sum([num ** 2 for num in vals]), int(dp) + 2)
            mean = round(sum(vals) / len(vals), int(dp) + 2)
            length = len(vals)
            res = round(math.sqrt(square_sum / len(vals) - mean ** 2), int(dp))
        else:
            square_sum = 0
            total = 0
            for i in range(len(vals)):
                total += vals[i] * freq_vals[i]
                square_sum += (vals[i] ** 2) * freq_vals[i]
            mean = total / sum(freq_vals)
            length = sum(freq_vals)
            res = round(math.sqrt(square_sum / sum(freq_vals) - mean ** 2), int(dp))

        params = {"square_sum": str(square_sum), "mean": str(mean), "res": str(res),
                  "formula": UnivariateAnalysis.standard_dev_formula,
                  "len": str(length)}
        method = r"""The formula for the standard deviation of $x_0, x_1, \dots, x_N$ is $#formula#$
    We have $\sum x^2 = #square_sum# $ and $\bar{x} = #mean#$.
    Substituting into the formula:
    $\sigma = \sqrt{\frac{#square_sum#}{#len#} - (#mean#)^2} = #res#$"""
        return {"res": str(res), "method": DistributionTemplates.replace_vals(params, method)}

    @staticmethod
    def calculate_percentile(x_vals: list, freq_vals: list | None, percentile: float, dp: str):
        idx = 0
        while idx < len(freq_vals):
            if sum(freq_vals[:idx + 1]) >= percentile:
                break
            idx += 1
        diff = percentile - sum(freq_vals[:idx])
        percentile_val = x_vals[idx - 1] + (x_vals[idx] - x_vals[idx - 1]) * (diff / freq_vals[idx - 1])
        return round(percentile_val, int(dp))

    @staticmethod
    def quartiles(x_vals: list, freq_vals: list | None, dp: str):
        if freq_vals:
            vals = {x_vals[i]: freq_vals[i] for i in range(len(x_vals))}
            sorted_vals = sorted(x_vals)
            sorted_freq_vals = [vals[x] for x in sorted_vals]
            total_freq = sum(freq_vals)
            lower_quartile = total_freq / 4
            median = 2 * lower_quartile
            upper_quartile = 3 * lower_quartile
            return {"res": str(UnivariateAnalysis.calculate_percentile(sorted_vals, sorted_freq_vals, lower_quartile, dp))}, {"res": str(UnivariateAnalysis.calculate_percentile(sorted_vals, sorted_freq_vals, median, dp))}, {"res": str(UnivariateAnalysis.calculate_percentile(sorted_vals, sorted_freq_vals, upper_quartile, dp))}
        else:
            sorted_vals = sorted(x_vals)
            lower_quartile_idx = len(sorted_vals) / 4
            median_idx = lower_quartile_idx * 2
            upper_quartile_idx = lower_quartile_idx * 3
            lower_quartile = (sorted_vals[math.floor(lower_quartile_idx)] + sorted_vals[
                math.ceil(lower_quartile_idx)]) / 2
            median = (sorted_vals[math.floor(median_idx)] + sorted_vals[math.ceil(median_idx)]) / 2
            upper_quartile = (sorted_vals[math.floor(upper_quartile_idx)] + sorted_vals[
                math.ceil(upper_quartile_idx)]) / 2
            return {"res": str(round(lower_quartile, int(dp)))}, {"res": str(round(median, int(dp)))}, {"res": str(round(upper_quartile, int(dp)))}

    @staticmethod
    def generate_results(x_vals: list, freq_vals: list | None, dp: str):
        print(x_vals)
        x_vals = [float(x) for x in x_vals]
        freq_vals = [int(f) for f in freq_vals] if freq_vals else None
        lower, median, upper = UnivariateAnalysis.quartiles(x_vals, freq_vals, dp)
        return {"sum": UnivariateAnalysis.sum(x_vals, freq_vals, dp),
                "mean": UnivariateAnalysis.mean(x_vals, freq_vals, dp),
                "square_sum": UnivariateAnalysis.square_sum(x_vals, freq_vals, dp),
                "sd": UnivariateAnalysis.standard_dev(x_vals, freq_vals, dp),
                "LQ": lower,
                "median": median,
                "UQ": upper}


if __name__ == "__main__":
    print(UnivariateAnalysis.generate_results([35, 36, 37, 38], [3, 17, 29, 34], 2))
