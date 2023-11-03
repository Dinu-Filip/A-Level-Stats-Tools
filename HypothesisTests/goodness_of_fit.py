from scipy.stats import binom
from scipy.stats import chi2
from Distributions.distribution_templates import DistributionTemplates


class GoodnessOfFit:
    binomial_mass_function = r"P(X = x) = \binom{n}{x} p^x (1-p)^{n-x}"

    @staticmethod
    def binomial_estimate(params: dict):
        method_1 = r"""The formula for estimate the probability of success with a binomial distribution is:
$p = \frac{\sum(r \times f_r)}{n \times N}$
where $r$ is the number of successes, $f_r$ is the number of times that $x$ successes were observed (the observed frequency), $n$ is the number of trials and $N$ is the number of observations"""
        total_successes = 0
        x_vals = params["xVals"]
        observed_vals = params["observedVals"]
        n = int(params["n"])
        num_observations = sum(params["observedVals"])
        for i in range(len(x_vals)):
            total_successes += x_vals[i] * observed_vals[i]
        method_1 += r"$\sum(r \times f_r) = " + str(total_successes) + "$"
        method_1 += r", $n \times N = " + str(n * num_observations) + "$. "
        prob = round(total_successes / (n * num_observations), 4)
        method_1 += r"Therefore, $p = " + str(prob) + "$"
        return prob, method_1

    @staticmethod
    def binomial_expected_freq(params):
        observed_vals = params["observedVals"]
        total_freq = sum(observed_vals)
        x_vals = params["xVals"]
        n = int(params["n"])
        p = float(params["p"])
        expected_vals = []
        for i in range(len(x_vals)):
            expected_vals.append(round(binom.pmf(x_vals[i], n, p) * total_freq, 4))
        method = r"The binomial probability mass function is $" + GoodnessOfFit.binomial_mass_function + "$. "
        method += ("""
To calculate expected frequencies, use the binomial mass function on every value of $x$.""")
        method += ("""
For this data, the expected values are """) + ", ".join([str(val) for val in expected_vals])
        return method, expected_vals

    @staticmethod
    def binomial_hypotheses(params: dict):
        method_1 = "Under the null hypothesis, the data can be modelled using the distribution $B(#n#, #p#)$"
        method_2 = "The chi-squared distribution is used to find the critical region assuming the null hypothesis is true; if the test statistic lies in the critical region, we reject the null hypothesis in favour of the alternative hypothesis"
        return method_1, method_2

    @staticmethod
    def merge_cells(params: dict, expected_frequencies: list):
        method_4 = "To be able to use the chi-squared distribution to model the measure of goodness of fit, $X^2$, each of the expected frequencies must be greater than 5. "
        x_vals = params["xVals"]
        merge_regions = []
        idx_1 = 0
        idx_2 = 1
        while idx_2 < len(expected_frequencies):
            current_freq = expected_frequencies[idx_1]
            #
            # Merges adjacent cells until they sum to greater than 5
            #
            if current_freq < 5:
                if sum(expected_frequencies[idx_1:idx_2 + 1]) < 5:
                    idx_2 += 1
                    continue
                else:
                    merge_regions.append((idx_1, idx_2))
            idx_1 = idx_2
            idx_2 += 1
        if expected_frequencies[idx_1] < 5:
            #
            # If the element at idx_1 is less than 5 and no merge regions have been added,
            # then all the elements from that index to the end sum to less than 5
            # so must be combined with the element before
            #
            if len(merge_regions) == 0:
                merge_regions.append([idx_1 - 1, idx_2 - 1])
            else:
                #
                # If the element overlaps with or is adjacent to the previous region, it and all
                # subsequent elements must be merged with that region
                #
                if merge_regions[-1][0] <= idx_1 <= (merge_regions[-1][1] + 1):
                    merge_regions[-1][1] = idx_2 - 1
                else:
                    merge_regions.append([idx_1 - 1, idx_2 - 1])

        small_freq = []
        small_freq_xs = []
        for i in range(len(expected_frequencies)):
            if expected_frequencies[i] < 5:
                small_freq.append(expected_frequencies[i])
                small_freq_xs.append(x_vals[i])

        if len(merge_regions) > 0:
            method_4 += "The expected frequencies " + ", ".join([str(freq) for freq in
                                                                 small_freq]) + " are less than 5, so they need to be merged with adjacent frequencies."
            method_4 += """
We can merge the expected frequencies at x = """
            for region in merge_regions:
                for i in range(region[0], region[1] + 1):
                    method_4 += str(x_vals[i]) + ", "
                method_4 += "x = "
            method_4 = method_4[:-6]
        else:
            method_4 += "None of the expected frequencies are less than 5, so they do not have to be combined with any other frequencies."

        return merge_regions, method_4

    @staticmethod
    def calculate_measure_of_fit(params, expected_vals, merge_regions):
        method_5 = r"To calculate the measure of goodness of fit, use the formula $X^2 = \sum{\frac{(O_i - E_i)^2}{E_i}}$. Use the values of the observed and expected frequencies from the table above after merging."
        observed_vals = params["observedVals"]
        region_idx = 0
        merged_observed = []
        merged_expected = []
        if len(merge_regions) > 0:
            for i in range(len(observed_vals)):
                current_region = merge_regions[region_idx]
                #
                # Adds the number in adjacent cells represented by the merge regions
                #
                if current_region[0] < i <= current_region[1]:
                    merged_observed[-1] += observed_vals[i]
                    merged_expected[-1] += expected_vals[i]
                    if i == current_region[1]:
                        region_idx += 1
                else:
                    merged_observed.append(observed_vals[i])
                    merged_expected.append(expected_vals[i])
        else:
            merged_observed = observed_vals
            merged_expected = expected_vals
        measure_of_fit = 0
        for i in range(len(merged_observed)):
            measure_of_fit += ((merged_observed[i] - merged_expected[i]) ** 2) / merged_expected[i]
        measure_of_fit = round(measure_of_fit, 4)
        return merged_observed, merged_expected, measure_of_fit, method_5

    @staticmethod
    def num_deg_free(merged_observed, estimate_param):
        method_6 = "The number of degrees of freedom is found by subtracting the number of constraints from the number of cells after merging. "
        if estimate_param == "true":
            method_6 += "Since the #estimateParamName# was estimated by calculation, this introduces an additional constraint. "
            method_6 += ("""
There are therefore two constrains total - the estimated #estimateParamName# and the requirement that the total number of expected frequencies is the same as the total number of observed frequencies: """)
            num_regions = len(merged_observed)
            num_deg = len(merged_observed) - 2
            method_6 += "$v = " + str(num_regions) + " - 2 = " + str(num_deg) + "$"
        else:
            method_6 += """
There is only one constraint - the total number of observed frequencies must equal the total number of expected frequencies: """
            num_regions = len(merged_observed)
            num_deg = len(merged_observed) - 1
            method_6 += "$v = " + str(num_regions) + " - 1 = " + str(num_deg) + "$"
        return num_deg, method_6

    @staticmethod
    def calculate_critical_val(num_deg_free, sig_level):
        sig_level = round(float(sig_level), 4)
        method_7 = "We use the chi-squared distribution to model the measure of goodness of fit. We can use the inverse chi-squared function with " + str(
            num_deg_free) + " degrees of freedom: "
        critical_value = round(chi2.ppf(1 - sig_level, df=num_deg_free), 4)
        method_7 += r"$\chi_" + str(num_deg_free) + r"^2(" + str(sig_level) + ") = " + str(critical_value) + "$"
        return critical_value, method_7

    @staticmethod
    def compare_critical_value(critical_value, measure_of_fit):
        method_8 = ""
        if measure_of_fit < critical_value:
            result = True
            method_8 += str(measure_of_fit) + " < " + str(
                critical_value) + " so the measure of fit is not in the critical region. "
            method_8 += "This suggests that there is insufficient evidence to reject the null hypothesis. This means that under this significance level, the data can be modelled using a #distribution# distribution"
        else:
            method_8 += str(measure_of_fit) + " > " + str(
                critical_value) + " so the measure of fit is in the critical region"
            method_8 += "This suggests that there is sufficient evidence to reject the null hypothesis. Under this significance level, a #distribution# distribution is not a suitable model for this data."
            result = False
        return result, method_8

    @staticmethod
    def goodness_of_fit(params):
        params["xVals"] = [int(num) for num in params["xVals"].split(params["delimiter"])]
        params["observedVals"] = [int(num) for num in params["observedVals"].split(params["delimiter"])]
        distribution = params["distribution"]
        prob = None
        if params["estimateParam"] == "true":
            param_name = params["estimateParamName"]
            step_estimate = f"Estimate the {param_name}"
            method_estimate = None
            match distribution:
                case "Binomial": prob, method_estimate = GoodnessOfFit.binomial_estimate(params)
            params["p"] = prob
        else:
            prob = params["p"]
        step_1 = "State the null hypothesis: $H_0$: #distribution# distribution is suitable"
        step_2 = "State the alternative hypothesis: $H_1$: #distribution# distribution is not suitable model"
        method_1 = None
        method_2 = None
        match distribution:
            case "Binomial": method_1, method_2 = GoodnessOfFit.binomial_hypotheses(params)
        step_3 = "Calculate the expected frequency of each value of $x$"
        method_3 = None
        expected_vals = None
        match distribution:
            case "Binomial": method_3, expected_vals = GoodnessOfFit.binomial_expected_freq(params)
        step_4 = "Combine any cells with expected frequencies greater than 5"
        merge_regions, method_4 = GoodnessOfFit.merge_cells(params, expected_vals)
        step_5 = "Calculate the measure of goodness of fit, $X^2$"
        merged_observed, merged_expected, measure_of_fit, method_5 = GoodnessOfFit.calculate_measure_of_fit(params,
                                                                                                            expected_vals,
                                                                                                            merge_regions)
        step_6 = "Calculate number of degrees of freedom"
        num_deg_free, method_6 = GoodnessOfFit.num_deg_free(merged_observed, params["estimateParam"])
        step_7 = "Use the chi-squared distribution with " + str(
            num_deg_free) + f" degrees {'s' if num_deg_free > 1 else ''} of freedom to find the critical value"
        critical_value, method_7 = GoodnessOfFit.calculate_critical_val(num_deg_free, params["sigLevel"])
        step_8 = "Compare measure of fit to critical value"
        result, method_8 = GoodnessOfFit.compare_critical_value(critical_value, measure_of_fit)
        result = {"res": {"outcome": str(result),
                          "critical_value": critical_value},
                  "test": {}}
        steps = [step_1, step_2, step_3, step_4, step_5, step_6, step_7, step_8]
        steps = [DistributionTemplates.replace_vals(params, step) for step in steps]
        methods = [method_1, method_2, method_3, method_4, method_5, method_6, method_7, method_8]
        methods = [DistributionTemplates.replace_vals(params, method) for method in methods]
        if params["estimateParam"] == "true":
            result["res"]["estimatedParam"] = prob
            result["test"]["1) " + step_estimate] = method_estimate
            for i in range(2, 10):
                result["test"][f"{i}) " + steps[i - 2]] = methods[i - 2]
        else:
            for i in range(1, 9):
                result["test"][f"{i}) " + steps[i - 1]] = methods[i - 1]
        return result


if __name__ == "__main__":
    print(GoodnessOfFit.goodness_of_fit({"distribution": "binomial",
                                   "estimateParam": False,
                                   "p": 0.5,
                                   "n": 5,
                                   "estimateParamName": "probability of success",
                                   "xVals": [0, 1, 2, 3, 4, 5],
                                   "observedVals": [13, 18, 38, 20, 10, 1],
                                   "sigLevel": 0.05}))