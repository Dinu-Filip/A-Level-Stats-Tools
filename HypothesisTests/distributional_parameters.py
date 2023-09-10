import math

from scipy.stats import *
from Distributions.distribution_templates import DistributionTemplates


class DistributionHT:
    discrete_distributions = ["binomial"]
    continuous_distributions = ["normal"]

    @staticmethod
    def binomial_cr(params: dict, type_test: str, type_tail=None) -> list:
        num_trials = int(params["num_trials"])
        print(num_trials)
        prob = float(params["population_param_value"])
        print(prob)
        sig_level = float(params["sig_level"])
        print(sig_level)
        if type_test == "one-tailed":
            if type_tail == "lower":
                cv = binom.ppf(sig_level, num_trials, prob)
                if binom.cdf(cv, num_trials, prob) > sig_level:
                    cv -= 1
                return [(0, cv)]
            else:
                cv = binom.ppf(1 - sig_level, num_trials, prob) + 1
                if 1 - binom.cdf(cv - 1, num_trials, prob) > sig_level:
                    cv += 1
                return [(cv, num_trials)]
        else:
            cv_1 = binom.ppf(sig_level / 2, num_trials, prob)
            if binom.cdf(cv_1, num_trials, prob) > sig_level / 2:
                cv_1 -= 1
            cv_2 = binom.ppf(1 - sig_level / 2, num_trials, prob)
            if 1 - binom.cdf(cv_2 - 1, num_trials, prob) > sig_level / 2:
                cv_2 += 1
            return [(0, cv_1), (cv_2, num_trials)]

    @staticmethod
    def normal_cr(params: dict, type_test: str, type_tail=None):
        mean = float(params["population_mean"])
        sd = float(params["sd"])
        N = float(params["N"])
        new_sd = math.sqrt((sd ** 2) / N)
        sig_level = float(params["sig_level"])
        if type_test == "one-tailed":
            if type_tail == "lower":
                cv = round(norm.ppf(sig_level, loc=mean, scale=new_sd), 5)
                return [("-\infty", cv)]
            else:
                cv = round(norm.ppf(1 - sig_level, loc=mean, scale=new_sd), 5)
                return [(cv, "\infty")]
        else:
            cv_1 = round(norm.ppf(sig_level / 2, loc=mean, scale=new_sd), 5)
            cv_2 = round(norm.ppf(1 - sig_level / 2, loc=mean, scale=new_sd), 5)
            return [("-\infty", cv_1), (cv_2, "\infty")]

    @staticmethod
    def binomial_p_value(params: dict, type_test, type_tail=None):
        num_trials = float(params["num_trials"])
        prob = float(params["population_param_value"])
        print(type_test)
        print(type_tail)
        if type_test == "one-tailed" and type_tail == "upper":
            return round(1 - binom.cdf(float(params["test_stat"]), num_trials, prob), 3)
        else:
            return round(binom.cdf(float(params["test_stat"]), num_trials, prob), 3)

    @staticmethod
    def normal_p_value(params, type_test, type_tail):
        mean = float(params["population_mean"])
        sd = float(params["sd"])
        new_sd = math.sqrt((sd ** 2) / float(params["N"]))
        if type_test == "one-tailed" and type_tail == "upper":
            return round(1 - norm.cdf(float(params["test_stat"]), loc=mean, scale=new_sd), 6)
        else:
            print(mean)
            return round(norm.cdf(float(params["test_stat"]), loc=mean, scale=new_sd), 6)

    @staticmethod
    def calculate_cr(distribution: str, params: dict, type_test: str, type_tail: str):
        cr = None
        match distribution:
            case "binomial": cr = DistributionHT.binomial_cr(params, type_test, type_tail)
            case "normal": cr = DistributionHT.normal_cr(params, type_test, type_tail)
        print(cr)
        method_3 = r"We model the random variable $X$ as"
        if distribution == "binomial":
            method_3 += r"X \approx B(#num_trials#, #popuation_param_value#)."
        elif distribution == "normal":
            method_3 += r"X \approx N(#population_mean#, \frac{#sd#^2}{N}); note that we divide the population variance by the sample size"

        step_3 = "3) Calculate the critical region, assuming the null hypothesis is true"
        if type_test == "one-tailed":
            if type_tail == "lower":
                method_3 = r"""The critical value $x$ is such that $P(X \leq x) = #sig_level#$.
We can find $x$ using the inverse #distribution# function; in this case, $x = """ + str(cr[0][1]) + r"$."
            else:
                diff = 1 - float(params["sig_level"])
                if distribution in DistributionHT.discrete_distributions:
                    method_3 = r"""The critical value $x$ is such that $P(X \geq x) = #sig_level#$.
Since $P(X \geq x) = 1 - P(X \leq x - 1)$ for discrete distributions, we have $P(X \leq x - 1) = """ + str(
                        diff) + "$. Using the inverse #distribution# function, we have $x = " + str(cr[0][0]) + " $."
                else:
                    method_3 = r"""The critical value $x$ is such that $P(X \geq x) = #sig_level#$. Since $P(X \geq x) = 1 - P(X \leq x)$ for continuous distributions, we have $P(X \leq x) = """ + str(
                        diff) + "$. Using the inverse #distribution# function, we have $x = " + str(cr[0][1]) + "$. "
            method_3 += r"""
Therefore the critical region is $[""" + str(cr[0][0]) + ", " + str(cr[0][1]) + "]$"
        else:
            half_prob = round(float(params["sig_level"]) / 2, 5)
            method_3 = r"The critical values $a$, $b$ on either end of the distribution are such that $P(X \leq a) = " + str(
                half_prob) + r"$ and $P(X \geq b) = " + str(half_prob) + r"""$. 
To calculate $a$, we can directly use the inverse #distribution# function; this gives $a = """ + str(cr[0][1]) + "$."
            method_3 += r"To be able to use the inverse #distribution# function to calculate b, we rewrite $P(X \geq b)$ as follows:"
            if distribution in DistributionHT.discrete_distributions:
                method_3 += r"$P(X \geq b) = 1 - P(X \leq b - 1) = " + str(half_prob) + r"\Rightarrow P(X \leq b - 1) = " + str(
                    1 - half_prob) + "$. "
            else:
                method_3 += r"$P(X \geq b) = 1 - P(X \leq b) = " + str(half_prob) + r"\Rightarrow P(X \leq b) = " + str(
                    1 - half_prob) + "$. "
            method_3 += r"$b$ is then calculated to be " + str(cr[1][0]) + ". "
            method_3 += r"Therefore the critical region is $[" + str(cr[0][0]) + ", " + str(
                cr[0][1]) + "] \cup [" + str(cr[1][0]) + ", " + str(cr[1][1]) + "]$"

        return cr, step_3, method_3

    @staticmethod
    def get_p_value(distribution: str, params: dict, type_test: str, type_tail=None):
        step_3 = r"3) Calculate p-value assuming null hypothesis is true"
        p_val = None
        match distribution:
            case "binomial": p_val = DistributionHT.binomial_p_value(params, type_test, type_tail)
            case "normal": p_val = DistributionHT.normal_p_value(params, type_test, type_tail)

        method_3 = r"We model the random variable $X$ as"
        if distribution == "binomial":
            method_3 += r"X \approx B(#num_trials#, #popuation_param_value#)."
        elif distribution == "normal":
            method_3 += r"X \approx N(#population_mean#, \frac{#sd#^2}{N}); note that we divide the population variance by the sample size"

        if type_test == "one-tailed":
            if type_tail == "lower":
                method_3 = r"Use the #distribution# CDF: P(X \leq #test_stat#) = " + str(p_val)
            else:
                if distribution in DistributionHT.discrete_distributions:
                    method_3 = r"Using $P(X \geq x) = 1 - P(X \leq x - 1)$ for discrete distributions, $P(X \geq #test_stat#) = " + str(p_val) + "$"
                else:
                    method_3 = r"Using $P(X \geq x) = 1 - P(X \leq x)$ for continuous distributions, $P(X \geq #test_stat#) = " + str(p_val) + "$"
        else:
            method_3 = r"We need to find the p value at both tails, i.e. $P(X \leq #test_stat#)$ and $P(X \geq #test_stat#)$ and compare each to half the significance level."
            method_3 += r"Using the #distribution# cumulative distribution function, $P(X \leq #test_stat#) = " + str(p_val) + "$"
            diff = 1 - p_val
            if distribution in DistributionHT.discrete_distributions:
                method_3 += r"Using $P(X \geq x) = 1 - P(X \leq x - 1)$ for discrete distributions, $P(X \geq x) = " + str(
                    diff) + "$"
            else:
                method_3 += r"Using $P(X \geq x) = 1 - P(X \leq x)$ for continuous distributions, $P(X \geq x) = " + str(
                    diff) + "$"
        return p_val, step_3, method_3

    @staticmethod
    def get_hypothesis(type_test: str, type_tail=None):
        step_1 = r"1) State the null hypothesis: $H_0$: $#population_param_name# = #population_param_value#$"
        method_1 = r"Under the null hypothesis, we model the random variable $X$ with a #distribution# distribution"
        step_2 = r"2) State the alternative hypothesis: $H_1$: "
        if type_test == "one-tailed":
            if type_tail == "lower":
                step_2 += r"$#population_param_name# < #population_param_value#$"
            else:
                step_2 += r"$#population_param_name# > #population_param_value#$"
            method_2 = r"For a one-tailed test, there is a single critical region"
        else:
            step_2 += r"$#population_param_name# \neq #population_param_value#$"
            method_2 = r"For a two-tailed test, there are two critical regions, one at each end of the distribution representing half the significance level."
        return step_1, method_1, step_2, method_2

    @staticmethod
    def get_test_result_cr(cr, test_stat, type_test, type_tail):
        step_4_CR = "4) Compare test statistic to critical region"
        method_4_CR = r"The test statistic is $#test_stat# "
        if type_test == "one-tailed":
            if type_tail == "lower":
                if test_stat <= cr[0][1]:
                    method_4_CR += r"<= " + str(cr[0][1])
                    result = False
                else:
                    method_4_CR += r"> " + str(cr[0][1])
                    result = True
            else:
                if test_stat >= cr[0][0]:
                    method_4_CR += r">= " + str(cr[0][0])
                    result = False
                else:
                    method_4_CR += r"< " + str(cr[0][0])
                    result = True
        else:
            if test_stat <= cr[0][1]:
                method_4_CR += r"<= " + str(cr[0][1])
                result = False
            elif test_stat >= cr[1][0]:
                method_4_CR += r">= " + str(cr[1][0])
                result = False
            else:
                method_4_CR += str(cr[0][1]) + r" <= " + str(cr[1][0])
                result = True
        if result:
            method_4_CR += "$ so the test statistic is not in the critical region."
        else:
            method_4_CR += "$ so the test statistic is in the critical region."
        return result, step_4_CR, method_4_CR

    @staticmethod
    def get_test_result_p_val(p_val, sig_level, type_test):
        step_4 = "4) Compare to the significance level"
        if type_test == "one-tailed":
            if float(p_val) < float(sig_level):
                method_4 = r"""The p-value is less than the significance level of #sig_level#, so there is sufficient evidence to suggest that the null hypothesis is false in favour of the alternative hypothesis"""
                result = False
            else:
                method_4 = r"""The p-value is greater than the significance level of #sig_level#, so there is insufficient evidence to suggest that the null hypothesis is false"""
                result = True
        else:
            method_4 = r"The p-value at the lower tail is " + str(p_val) + ", the p-value at the upper tail is " + str(1 - p_val) +  "and half the significance level is " + str(float(sig_level) / 2)
            if float(p_val) < float(sig_level) / 2:
                method_4 += r"The p-value at the lower tail is less than half the significance level, so there is sufficient evidence to suggest that the null hypothesis is false in favour of the alternative hypothesis."
                result = False
            elif float(1 - p_val) < float(sig_level) / 2:
                method_4 += r"The p-value at the upper tail is less than half the significance level, so there is sufficient evidence to suggest that the null hypothesis is false in favour of the alternative hypothesis."
                result = False
            else:
                method_4 += r"The p-value at both the upper and lower tails is less than half the significance level, so there is insufficient evidence to suggest that the null hypothesis is false."
                result = True
        return result, step_4, method_4

    @staticmethod
    def binomial_actual_sig_level(type_test, type_tail, cr, params):
        num_trials = float(params["num_trials"])
        prob = float(params["population_param_value"])
        if type_test == "one-tailed":
            if type_tail == "lower":
                actual_sig_level = round(binom.cdf(cr[0][1], num_trials, prob), 5)
                return actual_sig_level, "P(X <= " + str(cr[0][1]) + ") = " + str(actual_sig_level)
            else:
                actual_sig_level = round(
                    1 - binom.cdf(cr[0][0] - 1, num_trials, prob), 5)
                return actual_sig_level, "P(X >= " + str(cr[0][0]) + ") = " + str(actual_sig_level)
        else:
            actual_sig_level = round(
                binom.cdf(cr[0][1], num_trials, prob) + (1 - binom.cdf(cr[1][0] - 1, num_trials, prob)), 5)
            return actual_sig_level, "$P(X <= " + str(cr[0][1]) + ") + P(X >= " + str(cr[1][0]) + ") = " + str(actual_sig_level) + "$"

    @staticmethod
    def normal_actual_sig_level(type_test, type_tail, cr, params):
        mean = float(params["population_mean"])
        sd = float(params["sd"])
        N = float(params["N"])
        new_sd = math.sqrt((sd ** 2) / N)
        if type_test == "one-tailed":
            if type_tail == "lower":
                actual_sig_level = round(norm.cdf(cr[0][1], loc=mean, scale=new_sd), 5)
                return actual_sig_level, "$P(X \leq " + str(cr[0][1]) + ") = " + str(actual_sig_level) + "$"
            else:
                actual_sig_level = round(1 - norm.cdf(cr[0][0], loc=mean, scale=new_sd), 5)
                return actual_sig_level, "$P(X \geq " + str(cr[0][0]) + ") = " + str(actual_sig_level) + "$"
        else:
            actual_sig_level = round(norm.cdf(cr[0][1], loc=mean, scale=new_sd) + 1 - norm.cdf(cr[1][0], loc=mean, scale=new_sd), 5)
            return actual_sig_level, "$P(X \leq " + str(cr[0][1]) + ") + P(X >= " + str(cr[1][0]) + ") = " + str(actual_sig_level) + "$"

    @staticmethod
    def calculate_actual_sig_level(distribution, type_test, type_tail, cr, params):
        step_6 = "6) Calculate the actual significance level of the test"
        method_6 = "The actual significance level is the probability that the test statistic lies within the critical region assuming the null hypothesis: "
        actual_sig_level = None
        sig_calc = ""
        match distribution:
            case "binomial": actual_sig_level, sig_calc = DistributionHT.binomial_actual_sig_level(type_test, type_tail, cr, params)
            case "normal": actual_sig_level, sig_calc = DistributionHT.normal_actual_sig_level(type_test, type_tail, cr, params)
        return actual_sig_level, step_6, method_6 + sig_calc

    @staticmethod
    def distribution(distribution: str, type_test: str, params: dict):
        #
        # Finds which end of distribution to test if one-tailed test
        #
        type_tail = None
        if type_test == "one-tailed":
            type_tail = params["type_tail"]
        #
        # Gets substituted null and alternative hypotheses
        #
        step_1, method_1, step_2, method_2 = DistributionHT.get_hypothesis(type_test, type_tail)
        cr, step_3_CR, method_3_CR = DistributionHT.calculate_cr(distribution, params, type_test, type_tail)
        test_stat = float(params["test_stat"])
        result_CR, step_4_CR, method_4_CR = DistributionHT.get_test_result_cr(cr, test_stat, type_test, type_tail)
        print(result_CR)
        step_5_CR = "5) State result of test"

        if result_CR:
            method_5_CR = "Since the test statistic is not in the critical region, there is insufficient evidence to suggest the null hypothesis is false"
        else:
            method_5_CR = "Since the test statistic is in the critical region, there is sufficient evidence to suggest the null hypothesis is false in favour of the alternative hypothesis"

        p_val, step_3_p, method_3_p = DistributionHT.get_p_value(distribution, params, type_test, type_tail)
        result, step_4_p, method_4_p = DistributionHT.get_test_result_p_val(p_val, params["sig_level"], type_test)
        print(result)
        actual_sig_level, step_6_cr, method_6_cr = DistributionHT.calculate_actual_sig_level(distribution, type_test, type_tail, cr,
                                                                                             params)
        step_5_p, method_5_p = step_6_cr, method_6_cr
        step_5_p = "5" + step_6_cr[1:]
        if type_test == "one-tailed":
            cr_res = f"$[{cr[0][0]}, {cr[0][1]}]$"
        else:
            cr_res = fr"$[{cr[0][0]}, {cr[0][1]}] \cup [{cr[1][0]}, {cr[1][1]}]$"


        result = {"res": {"outcome": str(result),
                          "critical_region": cr_res,
                          "actual_sig_level": str(actual_sig_level)},
                  "critical region": {step_1: method_1,
                                      step_2: method_2,
                                      step_3_CR: method_3_CR,
                                      step_4_CR: method_4_CR,
                                      step_5_CR: method_5_CR,
                                      step_6_cr: method_6_cr},
                  "p value": {step_1: method_1,
                              step_2: method_2,
                              step_3_p: method_3_p,
                              step_4_p: method_4_p,
                              step_5_p: method_5_p}}
        for step, method in list(result["critical region"].items()):
            result["critical region"][
                DistributionTemplates.replace_vals(params, step)] = DistributionTemplates.replace_vals(params, result[
                "critical region"].pop(step))
        for step, method in list(result["p value"].items()):
            result["p value"][DistributionTemplates.replace_vals(params, step)] = DistributionTemplates.replace_vals(
                params, result["p value"].pop(step))
        return result


if __name__ == "__main__":
    print(DistributionHT.distribution("binomial", "two-tailed", {"num_trials":"10","prob":"0.333","X":"1","dp":"2","sig_level":"0.05","test_stat":"1","distribution":"binomial","population_param_name":"p","population_param_value":"0.333","type_test":"two-tailed"}))
