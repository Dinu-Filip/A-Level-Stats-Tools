from fastapi import FastAPI
from Distributions.binomial import Binomial
from Distributions.normal import Normal
from Distributions.chi_squared import ChiSquared
from pydantic import BaseModel
from DataAnalysis.bivariate import BivariateAnalysis
from DataAnalysis.univariate import UnivariateAnalysis
from HypothesisTests.distributional_parameters import DistributionHT
from HypothesisTests.goodness_of_fit import GoodnessOfFit

app = FastAPI()


@app.get("/Binomial/{type_func}")
async def binomial(type_func, n: str, p: str, dp: str, x: str | None = None, x_1: str | None = None,
                   x_2: str | None = None, P: str | None = None):
    result = {}
    if type_func == "pmf":
        result["pmf"] = Binomial.pmf(n, p, x, dp)
    elif type_func == "cdf":
        result["cdf"] = Binomial.cdf(n, p, x_1, x_2, dp)
    else:
        result["inv"] = Binomial.inverse_binomial(n, p, P, dp)
    result["mean"] = Binomial.mean(n, p, dp)
    result["variance"] = Binomial.variance(n, p, dp)
    result["skewness"] = Binomial.skewness(n, p, dp)
    result["graph_data"] = Binomial.binomial_bars(n, p)
    return result


@app.get("/Normal/{type_func}")
async def normal(type_func: str, mu: str, sigma: str, dp: str, x_1: str | None = None, x_2: str | None = None,
                 P: str | None = None):
    result = {}
    if type_func == "cdf":
        result["cdf"] = Normal.cdf(mu, sigma, x_1, x_2, dp)
    else:
        result["inv"] = Normal.inverse_normal(mu, sigma, P, dp)
    result["graph_data"] = Normal.normal_points(mu, sigma, x_1, x_2)
    return result


@app.get("/Chi-squared/{type_func}")
async def normal(type_func: str, df: str, dp: str, x_1: str | None = None, x_2: str | None = None,
                 P: str | None = None):
    result = {}
    if type_func == "cdf":
        result["cdf"] = ChiSquared.cdf(x_1, x_2, df, dp)
    else:
        result["inv"] = ChiSquared.inverse_chi_squared(df, P, dp)
    result["mean"] = ChiSquared.mean(df)
    result["variance"] = ChiSquared.variance(df)
    result["skewness"] = ChiSquared.skewness(df, dp)
    result["graph_data"] = ChiSquared.chi_squared_points(df, x_1, x_2)
    return result


class Item(BaseModel):
    xVals: list
    yVals: list
    dp: str

@app.post("/univariate/")
async def univariate_analysis(vals: Item):
    data_vals = vals.model_dump()
    return UnivariateAnalysis.generate_results(data_vals["xVals"], data_vals["yVals"], data_vals["dp"])

@app.post("/bivariate/")
async def bivariate_analysis(vals: Item):
    data_vals = vals.model_dump()
    return BivariateAnalysis.generate_results(data_vals["xVals"], data_vals["yVals"], data_vals["dp"])


@app.post("/dist-test/")
async def dist_param_test(params: dict[str, str]):
    return DistributionHT.distribution(params["distribution"], params["type_test"], params)


@app.post("/gof-test/")
async def gof_test(params: dict[str, str]):
    return GoodnessOfFit.goodness_of_fit(params)
