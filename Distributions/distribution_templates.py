class DistributionTemplates:
    @staticmethod
    def replace_vals(params: dict, method: str) -> str:
        del_1 = 0
        del_2 = 1
        while del_2 < len(method):
            if method[del_1] == "#":
                if method[del_2] == "#":
                    key = method[del_1 + 1: del_2]
                    method = method[:del_1] + params[key] + method[del_2 + 1:]
                    del_1 += len(params[key])
                    del_2 = del_1 + 1
                else:
                    del_2 += 1
            else:
                del_1 += 1
                del_2 = del_1 + 1
        return method

    @staticmethod
    def method_template(dist_name: str, type_func: str, pmf_formula: str, params: dict, method: str) -> str:
        return f"""The formula for the {dist_name} distribution {type_func} is $ {pmf_formula} $.\n{DistributionTemplates.replace_vals(params, method)}"""