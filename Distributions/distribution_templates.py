class DistributionTemplates:
    @staticmethod
    def params_msg(params: dict) -> str:
        params_msg = ""
        for k, v in params:
            params_msg += "\\(" + k + " = " + str(v) + "\\)\n"
        return params_msg

    @staticmethod
    def discrete(dist_name: str, type_func: str, pmf_formula: str, params: dict, method: str) -> str:
        return f"""
        The formula for the {dist_name} distribution {type_func} is \\( ${pmf_formula} \\).\n
        Inputted values are:\n{DistributionTemplates.params_msg(params)}
        \n \\[ {method} \\] 
        """