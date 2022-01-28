import numpy as np

from cgn.translator.multiparameter import MultiParameter

from ..test_problem.parameter_fixtures import x_parameter, z_parameter, u_parameter


def test_multi_parameter(x_parameter, z_parameter, u_parameter):
    param_list = [x_parameter, z_parameter, u_parameter]
    multi_parameter = MultiParameter([x_parameter, z_parameter, u_parameter])
    nparam = len(param_list)
    assert multi_parameter.nparams == nparam
    pos_i = 0
    for i in range(nparam):
        assert multi_parameter.position(i) == pos_i
        pos_i += param_list[i].dim
    concatenated_mean = np.concatenate([x_parameter.mean, z_parameter.mean, u_parameter.mean])
    assert np.isclose(concatenated_mean, multi_parameter.mean).all()
    concatenated_lb = np.concatenate([x_parameter.lb, z_parameter.lb, u_parameter.lb])
    assert np.isclose(concatenated_lb, multi_parameter.lb).all()
    concatenated_regop = multi_parameter.regop
    nsum = 0
    rsum = 0
    for param in param_list:
        nsum += param.dim
        rsum += param.rdim
    assert concatenated_regop.dim == nsum
    assert concatenated_regop.rdim == rsum

