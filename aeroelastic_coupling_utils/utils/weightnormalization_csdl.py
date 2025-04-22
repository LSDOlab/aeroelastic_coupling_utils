# from csdl import Model
# from python_csdl_backend import Simulator
# import csdl

import csdl_alpha as csdl

def weight_normalization(weight_array: csdl.Variable, column_scaling_vec: csdl.Variable=None, normalization_eps: float=0):
    if column_scaling_vec is not None:
        dist_weights_scaled = csdl.einsum(weight_array, column_scaling_vec, subscripts='ij,i->ji')
    else:
        dist_weights_scaled = weight_array

    # and compute the columnwise sum, before expanding the result
    dist_weights_scaled_colsums = csdl.sum(dist_weights_scaled, axes=(1,))
    dist_weights_scaled_colsums_exp = csdl.expand(dist_weights_scaled_colsums, dist_weights_scaled.shape, 'i->ij')
    # lastly we divide the scaled distance weights by the column sums
    dist_weights = csdl.div(dist_weights_scaled, dist_weights_scaled_colsums_exp + normalization_eps)

    return dist_weights