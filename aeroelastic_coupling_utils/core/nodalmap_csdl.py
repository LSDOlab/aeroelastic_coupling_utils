import csdl_alpha as csdl
import numpy as np

from aeroelastic_coupling_utils.utils.weightfunctions_csdl import WeightFunctions
from aeroelastic_coupling_utils.utils.distancecalculation_csdl import distance_calculation
from aeroelastic_coupling_utils.utils.weightnormalization_csdl import weight_normalization

class NodalMap:
    def __init__(self, weight_eps: float=1., weight_func_name: str='Gaussian', weight_to_be_normalized: bool=True, normalization_eps:float=0):
        # define constant model parameters
        self.weight_eps = weight_eps
        self.weight_func = weight_func_name
        self.weight_to_be_normalized = weight_to_be_normalized
        self.normalization_eps = normalization_eps

    def evaluate(self, input_mesh: csdl.Variable, output_mesh: csdl.Variable, column_scaling_vec: csdl.Variable=None):
        # calculate distances between input and output mesh nodes
        distance_array = distance_calculation(input_mesh, output_mesh)

        # initialize the inverse distance weight function and calculate the inverse-distance weight array
        weight_function = WeightFunctions(weight_func_name=self.weight_func, weight_eps=self.weight_eps)
        ID_weight_array = weight_function.evaluate(distance_array)

        # normalize the inverse-distance weight array with the column scaling vector
        if self.weight_to_be_normalized:
            output_ID_weight_array = weight_normalization(ID_weight_array, column_scaling_vec=column_scaling_vec, 
                                                          normalization_eps=self.normalization_eps)
        else:
            output_ID_weight_array = ID_weight_array
        return output_ID_weight_array
        

if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)

    in_shape_test = (5, 3)
    out_shape_test = (8, 3)
    weight_eps_test_val = 3.

    # define random solid and fluid mesh coordinates for testing
    rng_solid_mesh = np.random.random(in_shape_test)
    rng_fluid_mesh = np.random.random(out_shape_test)

    # create test model that wraps the NodalMap model
    recorder = csdl.Recorder()
    recorder.start()
    test_model_Gaussian = NodalMap(weight_eps=weight_eps_test_val, weight_func_name='Gaussian', weight_to_be_normalized=True)
    # construct mesh csdl variables 
    solid_mesh_csdl = csdl.Variable(value=rng_solid_mesh)
    fluid_mesh_csdl = csdl.Variable(value=rng_fluid_mesh)
    # run Gaussian function test model
    nodal_map = test_model_Gaussian.evaluate(solid_mesh_csdl, fluid_mesh_csdl)
    recorder.stop()

    recorder.execute()

    print("Projection map:")
    print(nodal_map.value)
