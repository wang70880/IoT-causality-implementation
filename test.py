from src.tigramite.tigramite.independence_tests import CMIsymb, ChiSquare
import numpy as np

cond_ind_test = ChiSquare()
obs_array = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1, 1]])
print(obs_array.shape)
print(cond_ind_test.get_dependence_measure(obs_array, None))