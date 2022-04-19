# Imports
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys

from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

# Test 1
np.random.seed(1)
var_names = ['x0', 'x1']
col1 = np.random.randint(low=0, high=2, size=2000)
np.random.seed(2)
col2 = np.random.randint(low=0, high=2, size=2000)
data = np.transpose(np.vstack((col1, col2)))
for t in range(1, 200):
    data[t, 0] = (data[t-1, 0] + 1) % 2
    data[t, 1] = data[t-1, 0]
dataframe = pp.DataFrame(data, var_names=var_names)
#tp.plot_timeseries(dataframe, figsize=(10,4)); plt.show()



df = pd.read_csv('./hh101-act1-test', sep = ' ')
data = df.to_numpy()
var_names = list(df.columns)
dataframe = pp.DataFrame(data, var_names=var_names)
cmi_symb = CMIsymb(significance='shuffle_test', n_symbs=None)

pcmci_cmi_symb = PCMCI(
    dataframe=dataframe, 
    cond_ind_test=cmi_symb,
    verbosity=1)

all_parents = pcmci_cmi_symb.run_pc_stable(selected_links=None,
                                 tau_min=0,
                                 tau_max=2,
                                 save_iterations=False,
                                 pc_alpha=0.005,
                                 max_conds_dim=None,
                                 max_combinations=10)
print(all_parents)

#results = pcmci_cmi_symb.run_pcmci(tau_max=1, pc_alpha=0.2, alpha_level = 0.01)
#tp.plot_time_series_graph(
#    figsize=(6, 4),
#    val_matrix=results['val_matrix'],
#    graph=results['graph'],
#    var_names=var_names,
#    link_colorbar_label='MCI',
#    ); plt.show()