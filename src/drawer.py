from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import ChiSquare
from src.tigramite.tigramite import plotting as tp

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class Drawer():

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.image_path = 'temp/image'

    def draw_1d_distribution(self, val_list, xlabel='', ylabel='', title='', fname='default'):
        sns.displot(val_list, kde=False, color='red', bins=1000)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig("temp/image/{}.pdf".format(fname))
        plt.close('all')
    
    def draw_2d_distribution(self, x_list, y_lists, label_list, x_label, y_label, title, fname):
        assert(len(label_list) == len(y_lists))
        for i, (label, y_list) in enumerate(list(zip(label_list, y_lists))):
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(x_list, y_list, label=label, c=col)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='best')
        plt.title(title)
        plt.savefig('temp/image/{}.pdf'.format(fname))
        plt.close('all')

    def draw_histogram(self, y_lists, legends, groups, x_label, y_label):
        assert(len(y_lists) == len(legends))
        assert(len(y_lists[0]) == len(groups))
        n_hist = len(y_lists); n_group = len(y_lists[0])
        width = 0.1; total_width = width * n_hist

        x = np.arange(n_group); x = x - (total_width - width) / 2
        for i in range(n_hist):
            cur_x = x + i * width; cur_y  = y_lists[i]; cur_legend = legends[i]
            plt.bar(cur_x, cur_y, width=width, label=cur_legend, linewidth=3.0, edgecolor='black')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='best')
        plt.xticks(np.arange(n_group), groups)
        plt.title('{} of interaction mining for {}'.format(y_label, self.dataset))
        plt.savefig('{}/{}_discovery_{}.pdf'.format(self.image_path, self.dataset, y_label))
        plt.close('all')

    def plot_interaction_graph(self, pcmci:'PCMCI', contingency_array:'np.ndarray', fname='default', link_label_fontsize=0):
        var_names = pcmci.var_names
        tp.plot_graph(
            figsize=(8, 6),
            val_matrix=np.ones(contingency_array.shape),
            graph=pcmci.convert_to_string_graph(contingency_array),
            var_names=var_names,
            vmax_edges=1.0,
            node_colorbar_label="auto-G^2",
            link_colorbar_label='G^2',
            link_label_fontsize=link_label_fontsize,
            show_colorbar=False
        )
        plt.savefig("{}/{}.pdf".format(self.image_path, fname), bbox_inches='tight')
        plt.close()