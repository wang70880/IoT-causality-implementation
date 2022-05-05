import numpy as np 
from matplotlib import pyplot as plt

def draw_accuracy_histogram():
    """ 
    This function plots the accuracy of PCMCI (proper background knowledge).
    The x-axis should be the optional parameter (e.g., partition-day, pc_alpha or alpha_level)
        For each x_axis, there are two columns, each of which represents precision or recall.
        The y-axis should be the ratio %.
    """
    precision_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    recall_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    xtick_list = ['5', '10', '15', '20', '30']
    assert(len(precision_list) == len(recall_list))

    fig = plt.subplots(figsize =(12, 8))

    barWidth = 0.25
    br1 = np.arange(len(precision_list))
    br2 = [x + barWidth for x in br1]

    plt.bar(br1, precision_list, color ='r', width = barWidth,
        edgecolor ='grey', label ='Precision')
    plt.bar(br2, recall_list, color ='g', width = barWidth,
        edgecolor ='grey', label ='Recall')

    plt.xlabel('Partitioned days', fontweight ='bold', fontsize = 15)
    plt.ylabel('Accuracy ratio', fontweight ='bold', fontsize = 15)
    plt.xticks([r + 0.5 * barWidth for r in range(len(precision_list))], xtick_list)

    plt.legend()
    plt.show()

draw_accuracy_histogram()