import numpy as np 
from matplotlib import pyplot as plt

def autolabel(rects, ax, height_ratio):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height_ratio * height,
                '%.1f%%' % (100 * height),
                ha='center', va='bottom', weight="bold")

def draw_accuracy_histogram():
    """ 
    This function plots the accuracy of PCMCI (proper background knowledge).
    The x-axis should be the optional parameter (e.g., partition-day, pc_alpha or alpha_level)
        For each x_axis, there are two columns, each of which represents precision or recall.
        The y-axis should be the %.
    """
    nontemporal_precision_list = [0.264, 0.245, 0.237, 0.229, 0.222]
    nontemporal_recall_list = [0.840, 0.880, 0.906, 0.909, 0.927]
    temporal_precision_list = [0.942, 0.930, 0.923, 0.928, 0.922]
    temporal_recall_list = [0.875, 0.912, 0.932, 0.936, 0.960]
    temporal_spatial_precision_list = [0.942, 0.929, 0.924, 0.929, 0.922]
    temporal_spatial_recall_list = [0.880, 0.917, 0.933, 0.937, 0.963]
    xtick_list = ['6', '9', '12', '15', '20']
    assert(len(nontemporal_precision_list) == len(nontemporal_recall_list))

    fig, ax = plt.subplots(figsize =(12, 8), nrows=2, ncols=2)

    barWidth = 0.25
    br1 = np.arange(len(nontemporal_precision_list))
    br2 = [x + barWidth for x in br1]

    rects_list = []
    ax_list = [ax[0, 0], ax[0, 0], ax[0, 1], ax[0, 1], ax[1, 0], ax[1, 0]]

    rects_list.append(ax[0, 0].bar(br1, nontemporal_precision_list, color ='r', width = barWidth,
        edgecolor ='grey', label ='Precision'))
    rects_list.append(ax[0, 0].bar(br2, nontemporal_recall_list, color ='g', width = barWidth,
        edgecolor ='grey', label ='Recall'))
    ax[0, 0].set_xlabel('Partitioned days', fontweight ='bold', fontsize = 10)
    ax[0, 0].set_ylabel('Accuracy', fontweight ='bold', fontsize = 10)
    ax[0, 0].set_xticks([r + 0.5 * barWidth for r in range(len(nontemporal_precision_list))], xtick_list)
    ax[0, 0].legend(loc='upper left')

    rects_list.append(ax[0, 1].bar(br1, temporal_precision_list, color ='r', width = barWidth,
        edgecolor ='grey', label ='Precision'))
    rects_list.append(ax[0, 1].bar(br2, temporal_recall_list, color ='g', width = barWidth,
        edgecolor ='grey', label ='Recall'))
    ax[0, 1].set_xlabel('Partitioned days', fontweight ='bold', fontsize = 10)
    ax[0, 1].set_ylabel('Accuracy (Temporal)', fontweight ='bold', fontsize = 10)
    ax[0, 1].set_xticks([r + 0.5 * barWidth for r in range(len(nontemporal_precision_list))], xtick_list)
    ax[0, 1].legend(loc='upper left')

    rects_list.append(ax[1, 0].bar(br1, temporal_spatial_precision_list, color ='r', width = barWidth,
        edgecolor ='grey', label ='Precision'))
    rects_list.append(ax[1, 0].bar(br2, temporal_spatial_recall_list, color ='g', width = barWidth,
        edgecolor ='grey', label ='Recall'))
    ax[1, 0].set_xlabel('Partitioned days', fontweight ='bold', fontsize = 10)
    ax[1, 0].set_ylabel('Accuracy (Temporal+Spatial)', fontweight ='bold', fontsize = 10)
    ax[1, 0].set_xticks([r + 0.5 * barWidth for r in range(len(nontemporal_precision_list))], xtick_list)
    ax[1, 0].legend(loc='upper left')

    for index, (rects, ax) in enumerate(zip(rects_list, ax_list)):
        if index % 2 == 0:
            autolabel(rects, ax, 0.5)
        else:
            autolabel(rects, ax, 0.6)

    plt.show()

draw_accuracy_histogram()