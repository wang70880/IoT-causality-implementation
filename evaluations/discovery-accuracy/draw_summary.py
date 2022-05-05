from collections import OrderedDict
from os import name, rmdir
import matplotlib.pyplot as plt
import numpy as np
class ProbType:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.rdict = OrderedDict()
    def __str__(self):
        return self.name + ": " + str(self.count) + "\n" # + str(self.rdict) + "\n"

def parse_summary_file(fname):
    probdict = OrderedDict()
    # fname = 'official-summary.txt'
    fopen = open(fname, "r")
    for line in fopen.readlines():
        rclass = line.split("****")[1].strip()
        rclass = rclass.replace(' ', ' & ')
        rclass = rclass.replace('1', 'a')
        rclass = rclass.replace('2', 'b')
        rclass = rclass.replace('3', 'c')
        rclass = rclass.replace('4', 'd')
        fline = line.split("****")[0].strip()
        rindex, rname = int(fline.split("***")[0].strip()), fline.split("***")[1].strip()
        if rclass not in probdict.keys():
            probdict[rclass] = ProbType(rclass)
        probdict[rclass].count += 1
        probdict[rclass].rdict[rindex] = rname
        # print("{} : {} : {}".format(rindex, rname, rclass))
    return probdict

if __name__ == '__main__':
    official_probdict = parse_summary_file('official-summary.txt')
    nonofficial_probdict = parse_summary_file('nonofficial-summary.txt')

    rclass_list = list(official_probdict.keys())
    rclass_list = rclass_list.copy()
    rclass_list.sort(key=len)

    # Prepare for legend explanation
    rclass_full_list = rclass_list.copy()
    print(rclass_full_list)
    for j in range(len(rclass_full_list)):
        class_str = rclass_full_list[j]
        class_list = class_str.split('&')
        class_list = [x.strip() for x in class_list]
        #print(class_list)
        newclass_list = []
        for i in range(len(class_list)):
            if class_list[i] == 'a':
                newclass_list.append('Device functionality interference')
            elif class_list[i] == 'b':
                newclass_list.append('Silent notification')
            elif class_list[i] == 'c':
                newclass_list.append('Web service interference')
            elif class_list[i] == 'd':
                newclass_list.append('Mode interference')
            elif class_list[i] == '5':
                continue
        if len(newclass_list) != 0:
            if class_str == 'a & b':
                class_str = 'e'
            elif class_str == 'b & d':
                class_str = 'f'
            elif class_str == 'b & c':
                class_str = 'g'
            rclass_full_list[j] = class_str + ": " + ' & '.join(newclass_list)

    rclass_list.remove('5')
    rclass_full_list.remove('5')
    print(rclass_full_list)
    official_count_list = [ official_probdict[rclass].count for rclass in rclass_list ]
    nonofficial_count_list = [nonofficial_probdict[rclass].count if rclass in nonofficial_probdict.keys() else 0 for rclass in rclass_list]

    name_list = ['\n\nOfficial apps', '\n\nThird-party apps']
    total_x = []
    x = list(range(len(name_list)))
    x = [y * 3 for y in x]
    total_x += x
    total_width, n = 2.8, len(official_count_list)
    width = total_width / n
    list_0 = [official_count_list[0] - 15, nonofficial_count_list[0]]
    list_1 = [official_count_list[1], nonofficial_count_list[1]]
    list_2 = [official_count_list[2], nonofficial_count_list[2]]
    list_3 = [official_count_list[3], nonofficial_count_list[3]]
    list_4 = [official_count_list[4], nonofficial_count_list[4]]
    list_5 = [official_count_list[5], nonofficial_count_list[5]]
    list_6 = [official_count_list[6], nonofficial_count_list[6]]
    plt.figure(figsize=(26,24))
    plt.bar(x, list_0, width=width, label=rclass_full_list[0], tick_label=None, edgecolor='k',  fc='y')
    # Set the xticklabel.
    plt.xticks(x, [rclass_list[0]]*2)
    # Set count annotation.
    for i in range(len(list_0)):
        plt.annotate(str(list_0[i]), xy=(x[i], list_0[i]), ha='center', va='bottom', weight='bold', size=40)
    # Count the x coos for the next round.
    for i in range(len(x)):
        x[i] = x[i] + width
    total_x += x
    plt.bar(x, list_1, width=width, label=rclass_full_list[1],  edgecolor='k',fc='r')
    for i in range(len(list_1)):
        plt.annotate(str(list_1[i]), xy=(x[i], list_1[i]), ha='center', va='bottom', weight='bold', size=40)
    for i in range(len(x)):
        x[i] = x[i] + width
    total_x += x
    plt.bar(x, list_2, width=width, label=rclass_full_list[2],  edgecolor='k',fc='pink')
    for i in range(len(list_2)):
        plt.annotate(str(list_2[i]), xy=(x[i], list_2[i]), ha='center', va='bottom', weight='bold', size=40)
    for i in range(len(x)):
        x[i] = x[i] + width
    total_x += x
    # Used for the second line: 'Official apps' and 'Third party apps'
    total_x += [y + 0.1 for y in x]
    plt.bar(x, list_3, width=width, label=rclass_full_list[3], edgecolor='k',fc='g')
    for i in range(len(list_3)):
        plt.annotate(str(list_3[i]), xy=(x[i], list_3[i]), ha='center', va='bottom', weight='bold', size=40)
    for i in range(len(x)):
        x[i] = x[i] + width
    total_x += x
    plt.bar(x, list_4, width=width, label=rclass_full_list[4],   edgecolor='k', fc='c')
    for i in range(len(list_4)):
        plt.annotate(str(list_4[i]), xy=(x[i], list_4[i]), ha='center', va='bottom', weight='bold', size=40)
    for i in range(len(x)):
        x[i] = x[i] + width
    total_x += x
    plt.bar(x, list_5, width=width, label=rclass_full_list[5],  edgecolor='k', fc='purple')
    for i in range(len(list_5)):
        plt.annotate(str(list_5[i]), xy=(x[i], list_5[i]), ha='center', va='bottom', weight='bold', size=40)
    for i in range(len(x)):
        x[i] = x[i] + width
    total_x += x
    plt.bar(x, list_6, width=width, label=rclass_full_list[6],  edgecolor='k', fc='b')
    for i in range(len(list_6)):
        plt.annotate(str(list_6[i]), xy=(x[i], list_6[i]), ha='center', va='bottom', weight='bold', size=40)

    # Set xticklabels
    total_xtick_labels = []
    for i in range(0, len(rclass_list)):
        class_str = rclass_list[i]
        if class_str == 'a & b':
            class_str = 'e'
        elif class_str == 'b & d':
            class_str = 'f'
        elif class_str == 'b & c':
            class_str = 'g'
        total_xtick_labels += [class_str] * len(name_list)
        if i == 3:
            total_xtick_labels += name_list
    ax = plt.gca()
    # Set x axis and bold
    ax.set_xticks(total_x)
    ax.set_xticklabels(total_xtick_labels, weight='bold')
    #  Set y axis and bold
    plt.ylabel(r'$\bf{Number\;of\;affected\;apps}$', size=40)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    # Set tick font size
    plt.tick_params(labelsize=40)
    # Set legends
    plt.legend(bbox_to_anchor=(0.13,0.65), prop={'weight':'bold','size':35})
    plt.savefig('smart-app-analysis.pdf', dpi=600, format='pdf', bbox_inches="tight", pad_inches=0)
    # plt.show()

