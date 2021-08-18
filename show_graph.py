import numpy as np
import matplotlib.pyplot as plt

with open("/home/nuc/DL_study/result.txt", 'r') as file:
    count = 0
    list_data = []
    dict_data = []
    dict_data_cc = []
    dict_data_ac = []

    strings = file.readline()
    for i, j in enumerate(strings):
        if (j == '}'):
            count = count + 1

    for i in range(0, count):
        cut = strings.split('}')[i] + '}'
        list_data.append(cut)

    for i in range(0, count):
        dict_data.append(eval(list_data[i]))

    #print(dict_data[0]['caculate count'])

    for i in range(0, count):
        #dict_data_cc.append(dict_data[i]['caculate count'])
        dict_data_ac.append(dict_data[i]['test acc'])

    x = np.arange(count)
    plt.bar(x, dict_data_ac)

    plt.show()