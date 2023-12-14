import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
from collections import Counter

#input data
re=np.load('generated_real_exp.npy')

colori=cm.get_cmap('tab10').colors
font_axes_titles = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 60,
                        }
font_title = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 60,
                        }
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 60}
plt.rc('font', **font_general)
colori=cm.get_cmap('tab10').colors

#const
nr_c=4
two_features=[]
all_rep=[]
all_reb=[]
for exp in re:
    rep = []
    for i in range(0, (nr_c * 10 - 1), 10):
        all_rep.append(float(exp[i]))
        rep.append(float(exp[i]))
    s_rep = np.array(rep).sum()

    bit = []
    for i in range(2, (2 + nr_c * 10 - 1), 10):
        bit.append(float(exp[i]))
    # sumbit
    s_bit = np.array(bit).sum()

    reb = []
    for i in range(1, (1 + nr_c * 10 - 1), 10):
        all_reb.append(float(exp[i]))
        reb.append(float(exp[i]))
    # sum of all reb
    s_reb = np.array(reb).sum()

    # differnces
    s_dif_bit = np.abs(np.array(bit[1:]) - np.array(bit[:-1])).sum()
    s_dif_rep = np.abs(np.array(rep[1:]) - np.array(rep[:-1])).sum()

    # collection
    two_features.append([s_rep, s_reb, s_dif_rep])
two_features=np.array(two_features).reshape(-1,3)

####################################################
import random
VIDEO_BIT_RATE = [0.235, 0.375, 0.560, 0.750, 1.050, 1.750, 2.350, 3, 4.3, 5.8, 8.1, 11.6, 16.8]
IDXS=[0,1,2,3,4,5,6,7,8,9,10,11,12]
all_reb_max=np.max(all_reb)
all_reb_min=np.min(all_reb)
all_reb_uni=[]
all_rep_uni=[]
for i in range(len(two_features)):
    stall_values_3 = [random.uniform(all_reb_min, all_reb_max) for _ in range(3)]
    quality_values_4 = [random.randint(0, 12) for _ in range(4)]
    for k in stall_values_3:
        all_reb_uni.append(k)
    for k in quality_values_4:
        all_rep_uni.append(k)
##################################################################

#sns
import seaborn as sns
for nr,v in enumerate([[all_rep,all_rep_uni],[all_reb,all_reb_uni]]):
    xl=['Representation index','Stall duration, s']
    data_variable1 = v[0]
    data_variable2 = v[1]

    # For nr = 1, create a bar plot (discrete density)
    if nr == 0:
        fig = plt.figure(figsize=(20, 10), dpi=100)
        #count occurence of data_variable1 and data_variable2
        counter1 = Counter(data_variable1)
        counter2 = Counter(data_variable2)


        # Calculate the total number of elements
        total_elements1 = sum(counter1.values())
        total_elements2 = sum(counter2.values())

        # Calculate the PDF using the Counter object
        pdf1 = {element: count / total_elements1 for element, count in counter1.items()}
        pdf2 = {element: count / total_elements2 for element, count in counter2.items()}
        #sort pdf1 and pdf2 based on key
        pdf1=dict(sorted(pdf1.items(), key=lambda item: item[0]))
        pdf2=dict(sorted(pdf2.items(), key=lambda item: item[0]))
        print('values fig 2a')
        print('reds: ',pdf1)
        print('blues: ',pdf2)
        plt.bar([i for i in range(len(counter1))], list(pdf1.values()), color='red',fill=True,edgecolor='black',width=0.3,linewidth=5)
        plt.bar([i+0.3 for i in range(len(counter2))], list(pdf2.values()), color='blue',fill=True,edgecolor='black', width=0.3,linewidth=5)
        plt.xticks([i+0.15 for i in range(len(counter1))], [str(i) for i in range(1, 14, 1)])
        ax = plt.gca()
        ax.tick_params(axis='x', which='major', width=7, length=24, pad=5)
        plt.yticks([i for i in np.arange(0, 0.3, 0.05)],['0.00', '0.05', '0.10', '0.15', '0.20', '0.25'])
        plt.ylim(0,0.21)

    else:
        # For other values of nr, create a kernel density plot
        import seaborn as sns

        fig= plt.figure(figsize=(20, 10), dpi=100)
        a=sns.kdeplot(data_variable2, label='Variable 2', color='blue',linewidth=7,linestyle='-')
        b=sns.kdeplot(data_variable1, label='Variable 1', color='red', linewidth=7,linestyle='-')
        ax = plt.gca()
        ax.tick_params(axis='x', which='major', width=7, length=24)
        plt.xlim(0,5)
        plt.yticks([i for i in np.arange(0, 3.1, 0.5)],['0.00', '0.05', '0.10', '0.15', '0.20', '0.25','0.30'])
        plt.ylim(0,3.1)

    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.gcf().subplots_adjust(left=0.15)  # add space left
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
    plt.xlabel(xl[nr])
    plt.ylabel('Density')
    #save plot as pdf
    plt.savefig('fig_'+str(nr)+'.pdf', bbox_inches='tight')
    plt.savefig('fig_'+str(nr)+'.png', bbox_inches='tight')
    plt.close()

max(all_reb_uni)
#sum every three all reb_uni
all_reb_uni_sum=[]
for i in range(0,len(all_reb_uni),3):
    all_reb_uni_sum.append(np.sum(all_reb_uni[i:i+3]))
all_reb_uni_sum=np.array(all_reb_uni_sum)
