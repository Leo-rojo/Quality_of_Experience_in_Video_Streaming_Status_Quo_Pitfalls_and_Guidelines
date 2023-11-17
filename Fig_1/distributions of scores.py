import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import cm

#input data
mydataset_folder='../dataset_120'
w4dataset_folder='../allfeat_allscores_WIV'
hdtv_scores=np.load(w4dataset_folder+'/users_scores_hdtv.npy', allow_pickle=True)
hdtv_scores=hdtv_scores.tolist()

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

nr_c = 4
array_of_users=[]

for file in os.listdir(mydataset_folder):
    if file.endswith(".xlsx"):
        array_of_users.append(pd.read_excel(mydataset_folder+'/'+file))

#for every user in the array extract the scores
users_scores=[]
for user in array_of_users:
    users_scores.append(user['score'].values.tolist())



#calculate the occurence of each score
#empty counter

for c,i in enumerate([users_scores,hdtv_scores]):
    fig = plt.figure(figsize=(20, 10), dpi=100)
    counter = Counter()
    for single_score in range(len(i)):
        counter = counter+Counter(i[single_score])
    sum_values = sum(counter.values())
    for key in counter:
        counter[key] = (counter[key] / sum_values)*100
    #counter=dict(sorted(counter.items(), key=lambda item: item[1],reverse=True))
    #print(counter)
    #sort counter based on key
    counter=dict(sorted(counter.items(), key=lambda item: item[0]))
    print(counter)
    #save the counter in xls file
    df = pd.DataFrame.from_dict(counter, orient='index')
    df.to_excel('counter'+['mydata','w4data'][c]+'.xlsx', header=False)

    #remove element whose value is lower than 0.5
    # for key in list(counter.keys()):
    #     if counter[key]<1:
    #         del counter[key]

    #plot the distribution of scores
    plt.bar(range(len(counter)), list(counter.values()), align='center')
    #plot ticks every ten elements
    myax=[i for i in range(101) if i % 10 == 0]
    myax[0]=1
    plt.xticks([i-1 for i in myax], [str(i) for i in myax])
    #plt.xticks(range(len(counter)), list(counter.keys()))
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['0', '1', '2', '3', '4', '5', '6', '7'])

    #plt.title('Distribution of scores '+['mydata','w4data'][c])
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.gcf().subplots_adjust(left=0.15)  # add space left
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    ax = plt.gca()
    #plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])
    plt.ylabel('% of occurrences', fontdict=font_axes_titles)
    plt.xlabel('Score', fontdict=font_axes_titles)
    ax.tick_params(axis='x', which='major', width=7, length=24)
    ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
    plt.ylim(0, 7.1)
    # plt.show()
    plt.savefig('all_scores_distribution'+['mydata','w4'][c]+'.pdf', bbox_inches='tight')
    plt.close()

    if c==0:
        counter_mydata=counter
    else:
        counter_w4data=counter

#print counter values for 1,10,20,30,40,50,60,70,80,90,100
print('counter_mydata',counter_mydata[1],counter_mydata[10],counter_mydata[20],counter_mydata[30],counter_mydata[40],counter_mydata[50],counter_mydata[60],counter_mydata[70],counter_mydata[80],counter_mydata[90],counter_mydata[100])
print('counter_w4data',counter_w4data[1],counter_w4data[10],counter_w4data[20],counter_w4data[30],counter_w4data[40],counter_w4data[50],counter_w4data[60],counter_w4data[70],counter_w4data[80],counter_w4data[90],counter_w4data[100])

print('sum_counter_mydata',sum([counter_mydata[1],counter_mydata[10],counter_mydata[20],counter_mydata[30],counter_mydata[40],counter_mydata[50],counter_mydata[60],counter_mydata[70],counter_mydata[80],counter_mydata[90],counter_mydata[100]]))
print('sum_counter_w4data',sum([counter_w4data[1],counter_w4data[10],counter_w4data[20],counter_w4data[30],counter_w4data[40],counter_w4data[50],counter_w4data[60],counter_w4data[70],counter_w4data[80],counter_w4data[90],counter_w4data[100]]))

#print sum of counter values for 1,10,20,30,40,50,60,70,80,90,100
#calculate variance of counter_mydata.values() every 10 elements
variance_mydata=[]
variance_w4data=[]
for i in range(0,100,10):
    variance_mydata.append(np.var(list(counter_mydata.values())[i:i+10]))
    variance_w4data.append(np.var(list(counter_w4data.values())[i:i+10]))
print('variance 10')
print(variance_mydata)
print(variance_w4data)

variance_mydata_20=[]
variance_w4data_20=[]
for i in range(0,100,20):
    variance_mydata_20.append(np.var(list(counter_mydata.values())[i:i+20]))
    variance_w4data_20.append(np.var(list(counter_w4data.values())[i:i+20]))
print('variance 20')
print(variance_mydata_20)
print(variance_w4data_20)
#print counter_mydata sorted on value
counter_mydata=dict(sorted(counter_mydata.items(), key=lambda item: item[1],reverse=True))
print('counter_mydata',counter_mydata)
counter_w4data=dict(sorted(counter_w4data.items(), key=lambda item: item[1],reverse=True))
print('counter_w4data',counter_w4data)
#print difference between value of first element vs the third
print('difference between value of first element vs the third')
print(counter_mydata[100]-counter_mydata[70])
print(counter_w4data[50]-counter_w4data[70])




