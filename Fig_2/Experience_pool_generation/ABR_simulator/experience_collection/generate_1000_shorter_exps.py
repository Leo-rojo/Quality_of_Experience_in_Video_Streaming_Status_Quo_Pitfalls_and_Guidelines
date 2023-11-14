import pickle
import numpy as np
import random
from random import randint
import os
os.chdir('C:\\Users\\leona\\Desktop\\iQoE_web\\video_generations\\Experience_pool_generation\\ABR_simulator\\experience_collection')
#PARAMS
nr_of_exp=1000
lb_perc_stall_ev=30
nr_of_stalled_exp=nr_of_exp*lb_perc_stall_ev/100
nr_c=4
synthetic_experiences=np.load('./experiences_with_features.npy')



#modifiy bitrate form bit/s to kbit/s like in W4
for i in range(len(synthetic_experiences)):
    for k in range(len(synthetic_experiences[0])):
        synthetic_experiences[i][k][2]=synthetic_experiences[i][k][2]/1000

#inf to 50 for psnr--limit inf
synthetic_experiences[synthetic_experiences>1e308]=50

#collect experiences in form of elaborated features
random.seed(42)

list_of_exp=[]
list_of_exp_for_models=[]
while nr_of_stalled_exp>0:
    random_trace = randint(0, len(synthetic_experiences) - 1)
    random_chunk = randint(0, len(synthetic_experiences[0]) - nr_c)
    # list of exp to be used for training
    ch = []
    for c in range(nr_c):
        ch = ch + synthetic_experiences[random_trace][random_chunk + c].tolist()
    ch.append(random_chunk)

    ###########filtra
    rep_sum = []
    sta_sum = []
    for i in range(0, 40, 10):
        rep = ch[i]
        rep_sum.append(ch[i])
        sta = ch[i + 1]
        sta_sum.append(ch[i + 1])
    # for the reason in NB_1 I have to shift the stall and put 0 at the end
    sta_sum2=[sta_sum[i] for i in range(1,len(sta_sum))]
    sta_sum2.append(0)
    if np.sum(sta_sum2) < 2 * nr_c / 2 + 1 and np.sum(sta_sum2)>0.9:
        list_of_exp.append(ch)
        nr_of_stalled_exp -= 1

contamaxrep=0
while nr_of_exp-(lb_perc_stall_ev*10)>0:
    random_trace=randint(0, len(synthetic_experiences)-1)
    random_chunk=randint(0, len(synthetic_experiences[0])-nr_c)
    #list of exp to be used for training
    ch=[]
    for c in range(nr_c):
        ch=ch + synthetic_experiences[random_trace][random_chunk+c].tolist()
    ch.append(random_chunk)

    ###########filtra
    rep_sum = []
    sta_sum = []
    for i in range(0, 40, 10):
        rep = ch[i]
        rep_sum.append(ch[i])
        sta = ch[i + 1]
        sta_sum.append(ch[i + 1])
    # for the reason in NB_1 I have to shift the stall and put 0 at the end
    sta_sum2 = [sta_sum[i] for i in range(1, len(sta_sum))]
    sta_sum2.append(0)
    if np.sum(sta_sum2)<2*nr_c/2+1:
        list_of_exp.append(ch)
        nr_of_exp-=1

        # if contamaxrep<100:
        #     list_of_exp.append(ch)
        #     nr_of_exp-=1
        #     #conta se maxrep without stall and change
        #     #contamaxrep+=1
        # else:
        #     #check that it is not maxrep without stall
        #         list_of_exp.append(ch)
        #         nr_of_exp -= 1


#####################################################
###NB_1
#park generate rep-sta---rep-sta---rep-sta but sta is referred to the chunk before and this one so
#I need to shift left all the stalls and remove the stall at the end of the video because it does not have
#any effect

#at the end I've added the initial chunk number
#list_of_exp=np.load('iQoE_synth_exp.npy')
for exp in list_of_exp:
    stalls_positions=range(1, len(exp), 10)
    for k in stalls_positions: #iterate across all stalls of the experience
        if k==stalls_positions[-1]:
            exp[k]=0
        else:
            exp[k]=exp[k+10]
np.save('iQoE_synth_exp',list_of_exp)












