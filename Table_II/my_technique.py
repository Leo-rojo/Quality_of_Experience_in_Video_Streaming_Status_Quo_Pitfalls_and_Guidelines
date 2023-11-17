import numpy as np
import os
import csv
os.chdir('/Table_II')
#yin vmaf
yin_psnr_model=np.load('trained_models_allabr/psnr_allabr.npy')
rdos=69.58
rate=67.90
alpha, beta, gamma = yin_psnr_model
x3=0 #no change in quality
x1=50*7 #max hypotetical sum psnr
reb_hypo_rdos=(alpha*x1-rdos+gamma*x3)/beta
reb_hypo_rate=(alpha*x1-rate+gamma*x3)/beta
print(reb_hypo_rdos)
print(reb_hypo_rate)
print(reb_hypo_rdos-reb_hypo_rate) #difference is in indipendent of the max hypotetical
#results are in secs



#videoAtlas
import numpy as np
a=np.load('predictions_Rdos/scores_videoAtlas_Rdos.npy')
f=np.load('features_and_scores_qoes_Rdos/feat_va.npy',allow_pickle=True)#[30]
nr_c=7
videoatlas_model=np.load('trained_models_allabr/videoAtlas_allabr.pkl',allow_pickle=True)
rdos=76.72
rate=69.44
s_vmaf_ave=88 #max possible vmaf ave

#brute force rdos
closest_difference_rdos = float('inf')  # Initialize the difference to a large value
closest_val = None
for val in np.arange(0.1,10,0.1):#val in np.arange(0, 10, 0.1):
    #for rounds in range(100):
    target_value = rdos
    #reb = [val, val, 0, 0, 0, 0, 0]
    #reb = np.random.randint(2, size=7)
    reb = [1,0,0,0,0,0,0]
    reb=[i*val for i in reb]
    nr_stall = np.count_nonzero(reb)
    s_reb = np.array(reb).sum()

    is_best = np.array([0, 0, 0, 0, 0, 0, 0]) #all chunks are best
    m = 0
    for idx in range(is_best.size - 1, -1, -1):
        if is_best[idx]:
            m += 2
        rebatl = [0] + reb
        if rebatl[idx] > 0 or is_best[idx] == 0:
            break
    tot_dur_plus_reb = nr_c * 4 + s_reb
    m /= tot_dur_plus_reb
    i = (np.array([2 for i in is_best if i == 0]).sum() + s_reb) / tot_dur_plus_reb


    hypot_ifs=[s_vmaf_ave, s_reb / tot_dur_plus_reb, nr_stall, m, i]
    #print(hypot_ifs)

    temp_score_videoAtlas = videoatlas_model.predict(np.array(hypot_ifs).reshape(1,-1))
    #print(temp_score_videoAtlas)

    # Calculate the difference between the result and the target value
    difference = abs(temp_score_videoAtlas - target_value)

    # Check if the current difference is closer than the previously found closest difference
    if difference < closest_difference_rdos:
        closest_difference_rdos = difference
        closest_val_rdos = reb
        VA_score= temp_score_videoAtlas
# print(closest_val_rdos)
# print(closest_difference_rdos)
# print(VA_score)
############################
#brute force rate
closest_difference_rate = float('inf')  # Initialize the difference to a large value
closest_val = None
for val in np.arange(0.1,10,0.1):#val in np.arange(0, 10, 0.1):
    #for rounds in range(150):
    target_value = rate
    #reb = [val, val, 0, 0, 0, 0, 0]
    #reb = np.random.randint(2, size=7)
    reb=[1,1,0,0,0,0,0]
    reb = [i * val for i in reb]
    nr_stall = np.count_nonzero(reb)
    s_reb = np.array(reb).sum()

    is_best = np.array([0, 0, 0, 0, 0, 0, 0]) #all chunks are best
    m = 0
    for idx in range(is_best.size - 1, -1, -1):
        if is_best[idx]:
            m += 2
        rebatl = [0] + reb
        if rebatl[idx] > 0 or is_best[idx] == 0:
            break
    tot_dur_plus_reb = nr_c * 4 + s_reb
    m /= tot_dur_plus_reb
    i = (np.array([2 for i in is_best if i == 0]).sum() + s_reb) / tot_dur_plus_reb


    hypot_ifs=[s_vmaf_ave, s_reb / tot_dur_plus_reb, nr_stall, m, i]

    temp_score_videoAtlas = videoatlas_model.predict(np.array(hypot_ifs).reshape(1,-1))
    #print(temp_score_videoAtlas)

    # Calculate the difference between the result and the target value
    difference = abs(temp_score_videoAtlas - target_value)
    #print(difference)

    # Check if the current difference is closer than the previously found closest difference
    if difference < closest_difference_rate:
        #print(temp_score_videoAtlas)
        closest_difference_rate = difference
        closest_val_rate = reb

print(closest_val_rdos)
print(closest_val_rate)
# print(closest_difference_rdos)
# print(closest_difference_rate)


import numpy as np
import os
import csv
os.chdir('/Table_II')
#lstm
rdos=95.39
buffer=93.25
nr_c=7
#brute force rdos
closest_difference_rdos = float('inf')  # Initialize the difference to a large value
closest_val = None
for val in np.arange(0.95,1.05,0.01):#val in np.arange(0, 10, 0.1):
    target_value = rdos
    #for rounds in range(100):
    reb = np.random.randint(2, size=7)
    reb=[i*val for i in reb]
    reb = [val, 0, 0, 0, 0, 0, 0]
    exp=[]
    #create exp
    bit=[458.496,834.436,612.09,1099.872,1933.474,1088.98,1816.116]
    res=[147456,147456,147456,230400,921600,230400,921600]
    qps=[13.415,10.648,15.181,14.716,14.952,14.795,14.915]
    for i in range(0, nr_c):
        exp.append([reb[i], qps[i], bit[i], res[i], 30])
    with open('output_file_my_technique/exp_' + str(val) +".csv", "a+", newline='') as f: #+ '_round_'+str(rounds)
        writer = csv.writer(f)
        writer.writerow(['SD', 'QP', 'BR', 'RS', 'FR'])
        for i in exp:
            writer.writerow(i)
    #evaluate exp
    os.chdir('/Table_II/output_file_my_technique')
    os.system('biQPS exp_'+ str(val)+'.csv') #+ '_round_'+str(rounds)
    #os.system('biQPS exp6.csv') #+ '_round_'+str(rounds)
    os.chdir('/Table_II')
    scoresbiqps = []
    with open('output_file_my_technique/output.txt') as f:
        for line in f.readlines()[1:]:
            onefive = float(line.split('\t')[-1])
            X_std = (onefive - 1) / (5 - 1)
            X_scaled = X_std * (100 - 1) + 1

    # Calculate the difference between the result and the target value
    difference = abs(X_scaled - target_value)
    #print(difference)
    # Check if the current difference is closer than the previously found closest difference
    if difference < closest_difference_rdos:
        closest_difference_rdos = difference
        closest_val_rdos = val

#delete content of output_file_my_technique
import os
os.chdir('/Table_II/output_file_my_technique')
for file in os.listdir():
    if file.endswith(".csv") or file.endswith(".txt"):
        os.remove(file)
os.chdir('/Table_II')

closest_difference_buffer = float('inf')  # Initialize the difference to a large value
closest_val = None
for val in np.arange(0.93,1.06,0.01):#val in np.arange(0, 10, 0.1):
    target_value = buffer
    #for rounds in range(100):
    reb = np.random.randint(2, size=7)
    reb=[i*val for i in reb]
    reb = [val, 0, 0, 0, 0, 0, 0]
    exp=[]
    #create exp
    bit=[458.496,834.436,612.09,1099.872,1933.474,1088.98,1816.116]
    res=[147456,147456,147456,230400,921600,230400,921600]
    qps=[13.415,10.648,15.181,14.716,14.952,14.795,14.915]
    for i in range(0, nr_c):
        exp.append([reb[i], qps[i], bit[i], res[i], 30])
    with open('output_file_my_technique/exp_' + str(val) +".csv", "a+", newline='') as f: #+ '_round_'+str(rounds)
        writer = csv.writer(f)
        writer.writerow(['SD', 'QP', 'BR', 'RS', 'FR'])
        for i in exp:
            writer.writerow(i)
    #evaluate exp
    os.chdir('/Table_II/output_file_my_technique')
    os.system('biQPS exp_'+ str(val)+'.csv') #+ '_round_'+str(rounds)
    #os.system('biQPS exp6.csv') #+ '_round_'+str(rounds)
    os.chdir('/Table_II')
    scoresbiqps = []
    with open('output_file_my_technique/output.txt') as f:
        for line in f.readlines()[1:]:
            onefive = float(line.split('\t')[-1])
            X_std = (onefive - 1) / (5 - 1)
            X_scaled = X_std * (100 - 1) + 1

    # Calculate the difference between the result and the target value
    difference = abs(X_scaled - target_value)
    #print(difference)
    # Check if the current difference is closer than the previously found closest difference
    if difference < closest_difference_buffer:
        closest_difference_buffer = difference
        closest_val_buffer = val
    #print(difference)

print(closest_val_rdos)
print(closest_val_buffer)
# print(closest_difference_rdos)
# print(closest_difference_buffer)
# import os
# os.chdir('C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/Table_II')
# import numpy as np
# a=np.load('predictions_Rdos/scores_biqps_Rdos.npy')