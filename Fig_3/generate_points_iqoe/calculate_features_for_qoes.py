import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
nr_c = 4
path_iQoE='users'
maes=[]
rmses=[]
scores_more_users=[]
videos_more_users=[]
for fold in os.listdir(path_iQoE):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        user_folder = path_iQoE+'/user_' + identifier

        ##train data
        # save dictonary idx_original-score
        a=[]
        a2=[]
        d_train = {}
        #exp_orig_train_t = []
        #scaled_exp_orig_train_t = []
        #take first 10 of train
        with open(user_folder + '/Scores_' + identifier + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_train[int(key)] = val
                if len(d_train)==10:
                    break
        a.append([int(i) for i in list(d_train.values())])
        a2.append([int(i) for i in list(d_train.keys())])

        ###baseline data
        b=[]
        b2=[]
        d_baselines={}
        #exp_orig_train_b = []
        #scaled_exp_orig_train_b = []
        with open(user_folder + '/Scores_baseline' + identifier + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_baselines[int(key)] = val
        b.append([int(i) for i in list(d_baselines.values())])
        b2.append([int(i) for i in list(d_baselines.keys())])
        scores_more_users.append(a[0]+b[0])
        videos_more_users.append(a2[0]+b2[0])
idx_col_train = np.load(path_iQoE+'/original_database/idx_col_train.npy')
exp_orig = np.load(path_iQoE+'/original_database/synth_exp_train.npy')
scaled_exp_orig = np.load(path_iQoE+'/original_database/X_train_scaled.npy')

exp_orig_train = []
scaled_exp_orig_train = []
idx_zero_buffer=[]
merge_dictionary = {**d_train, **d_baselines}
for x in merge_dictionary.keys():  # array of original idxs
    idx_standard= np.where(idx_col_train == x)
    exp_orig_train.append(exp_orig[idx_standard])
    scaled_exp_orig_train.append(scaled_exp_orig[idx_standard])

for idx,exp in enumerate(exp_orig_train):
    exp=exp[0]
    reb = []
    for i in range(1, (1 + nr_c * 10 - 1), 10):
        reb.append(float(exp[i]))
    # sum of all reb
    s_reb = np.array(reb).sum()
    if s_reb==0:
        idx_zero_buffer.append(idx)


path_iQoE='users'
maes=[]
rmses=[]
scores_more_users=[]
videos_more_users=[]
for fold in os.listdir(path_iQoE):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        user_folder = path_iQoE+'/user_' + identifier

        ##train data
        # save dictonary idx_original-score
        a=[]
        a2=[]
        d_train = {}
        #exp_orig_train_t = []
        #scaled_exp_orig_train_t = []
        #take first 20 of train
        with open(user_folder + '/Scores_' + identifier + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_train[int(key)] = val
                if len(d_train)==10:
                    break
        a.append([int(i) for i in list(d_train.values())])
        a2.append([int(i) for i in list(d_train.keys())])

        ###baseline data
        b=[]
        b2=[]
        d_baselines={}
        #exp_orig_train_b = []
        #scaled_exp_orig_train_b = []
        with open(user_folder + '/Scores_baseline' + identifier + '.txt') as f:
            for line in f:
                val = line.split()[-1]
                nextline = next(f)
                key = nextline.split()[-1]
                d_baselines[int(key)] = val
        b.append([int(i) for i in list(d_baselines.values())])
        b2.append([int(i) for i in list(d_baselines.keys())])
        scores_more_users.append(a[0]+b[0])
        videos_more_users.append(a2[0]+b2[0])

moses=np.mean(np.array(scores_more_users),axis=0)


train_zero_buff=np.array(exp_orig_train)[idx_zero_buffer]
points=[]
mos_no_rebuf_train=moses[idx_zero_buffer]
for idx,exp in enumerate(np.array(exp_orig_train)[idx_zero_buffer]):
    exp=exp[0]
    # psnr
    psnr = []
    for i in range(7, (7 + nr_c * 10 - 1), 10):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()
    mos=mos_no_rebuf_train[idx]
    points.append([s_psnr,mos])


##test
########test#############
scores_more_users_test=[]
videos_more_users_test=[]
for fold in os.listdir(path_iQoE):
    if fold.split('_')[0]=='user':
        identifier=fold.split('_')[-1]
        user_folder = path_iQoE+'/user_' + identifier
        #print(identifier)

        ##test data
        #save dictonary idx_original-score
        d_test = {}
        exp_orig_test=[]
        scaled_exp_orig_test=[]
        with open(user_folder+'/Scores_test_'+identifier+'.txt') as f:
            for line in f:
               val = line.split()[-1]
               nextline=next(f)
               key = nextline.split()[-1]
               d_test[int(key)] = val
        scores_more_users_test.append([int(i) for i in d_test.values()])
        videos_more_users_test.append([i for i in d_test.keys()])

idx_col_test = np.load(path_iQoE + '/original_database/idx_col_test.npy')
exp_orig = np.load(path_iQoE+'/original_database/synth_exp_test.npy')
scaled_exp_orig = np.load(path_iQoE+'/original_database/X_test_scaled.npy')
for x in d_test.keys():  # array of original idxs
    idx_standard = np.where(idx_col_test == x)
    exp_orig_test.append(exp_orig[idx_standard])
    scaled_exp_orig_test.append(scaled_exp_orig[idx_standard])
moses_test=np.mean(np.array(scores_more_users_test),axis=0)

idx_zero_buffer=[]
for idx,exp in enumerate(exp_orig_test):
    exp=exp[0]
    reb = []
    for i in range(1, (1 + nr_c * 10 - 1), 10):
        reb.append(float(exp[i]))
    # sum of all reb
    s_reb = np.array(reb).sum()
    if s_reb==0:
        idx_zero_buffer.append(idx)

test_zero_buf=np.array(exp_orig_test)[idx_zero_buffer]
mos_no_rebuf_test=moses_test[idx_zero_buffer]
for idx,exp in enumerate(np.array(exp_orig_test)[idx_zero_buffer]):
    exp=exp[0]
    # psnr
    psnr = []
    for i in range(7, (7 + nr_c * 10 - 1), 10):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()
    mos=mos_no_rebuf_test[idx]
    points.append([s_psnr,mos])

np.save('../points_iqoe_dataset.npy',points)

exp_iqoe_zero_buf=np.concatenate((train_zero_buff,test_zero_buf),axis=0)
np.save('../exp_iqoe_zero_buf.npy',exp_iqoe_zero_buf)
final_mos_zero_buff=np.concatenate((mos_no_rebuf_train,mos_no_rebuf_test),axis=0)
np.save('../mos_iQoE_zero_buff.npy',final_mos_zero_buff)
