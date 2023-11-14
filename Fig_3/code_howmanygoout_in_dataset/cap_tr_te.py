import json
from itu_p1203 import P1203Standalone
import numpy as np
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split
import scipy
#remove warnings
import warnings
warnings.filterwarnings("ignore")

nr_c = 7
folder_w4='C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/features_and_scores_WIV_all_device_best_models/'
for device in ['hdtv']:#,'phone','uhdtv']:
    individual_features_device = np.load(folder_w4+'all_feat_'+device+'.npy', allow_pickle=True)
    #organize by features for each qoe
    collect_sumbit = []
    collect_sumpsnr = []
    collect_sumssim = []
    collect_sumvmaf = []
    collect_logbit = []
    collect_FTW = []
    collect_SDNdash = []
    collect_videoAtlas = []
    exp_orig = individual_features_device
    #min training bitrate
    bit = []
    for exp in exp_orig:
        for i in range(2, (2 + nr_c * 13 - 1), 13):
            bit.append(float(exp[i]))
    min_bit=np.min(bit)

    for exp in exp_orig:
        bit = []
        logbit = []
        for i in range(2, (2 + nr_c * 13 - 1), 13):
            bit.append(float(exp[i]))
            bit_log = np.log(float(exp[i]) / min_bit)
            logbit.append(bit_log)
        # sumbit
        s_bit = np.array(bit).sum()
        # sumlogbit
        l_bit = np.array(logbit).sum()

        reb = []
        for i in range(1, (1 + nr_c * 13 - 1), 13):
            reb.append(float(exp[i]))
        # sum of all reb
        s_reb = np.array(reb).sum()
        # ave of all reb
        s_reb_ave = np.array(reb).mean()
        # nr of stall
        nr_stall = np.count_nonzero(reb)
        # duration stall+normal
        tot_dur_plus_reb = nr_c * 4 + s_reb

        # psnr
        psnr = []
        for i in range(10, (10 + nr_c * 13 - 1), 13):
            psnr.append(float(exp[i]))
        s_psnr = np.array(psnr).sum()

        # ssim
        ssim = []
        for i in range(11, (11 + nr_c * 13 - 1), 13):
            ssim.append(float(exp[i]))
        s_ssim = np.array(ssim).sum()

        # vmaf
        vmaf = []
        for i in range(12, (12 + nr_c * 13 - 1), 13):
            vmaf.append(float(exp[i]))
        # sum
        s_vmaf = np.array(vmaf).sum()
        # ave
        s_vmaf_ave = np.array(vmaf).mean()

        # is best features for videoAtlas
        # isbest
        isbest = []
        for i in range(9, (9 + nr_c * 13 - 1), 13):
            isbest.append(float(exp[i]))

        is_best = np.array(isbest)
        m = 0
        for idx in range(is_best.size - 1, -1, -1):
            if is_best[idx]:
                m += 2
            rebatl = [0] + reb
            if rebatl[idx] > 0 or is_best[idx] == 0:
                break
        m /= tot_dur_plus_reb
        i = (np.array([2 for i in is_best if i == 0]).sum() + s_reb) / tot_dur_plus_reb

        # differnces
        s_dif_bit = np.abs(np.array(bit[1:]) - np.array(bit[:-1])).sum()
        s_dif_bitlog = np.abs(np.array(logbit[1:]) - np.array(logbit[:-1])).sum()
        s_dif_psnr = np.abs(np.array(psnr[1:]) - np.array(psnr[:-1])).sum()
        s_dif_ssim = np.abs(np.array(ssim[1:]) - np.array(ssim[:-1])).sum()
        s_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).sum()
        a_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).mean()

        # collection
        collect_sumbit.append([s_bit, s_reb, s_dif_bit])
        collect_sumpsnr.append([s_psnr, s_reb, s_dif_psnr])
        collect_sumssim.append([s_ssim, s_reb, s_dif_ssim])
        collect_sumvmaf.append([s_vmaf, s_reb, s_dif_vmaf])

        collect_logbit.append([l_bit, s_reb, s_dif_bitlog])
        collect_FTW.append([s_reb_ave, nr_stall])
        collect_SDNdash.append(
            [s_vmaf_ave, s_reb_ave, a_dif_vmaf])  # without initial stall since we don't have it in our dataset
        collect_videoAtlas.append([s_vmaf_ave, s_reb / tot_dur_plus_reb, nr_stall, m, i])


def fit_linear(all_features, users_scores):
    # multi-linear model fitting
    X = all_features
    y = users_scores

    lm = linear_model.LinearRegression(fit_intercept=False)
    model = lm.fit(X, y)

    alpha = lm.coef_[0]
    beta = lm.coef_[1]
    gamma = lm.coef_[2]

    return [alpha, beta, gamma]
def fit_nonlinear(all_features, users_scores):
    def fun(data, a, b, c, d):
        x1, x2 = data
        y = a * np.exp(-(b * x1 + c) * x2) + d
        return y

    # Fit the curve
    popt, pcov = curve_fit(fun, all_features, users_scores, maxfev=1000000)
    estimated_a, estimated_b, estimated_c, estimated_d = popt
    return estimated_a, estimated_b, estimated_c, estimated_d
def fit_supreg(all_features, users_scores):
    data = np.array(all_features)
    target = np.array(users_scores)

    regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=3,
                             param_grid={'C': [1e-1, 1e0, 1e1, 1e2, 1e3],
                                         'gamma': np.logspace(-2, 2, 15)})
    regressor.fit(data, np.ravel(target))

    return regressor.best_estimator_



scores_hdtv = np.load(folder_w4+'users_scores_'+device+'.npy', allow_pickle=True)
mos_hdtv = np.mean(scores_hdtv, axis=0)
collect_all_features=[collect_sumbit,collect_logbit,collect_sumpsnr,collect_sumssim,collect_sumvmaf,np.array(collect_FTW),collect_SDNdash,collect_videoAtlas]
l = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']
import random

print('w4')
random_array = [random.randint(1, 255) for _ in range(100)]
for rs in random_array:
    for ts in [0.2,0.25,0.3,0.35]:
        #print('rs: '+str(rs)+' ts: '+str(ts))
        collect_all = []
        temp_score_FTW = []
        temp_score_bits = []
        temp_score_logbits = []
        temp_score_psnr = []
        temp_score_ssim = []
        temp_score_vmaf = []
        temp_score_SDNdash = []
        temp_score_videoAtlas = []
        for idx_i,i in enumerate(l):
            collect_temp = []
            all_features = collect_all_features[idx_i]
            #X_train, X_test, indices_train, indices_test = train_test_split(all_features, range(len(all_features)), test_size=0.3, random_state=rs)
            X_train_mos, X_test_mos, y_train_mos, y_test_mos = train_test_split(all_features, mos_hdtv, test_size=ts, random_state=rs)


            if i == 'FTW':
                a, b, c, d=fit_nonlinear((X_train_mos[:, 0], X_train_mos[:, 1]), y_train_mos)
                x1, x2 = X_test_mos[:,0], X_test_mos[:, 1]
                score = a * np.exp(-(b * x1 + c) * x2) + d
                temp_score_FTW.append(score)
            elif i == 'videoAtlas':
                pickle.dump(fit_supreg(X_train_mos, y_train_mos), open('../videoAtlas_mos.pkl', 'wb'))
                with open('../videoAtlas_mos.pkl', 'rb') as handle:
                    pickled_atlas = pickle.load(handle)
                videoAtlasregressor = pickled_atlas  # 0 there is the mdoel,
                temp_score_videoAtlas = videoAtlasregressor.predict(X_test_mos)
            else:
                params = fit_linear(X_train_mos, y_train_mos)
                for single_X_test_mos in X_test_mos:
                    score = np.dot(params, single_X_test_mos)
                    if i=='bit':
                        temp_score_bits.append(score)
                    elif i=='logbit':
                        temp_score_logbits.append(score)
                    elif i=='psnr':
                        temp_score_psnr.append(score)
                    elif i=='ssim':
                        temp_score_ssim.append(score)
                    elif i=='vmaf':
                        temp_score_vmaf.append(score)
                    elif i=='SDNdash':
                        temp_score_SDNdash.append(score)
        srccs=[]
        pear=[]
        rmses=[]
        maes=[]
        collect_all_models=[temp_score_bits,temp_score_logbits,temp_score_psnr,temp_score_ssim,temp_score_vmaf,temp_score_FTW[0],temp_score_SDNdash,temp_score_videoAtlas]
        conta=[]
        for nr_m,model in enumerate(collect_all_models):
            #nr elements bigger than 100 and smaller than 1
            conta.append(len([i for i in model if i>100 or i<1]))
            # calculate srcc
            srccs.append(scipy.stats.spearmanr(model, y_test_mos)[0])
            # calculate plcc
            pear.append(scipy.stats.pearsonr(model, y_test_mos)[0])
            # calculate rmse
            rmses.append(sqrt(mean_squared_error(model, y_test_mos)))
            # calculate mae
            maes.append(mean_absolute_error(model, y_test_mos))
        #print(l)
        print('rs: ' + str(rs) + ' ts: ' + str(ts))
        print(conta)
        #print(conta)
        # print('srccs', srccs)
        # print('pear', pear)
        # print('rmses', rmses)
        # print('maes', maes)
#
#
#
#
#
#
#
##my dataset
import json
from itu_p1203 import P1203Standalone
import numpy as np
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split
def samearray(a,b):
    for i in range(len(a)):
        if a[i]!=b[i]:
            return False
    return True
def index_same_rows(array_of_users):
    # extract index of rows wich are identical across users
    index_identical_rows = []
    for idx_row, row in enumerate(array_of_users[0]):
        identical = True
        for idx_user, user in enumerate(array_of_users):
            if not samearray(row, user[idx_row]):
                identical = False
                break
        if identical:
            index_identical_rows.append(idx_row)
    return index_identical_rows
def fit_linear(all_features, users_scores):
    # multi-linear model fitting
    X = all_features
    y = users_scores

    lm = linear_model.LinearRegression(fit_intercept=False)
    model = lm.fit(X, y)

    alpha = lm.coef_[0]
    beta = lm.coef_[1]
    gamma = lm.coef_[2]

    return [alpha, beta, gamma]
def fit_nonlinear(all_features, users_scores):
    def fun(data, a, b, c, d):
        x1, x2 = data
        y = a * np.exp(-(b * x1 + c) * x2) + d
        return y

    # Fit the curve
    popt, pcov = curve_fit(fun, all_features, users_scores, maxfev=1000000)
    estimated_a, estimated_b, estimated_c, estimated_d = popt
    return estimated_a, estimated_b, estimated_c, estimated_d
def fit_supreg(all_features, users_scores):
    data = np.array(all_features)
    target = np.array(users_scores)

    regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=3,
                             param_grid={'C': [1e-1, 1e0, 1e1, 1e2, 1e3],
                                         'gamma': np.logspace(-2, 2, 15)})
    regressor.fit(data, np.ravel(target))

    return regressor.best_estimator_
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
nr_c = 4

array_of_users=[]
mydataset_folder='C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/features_and_scores_mydataset/dataset'
for file in os.listdir(mydataset_folder):
    if file.endswith(".xlsx"):
        df=pd.read_excel(mydataset_folder+'/'+file)
        #remove last column
        df=df.iloc[:,:-1]
        array_of_users.append(df.values)
idxs=index_same_rows(array_of_users)
#index not present in idxs
idxs_not_present=[i for i in range(len(array_of_users[0])) if i not in idxs]

#extract rows which are identical across users
array_of_users_clean=[]
for user in array_of_users:
    array_of_users_clean.append(np.delete(user,idxs_not_present,axis=0))


#organize by features for each qoe
collect_sumbit = []
collect_sumpsnr = []
collect_sumssim = []
collect_sumvmaf = []
collect_logbit = []
collect_FTW = []
collect_SDNdash = []
collect_videoAtlas = []
#min training bitrate
bit = []
for exp in array_of_users_clean[0]:
    for i in range(2, (2 + nr_c * 10 - 1), 10):
        bit.append(float(exp[i]))
min_bit=np.min(bit)

for exp in array_of_users_clean[0]:

    bit = []
    logbit = []
    for i in range(2, (2 + nr_c * 10 - 1), 10):
        bit.append(float(exp[i]))
        bit_log = np.log(float(exp[i]) / min_bit)
        logbit.append(bit_log)
    # sumbit
    s_bit = np.array(bit).sum()
    # sumlogbit
    l_bit = np.array(logbit).sum()

    reb = []
    for i in range(1, (1 + nr_c * 10 - 1), 10):
        reb.append(float(exp[i]))
    # sum of all reb
    s_reb = np.array(reb).sum()
    # ave of all reb
    s_reb_ave = np.array(reb).mean()
    # nr of stall
    nr_stall = np.count_nonzero(reb)
    # duration stall+normal
    tot_dur_plus_reb = nr_c * 4 + s_reb

    # psnr
    psnr = []
    for i in range(7, (7 + nr_c * 10 - 1), 10):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()

    # ssim
    ssim = []
    for i in range(8, (8 + nr_c * 10 - 1), 10):
        ssim.append(float(exp[i]))
    s_ssim = np.array(ssim).sum()

    # vmaf
    vmaf = []
    for i in range(9, (9 + nr_c * 10 - 1), 10):
        vmaf.append(float(exp[i]))
    # sum
    s_vmaf = np.array(vmaf).sum()
    # ave
    s_vmaf_ave = np.array(vmaf).mean()

    # is best features for videoAtlas
    # isbest
    isbest = []
    for i in range(6, (6 + nr_c * 10 - 1), 10):
        isbest.append(float(exp[i]))

    is_best = np.array(isbest)
    m = 0
    for idx in range(is_best.size - 1, -1, -1):
        if is_best[idx]:
            m += 2
        rebatl = [0] + reb
        if rebatl[idx] > 0 or is_best[idx] == 0:
            break
    m /= tot_dur_plus_reb
    i = (np.array([2 for i in is_best if i == 0]).sum() + s_reb) / tot_dur_plus_reb

    # differnces
    s_dif_bit = np.abs(np.array(bit[1:]) - np.array(bit[:-1])).sum()
    s_dif_bitlog = np.abs(np.array(logbit[1:]) - np.array(logbit[:-1])).sum()
    s_dif_psnr = np.abs(np.array(psnr[1:]) - np.array(psnr[:-1])).sum()
    s_dif_ssim = np.abs(np.array(ssim[1:]) - np.array(ssim[:-1])).sum()
    s_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).sum()
    a_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).mean()

    # collection
    collect_sumbit.append([s_bit, s_reb, s_dif_bit])
    collect_sumpsnr.append([s_psnr, s_reb, s_dif_psnr])
    collect_sumssim.append([s_ssim, s_reb, s_dif_ssim])
    collect_sumvmaf.append([s_vmaf, s_reb, s_dif_vmaf])

    collect_logbit.append([l_bit, s_reb, s_dif_bitlog])
    collect_FTW.append([s_reb_ave, nr_stall])
    collect_SDNdash.append(
        [s_vmaf_ave, s_reb_ave, a_dif_vmaf])  # without initial stall since we don't have it in our dataset
    collect_videoAtlas.append([s_vmaf_ave, s_reb / tot_dur_plus_reb, nr_stall, m, i])


array_of_users=[]
mydataset_folder='C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/features_and_scores_mydataset/dataset'
for file in os.listdir(mydataset_folder):
    if file.endswith(".xlsx"):
        array_of_users.append(pd.read_excel(mydataset_folder+'/'+file))

#for every user in the array extract the scores
users_scores=[]
for user in array_of_users:
    users_scores.append(user['score'].values.tolist())

#extract idx from users_scores
for u in users_scores:
    for i in idxs_not_present:
        u.pop(i)


mos_mydata = np.mean(users_scores, axis=0)
collect_all_features=[collect_sumbit,collect_logbit,collect_sumpsnr,collect_sumssim,collect_sumvmaf,np.array(collect_FTW),collect_SDNdash,collect_videoAtlas]
l = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']
#random array of 30 numbers
import random
random_array = [random.randint(1, 255) for _ in range(100)]
for rs in random_array:
    for ts in [0.2,0.25,0.3,0.35]:
        #print('rs: '+str(rs)+' ts: '+str(ts))
        collect_all = []
        temp_score_FTW = []
        temp_score_bits = []
        temp_score_logbits = []
        temp_score_psnr = []
        temp_score_ssim = []
        temp_score_vmaf = []
        temp_score_SDNdash = []
        temp_score_videoAtlas = []
        for idx_i,i in enumerate(l):
            collect_temp = []
            all_features = collect_all_features[idx_i]
            #X_train, X_test, indices_train, indices_test = train_test_split(all_features, range(len(all_features)), test_size=0.3, random_state=rs)
            X_train_mos, X_test_mos, y_train_mos, y_test_mos = train_test_split(all_features, mos_mydata, test_size=ts, random_state=rs)


            if i == 'FTW':
                a, b, c, d=fit_nonlinear((X_train_mos[:, 0], X_train_mos[:, 1]), y_train_mos)
                x1, x2 = X_test_mos[:,0], X_test_mos[:, 1]
                score = a * np.exp(-(b * x1 + c) * x2) + d
                temp_score_FTW.append(score)
            elif i == 'videoAtlas':
                pickle.dump(fit_supreg(X_train_mos, y_train_mos), open('../videoAtlas_mos.pkl', 'wb'))
                with open('../videoAtlas_mos.pkl', 'rb') as handle:
                    pickled_atlas = pickle.load(handle)
                videoAtlasregressor = pickled_atlas  # 0 there is the mdoel,
                temp_score_videoAtlas = videoAtlasregressor.predict(X_test_mos)
            else:
                params = fit_linear(X_train_mos, y_train_mos)
                for single_X_test_mos in X_test_mos:
                    score = np.dot(params, single_X_test_mos)
                    if i=='bit':
                        temp_score_bits.append(score)
                    elif i=='logbit':
                        temp_score_logbits.append(score)
                    elif i=='psnr':
                        temp_score_psnr.append(score)
                    elif i=='ssim':
                        temp_score_ssim.append(score)
                    elif i=='vmaf':
                        temp_score_vmaf.append(score)
                    elif i=='SDNdash':
                        temp_score_SDNdash.append(score)

        pear=[]
        srccs=[]
        rmses=[]
        maes=[]
        collect_all_models=[temp_score_bits,temp_score_logbits,temp_score_psnr,temp_score_ssim,temp_score_vmaf,temp_score_FTW[0],temp_score_SDNdash,temp_score_videoAtlas]
        conta=[]
        for nr_m,model in enumerate(collect_all_models):
            #nr elements bigger than 100 and smaller than 1
            conta.append(len([i for i in model if i>100 or i<1]))
            #calculate srcc
            srccs.append(scipy.stats.spearmanr(model,y_test_mos)[0])
            #calculate plcc
            pear.append(scipy.stats.pearsonr(model,y_test_mos)[0])
            #calculate rmse
            rmses.append(sqrt(mean_squared_error(model,y_test_mos)))
            #calculate mae
            maes.append(mean_absolute_error(model,y_test_mos))

        print('rs: '+str(rs)+' ts: '+str(ts))
        print(conta)
        #print(conta)
    # print(l)
    # print('srccs',srccs)
    # print('pear',pear)
    # print('rmses',rmses)
    # print('maes',maes)
