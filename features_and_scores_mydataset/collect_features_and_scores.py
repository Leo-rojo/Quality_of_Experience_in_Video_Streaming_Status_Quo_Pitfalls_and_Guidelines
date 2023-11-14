import os
import csv
import json
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd
nr_c = 4

array_of_users=[]
dataset_folder='C:/Users/leona/Desktop/QoE_combination/features_and_scores_mydataset/dataset'
for file in os.listdir(dataset_folder):
    if file.endswith(".xlsx"):
        array_of_users.append(pd.read_excel(dataset_folder+'/'+file))

#for every user in the array extract the scores
users_scores=[]
for user in array_of_users:
    users_scores.append(user['score'].values.tolist())

#calculate mos and plot distribtuion of scores
mos=np.mean(users_scores,axis=0)
import matplotlib.pyplot as plt
plt.hist(mos, bins=5)
plt.savefig('mos.png')
plt.close()






for device in ['hdtv','phone','uhdtv']:
    if not os.path.exists('features_and_scores_qoes_'+device):
        os.makedirs('features_and_scores_qoes_'+device)
    else:
        #delete folders
        shutil.rmtree('features_and_scores_qoes_'+device)
        os.makedirs('features_and_scores_qoes_'+device)
    individual_features_device = np.load('all_feat_'+device+'.npy', allow_pickle=True)
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

    np.save('features_and_scores_qoes_'+device+'/feat_bit', collect_sumbit)
    np.save('features_and_scores_qoes_'+device+'/feat_psnr', collect_sumpsnr)
    np.save('features_and_scores_qoes_'+device+'/feat_ssim', collect_sumssim)
    np.save('features_and_scores_qoes_'+device+'/feat_vmaf', collect_sumvmaf)
    np.save('features_and_scores_qoes_'+device+'/feat_logbit', collect_logbit)
    np.save('features_and_scores_qoes_'+device+'/feat_ftw', collect_FTW)
    np.save('features_and_scores_qoes_'+device+'/feat_sdn', collect_SDNdash)
    np.save('features_and_scores_qoes_'+device+'/feat_va', collect_videoAtlas)