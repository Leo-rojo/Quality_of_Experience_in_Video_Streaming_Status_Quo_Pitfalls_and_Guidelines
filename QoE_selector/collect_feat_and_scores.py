import os
import csv
import json
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

nr_c = 7
max_psnr=0
# individual_scores_hdtv=np.load('users_scores_hdtv.npy',allow_pickle=True)
# individual_features_hdtv=np.load('all_feat_hdtv.npy',allow_pickle=True)
# individual_scores_phone=np.load('users_scores_phone.npy',allow_pickle=True)
# individual_features_phone=np.load('all_feat_phone.npy',allow_pickle=True)
# individual_features_uhdtv=np.load('all_feat_uhdtv.npy',allow_pickle=True)
# individual_scores_uhdtv=np.load('users_scores_uhdtv.npy',allow_pickle=True)
#
# #calculate mos
# mos_hdtv=np.mean(individual_scores_hdtv,axis=0)
# mos_phone=np.mean(individual_scores_phone,axis=0)
# mos_uhdtv=np.mean(individual_scores_uhdtv,axis=0)

for device in ['hdtv','phone','uhdtv']:
    if not os.path.exists('features_and_scores_qoes_'+device):
        os.makedirs('features_and_scores_qoes_'+device)
    else:
        #delete folders
        shutil.rmtree('features_and_scores_qoes_'+device)
        os.makedirs('features_and_scores_qoes_'+device)
    individual_features_device = np.load('allfeat_allscores/all_feat_'+device+'.npy', allow_pickle=True)
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
            if float(exp[i])>max_psnr:
                max_psnr=float(exp[i])
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


    #Lstm features
    feat_each_video_lstm=[]
    min_qps=50
    max_bit=0

    max_h=0
    max_w=0
    for count, exp in enumerate(exp_orig):
        save_each_exp = []
        reb = []
        for i in range(1, (1 + nr_c * 13 - 1), 13):
            reb.append(float(exp[i]))
        bit = []
        for i in range(2, (2 + nr_c * 13 - 1), 13):
            if float(exp[i])>max_bit:
                max_bit=float(exp[i])
            bit.append(float(exp[i]))
        height = []
        for i in range(8, (8 + nr_c * 13 - 1), 13):
            if float(exp[i])>max_h:
                max_h=float(exp[i])
            height.append(float(exp[i]))
        width = []
        for i in range(7, (7 + nr_c * 13 - 1), 13):
            if float(exp[i])>max_w:
                max_w=float(exp[i])
            width.append(float(exp[i]))
        qps = []
        for i in range(5, (5 + nr_c * 13 - 1), 13):
            if float(exp[i])<min_qps:
                min_qps=float(exp[i])
            qps.append(float(exp[i]))

        for i in range(0, nr_c):
            save_each_exp.append([reb[i], qps[i], bit[i], height[i]*width[i],30])
        feat_each_video_lstm.append(save_each_exp)
    print('max values')
    print(min_qps)
    print(max_bit)
    print(max_h)
    print(max_w)

    #_, _, _, indices_test = train_test_split(exp_orig, range(len(exp_orig)), test_size=0.3, random_state=42)
    for count,exp in enumerate(feat_each_video_lstm):
        with open('features_and_scores_qoes_'+device+'/exp' + str(count) + ".csv", "a+", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['SD', 'QP', 'BR', 'RS', 'FR'])
            for i in exp:
                writer.writerow(i)
        with open('features_and_scores_qoes_'+device+'/filelist.txt', 'a+') as f:
            #if count in indices_test:
            f.write('features_and_scores_qoes_'+device+'/exp' + str(count) + '.csv' + '\n')

    #p1203 features
    nr_chunks=7
    nr_feat_ec=13
    chunk_length=4
    tot_nr_feat=nr_chunks*nr_feat_ec

    if device == 'hdtv':
        vd='100cm'
        ds='1920x1080'
        device_json='tv'
    elif device == 'phone':
        vd='20cm'
        device_json='phone'
    elif device == 'uhdtv':
        vd='400cm'
        ds='3840x2160'
        device_json='tv'
    for count,exp in enumerate(exp_orig):
        # create a dictionary with 3 keys called I11 I13 I23
        dict = {'I11': {'segments': []}, 'I13': {'segments': []}, 'I23': {'stalling': []}, 'IGen': {'device': device_json, 'displaySize': ds,'viewingDistance': vd}}

        reb = []
        for i in range(1, (1 + nr_c * 13 - 1), 13):
            reb.append(float(exp[i]))
        bit = []
        for i in range(2, (2 + nr_c * 13 - 1), 13):
            bit.append(float(exp[i]))
        height = []
        for i in range(8, (8 + nr_c * 13 - 1), 13):
            height.append(float(exp[i]))
        width = []
        for i in range(7, (7 + nr_c * 13 - 1), 13):
            width.append(float(exp[i]))

        start = 0
        for i in range(nr_chunks):
            seg_feat={"bitrate": bit[i],
            'codec': 'h264',
            "duration": chunk_length,
            "fps": 30.0,
            "resolution": str(int(width[i]))+'x'+str(int(height[i])),
            "start": start}

            dict['I13']['segments'].append(seg_feat)

            start+=chunk_length

        ts=[0, 4, 8, 12 , 16 , 20, 24]
        reb=reb[0:6]
        stallarray=[[ts[i],reb[i]] for i,x in enumerate(reb) if x!=0]
        if stallarray==[]:
            pass
        else:
            for eachstall in stallarray:
                dict['I23']['stalling'].append(eachstall)
        #if count in indices_test:
        with open('features_and_scores_qoes_'+device+'/exp'+str(count)+'.json', 'w') as fp:
            json.dump(dict, fp)
print('max psnr')
print(max_psnr)