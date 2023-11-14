import numpy as np
import random
import os
import pickle
import csv
import json
import shutil
from itu_p1203 import P1203Standalone
os.chdir('C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/Table_IV')
ef=np.load('exp_different_abrs/experiences_with_features.npy')# th+bb+mpc
lstm_exp=np.load('exp_different_abrs/synthetic_experiences.npy') #orderd #th_exp_ordered+bb_exp_ordered+mpc_exp_ordered

shutil.rmtree('lstm0')
os.makedirs('lstm0')
shutil.rmtree('lstm1')
os.makedirs('lstm1')
shutil.rmtree('lstm2')
os.makedirs('lstm2')
shutil.rmtree('p12030')
os.makedirs('p12030')
shutil.rmtree('p12031')
os.makedirs('p12031')
shutil.rmtree('p12032')
os.makedirs('p12032')


np.seed=42
start_indeces=[random.randint(0, 30 - 7) for i in range(102)]
for nr_abr,abr in enumerate(['th','bb','mpc']):
    if nr_abr==0:
        all_abr_feat=ef[0:102]
        lstm_feat=lstm_exp[0:102]
        th_elements=[]
        th_lstm_elements=[]
        for nr,exp in enumerate(all_abr_feat):  # Ensure at least 7 elements are remaining
            start_index=start_indeces[nr]
            th_elements.append(np.concatenate(exp[start_index:start_index + 7]))
        for nr,lstm_single_exp in enumerate(lstm_feat):
            start_index=start_indeces[nr]
            th_lstm_elements.append(np.concatenate(lstm_single_exp[start_index:start_index + 7]))
    elif nr_abr==1:
        all_abr_feat=ef[102:204]
        lstm_feat=lstm_exp[102:204]
        bb_elements=[]
        bb_lstm_elements=[]
        for nr,exp in enumerate(all_abr_feat):
            start_index=start_indeces[nr]
            bb_elements.append(np.concatenate(exp[start_index:start_index + 7]))
        for nr,lstm_single_exp in enumerate(lstm_feat):
            start_index=start_indeces[nr]
            bb_lstm_elements.append(np.concatenate(lstm_single_exp[start_index:start_index + 7]))
    else:
        all_abr_feat=ef[204:306]
        lstm_feat=lstm_exp[204:306]
        mpc_elements=[]
        mpc_lstm_elements=[]
        for nr,exp in enumerate(all_abr_feat):
            start_index=start_indeces[nr]
            mpc_elements.append(np.concatenate(exp[start_index:start_index + 7]))
        for nr,lstm_single_exp in enumerate(lstm_feat):
            start_index = start_indeces[nr]
            mpc_lstm_elements.append(np.concatenate(lstm_single_exp[start_index:start_index + 7]))

#now I have for each abr 102 elements of 7 chunks
#now I want the features of every exp
scores_FTWs = []
scores_bitss = []
scores_logbitss = []
scores_psnrs = []
scoress_ssims = []
scores_vmafs = []
scores_SDNdashs = []
scores_videoAtlass = []
scores_biqpss = []
scores_p1203s = []
nr_c=7
all_ABRs_feat=[th_elements,bb_elements,mpc_elements]
Lstm_all_feat = [th_lstm_elements, bb_lstm_elements, mpc_lstm_elements]
for nr_abr,abr_feat in enumerate(all_ABRs_feat):
    collect_sumbit = []
    collect_sumpsnr = []
    collect_sumssim = []
    collect_sumvmaf = []
    collect_logbit = []
    collect_FTW = []
    collect_SDNdash = []
    collect_videoAtlas = []
    exp_orig = abr_feat
    #min training bitrate
    bit = []
    for exp in exp_orig:
        for i in range(2, (2 + nr_c * 10 - 1), 10):
            bit.append(float(exp[i]))
    min_bit = np.min(bit)

    for exp in exp_orig:
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
            if float(exp[i])>50:
                psnr.append(50)
            else:
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

    print(Lstm_all_feat)
    # Lstm features
    Lstm_feat_seven=Lstm_all_feat[nr_abr]
    print(Lstm_feat_seven)
    print(nr_abr)
    feat_each_video_lstm = []
    for count, exp in enumerate(Lstm_feat_seven):
        save_each_exp = []
        reb = []
        for i in range(0, (0 + nr_c * 5 - 1), 5):
            print(i)
            reb.append(float(exp[i]))
        bit = []
        for i in range(2, (2 + nr_c * 5 - 1), 5):
            bit.append(float(exp[i]))
        res = []
        for i in range(3, (3 + nr_c * 5 - 1), 5):
            res.append(float(exp[i]))
        qps = []
        for i in range(1, (1 + nr_c * 5 - 1), 5):
            qps.append(float(exp[i]))

        for i in range(0, nr_c):
            save_each_exp.append([reb[i], qps[i], bit[i], res[i], 24])
        feat_each_video_lstm.append(save_each_exp)


    # _, _, _, indices_test = train_test_split(exp_orig, range(len(exp_orig)), test_size=0.3, random_state=42)
    for count, exp in enumerate(feat_each_video_lstm):
        with open('lstm'+str(nr_abr)+'/exp' + str(count) + ".csv", "a+", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['SD', 'QP', 'BR', 'RS', 'FR'])
            for i in exp:
                writer.writerow(i)
        with open('lstm'+str(nr_abr)+'/filelist.txt', 'a+') as f:
            # if count in indices_test:
            f.write('lstm'+str(nr_abr)+'/exp' + str(count) + '.csv' + '\n')

    # p1203 features
    nr_chunks = 7
    nr_feat_ec = 10
    chunk_length = 4
    tot_nr_feat = nr_chunks * nr_feat_ec

    for count, exp in enumerate(exp_orig):
        device = 'hdtv'
        if device == 'hdtv':
            vd = '100cm'
            ds = '1920x1080'
            device_json = 'tv'
        elif device == 'phone':
            vd = '20cm'
            device_json = 'phone'
        elif device == 'uhdtv':
            vd = '400cm'
            ds = '3840x2160'
            device_json = 'tv'
        # create a dictionary with 3 keys called I11 I13 I23
        dict = {'I11': {'segments': []}, 'I13': {'segments': []}, 'I23': {'stalling': []},
                'IGen': {'device': device_json, 'displaySize': ds, 'viewingDistance': vd}}

        reb = []
        for i in range(1, (1 + nr_c * 10 - 1), 10):
            reb.append(float(exp[i]))
        bit = []
        for i in range(2, (2 + nr_c * 10 - 1), 10):
            bit.append(float(exp[i]))
        height = []
        for i in range(5, (5 + nr_c * 10 - 1), 10):
            height.append(float(exp[i]))
        width = []
        for i in range(4, (4 + nr_c * 10 - 1), 10):
            width.append(float(exp[i]))

        start = 0
        for i in range(nr_chunks):
            seg_feat = {"bitrate": bit[i],
                        "codec": 'h264',
                        "duration": chunk_length,
                        "fps": 24.0,
                        "resolution": str(int(width[i])) + 'x' + str(int(height[i])),
                        "start": start}

            dict['I13']['segments'].append(seg_feat)

            start += chunk_length

        ts = [0, 4, 8, 12, 16, 20, 24]
        reb = reb[0:6]
        stallarray = [[ts[i], reb[i]] for i, x in enumerate(reb) if x != 0]
        if stallarray == []:
            pass
        else:
            for eachstall in stallarray:
                dict['I23']['stalling'].append(eachstall)
        # if count in indices_test:
        with open('p1203'+str(nr_abr)+'/exp' + str(count) + '.json', 'w') as fp:
            json.dump(dict, fp)



        #############################################

    # for each ABR calculate the scores of each model
    l = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']
    scores_FTW = []
    scores_bits = []
    scores_logbits = []
    scores_psnr = []
    scoress_ssim = []
    scores_vmaf = []
    scores_SDNdash = []
    scores_videoAtlas = []
    scores_biqps = []
    scores_p1203 = []

    for nr_mod, mod in enumerate(l):
        if mod == 'FTW':
            a, b, c, d = np.load('trained_models_allabr/FTW_allabr.npy')
            x1, x2 = np.array(collect_FTW)[:,0], np.array(collect_FTW)[:,1]
            score = a * np.exp(-(b * x1 + c) * x2) + d
            #np.save('predictions_' + ABRs[nr_abr] + '/scores_FTW_' + ABRs[nr_abr], score)
            scores_FTW.append(score)
        elif mod == 'videoAtlas':
            with open('trained_models_allabr/videoAtlas_allabr.pkl', 'rb') as handle:
                pickled_atlas = pickle.load(handle)
            videoAtlasregressor = pickled_atlas
            score = videoAtlasregressor.predict(np.array(collect_videoAtlas))
            #np.save('predictions_' + ABRs[nr_abr] + '/scores_videoAtlas_' + ABRs[nr_abr], score)
            scores_videoAtlas.append(score)
        else:
            params = np.load('trained_models_allabr/' + mod + '_allabr.npy')

            #np.save('predictions_' + ABRs[nr_abr] + '/scores_' + mod + '_' + ABRs[nr_abr], score)
            if mod == 'bit':
                score = np.dot(params, np.array(collect_sumbit).T)
                scores_bits.append(score)
            elif mod == 'logbit':
                score = np.dot(params, np.array(collect_logbit).T)
                scores_logbits.append(score)
            elif mod == 'psnr':
                score = np.dot(params, np.array(collect_sumpsnr).T)
                scores_psnr.append(score)
            elif mod == 'ssim':
                score = np.dot(params, np.array(collect_sumssim).T)
                scoress_ssim.append(score)
            elif mod == 'vmaf':
                score = np.dot(params, np.array(collect_sumvmaf).T)
                scores_vmaf.append(score)
            elif mod == 'SDNdash':
                score = np.dot(params, np.array(collect_SDNdash).T)
                scores_SDNdash.append(score)
    scores_FTWs.append(scores_FTW)
    scores_bitss.append(scores_bits)
    scores_logbitss.append(scores_logbits)
    scores_psnrs.append(scores_psnr)
    scoress_ssims.append(scoress_ssim)
    scores_vmafs.append(scores_vmaf)
    scores_SDNdashs.append(scores_SDNdash)
    scores_videoAtlass.append(scores_videoAtlas)

    # LSTM
    # calculate the biLSTM scores for each experience
    filetxt = 'lstm'+str(nr_abr)+'/filelist.txt'
    os.system('biQPS ' + filetxt)
    scoresbiqps = []
    with open('output.txt') as f:
        for line in f.readlines()[1:]:
            onefive = float(line.split('\t')[-1])
            X_std = (onefive - 1) / (5 - 1)
            X_scaled = X_std * (100 - 1) + 1
            scoresbiqps.append(X_scaled)
    #np.save('predictions_' + ABRs[nr_abr] + '/scores_biqps_' + ABRs[nr_abr], scoresbiqps)
    scores_biqps.append(scoresbiqps)
    scores_biqpss.append(scoresbiqps)
    #os.remove('output.txt')
    #delete filelist.txt


    # P1203
    scoresp1203 = []
    for conta in range(102):
        f = open('p1203'+str(nr_abr)+'/exp' + str(conta) + '.json')
        data = json.load(f)
        p1203_results = P1203Standalone(data).calculate_complete()
        onefive = p1203_results['O46']
        print(onefive)
        X_std = (onefive - 1) / (5 - 1)
        X_scaled = X_std * (100 - 1) + 1
        scoresp1203.append(X_scaled)

    #np.save('predictions_' + ABRs[nr_abr] + '/scores_p1203_' + ABRs[nr_abr], scoresp1203)
    scores_p1203s.append(scoresp1203)



#################################print
#realign bitrates because too much sensible to ifs
scores_bitss_rescaled=[]
for sco in scores_bitss:
    ss=[]
    for nr_s,s in enumerate(sco[0]):
        X_scaled = ((s - min(sco[0])) / (max(sco[0]) - min(sco[0]))) * (100 - 1) + 1
        ss.append(X_scaled)
    scores_bitss_rescaled.append(ss)
import pandas as pd
df = pd.DataFrame(columns=['ABR', 'Mean_FTW', 'Mean_bits', 'Mean_logbits', 'Mean_psnr', 'Mean_ssim', 'Mean_vmaf', 'Mean_SDNdash', 'Mean_videoAtlas', 'Mean_scoresbiqps'])
ABRs=['th','bb','mpc']
for nr_abr,abr in enumerate(ABRs):
    data = {
        'ABR': ABRs[nr_abr],
        'Mean_FTW': np.mean(scores_FTWs[nr_abr]),
        'Mean_bits': np.mean(scores_bitss_rescaled[nr_abr]),
        'Mean_logbits': np.mean(scores_logbitss[nr_abr]),
        'Mean_psnr': np.mean(scores_psnrs[nr_abr]),
        'Mean_ssim': np.mean(scoress_ssims[nr_abr]),
        'Mean_vmaf': np.mean(scores_vmafs[nr_abr]),
        'Mean_SDNdash': np.mean(scores_SDNdashs[nr_abr]),
        'Mean_videoAtlas': np.mean(scores_videoAtlass[nr_abr]),
        'Mean_scoresbiqps': np.mean(scores_biqpss[nr_abr]),
        'Mean_scoresp1203': np.mean(scores_p1203s[nr_abr])
    }

    # Append the data to the DataFrame
    df = df.append(data, ignore_index=True)
#
# Save the final DataFrame to a CSV file
df.to_csv('qoes_new.csv', index=False)

#print every row of df
for index, row in df.iterrows():
    print(
        str(row['ABR']) + ' & ',
        str(round(row['Mean_FTW'], 2)) + ' & ',
        str(round(row['Mean_bits'], 2)) + ' & ',
        str(round(row['Mean_logbits'], 2)) + ' & ',
        str(round(row['Mean_psnr'], 2)) + ' & ',
        str(round(row['Mean_ssim'], 2)) + ' & ',
        str(round(row['Mean_vmaf'], 2)) + ' & ',
        str(round(row['Mean_SDNdash'], 2)) + ' & ',
        str(round(row['Mean_videoAtlas'], 2)) + ' & ',
        str(round(row['Mean_scoresbiqps'], 2)) + ' & ',
        str(round(row['Mean_scoresp1203'], 2)) + ' \\\\'
    )
    print('hline')

#copy df without first column
df_copy = df.iloc[:, 1:]

# Find the highest and second-highest values for each column
highest_values = df_copy.apply(lambda x: x.nlargest(2).tolist())
pi=[]
for column in highest_values:
    v2=highest_values[column][0]
    v1=highest_values[column][1]
    percentage_increase = ((v2 - v1) / v1) * 100
    pi.append(percentage_increase)
    s=''
for i in pi:
    s=s+str(round(i,2))+' & '
print('Improvement % & '+s)



