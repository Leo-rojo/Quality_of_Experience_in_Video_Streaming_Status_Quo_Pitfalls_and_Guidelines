import numpy as np
import pickle
import os
import json
from itu_p1203 import P1203Standalone
os.chdir('C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/QoE_selector')
for device in ['hdtv']:
    #create folder prediction all data
    if not os.path.exists('predictions_all_data_'+device):
        os.makedirs('predictions_all_data_'+device)
    collect_sumbit=np.load('features_and_scores_qoes_'+device+'/feat_bit.npy')
    collect_sumpsnr=np.load('features_and_scores_qoes_'+device+'/feat_psnr.npy')
    collect_sumssim=np.load('features_and_scores_qoes_'+device+'/feat_ssim.npy')
    collect_sumvmaf=np.load('features_and_scores_qoes_'+device+'/feat_vmaf.npy')
    collect_logbit=np.load('features_and_scores_qoes_'+device+'/feat_logbit.npy')
    collect_FTW=np.load('features_and_scores_qoes_'+device+'/feat_ftw.npy')
    collect_SDNdash=np.load('features_and_scores_qoes_'+device+'/feat_sdn.npy')
    collect_videoAtlas=np.load('features_and_scores_qoes_'+device+'/feat_va.npy')
    users_scores = np.load('allfeat_allscores/users_scores_' + device + '.npy', allow_pickle=True)
    mos_hdtv = np.mean(users_scores, axis=0)

    collect_all_features = [collect_sumbit, collect_logbit, collect_sumpsnr, collect_sumssim, collect_sumvmaf,
                            np.array(collect_FTW), collect_SDNdash, collect_videoAtlas]
    l = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']

    collect_all = []
    temp_score_FTW = []
    temp_score_bits = []
    temp_score_logbits = []
    temp_score_psnr = []
    temp_score_ssim = []
    temp_score_vmaf = []
    temp_score_SDNdash = []
    temp_score_videoAtlas = []
    for idx_i, i in enumerate(l):
        collect_temp = []
        all_features = collect_all_features[idx_i]

        if i == 'FTW':
            a, b, c, d = np.load('trained_models_' + device + '/FTW_' + device + '.npy')
            x1, x2 = all_features[:, 0], all_features[:, 1]
            score = a * np.exp(-(b * x1 + c) * x2) + d
            temp_score_FTW.append(score)
        elif i == 'videoAtlas':
            with open('trained_models_' + device + '/videoAtlas_mos' + device + '.pkl', 'rb') as handle:
                pickled_atlas = pickle.load(handle)
            videoAtlasregressor = pickled_atlas  # 0 there is the mdoel,
            temp_score_videoAtlas = videoAtlasregressor.predict(all_features)
        else:
            for single_X_test_mos in all_features:
                params = np.load('trained_models_' + device + '/' + i + '_' + device + '.npy')
                score = np.dot(params, single_X_test_mos)
                if i == 'bit':
                    temp_score_bits.append(score)
                elif i == 'logbit':
                    temp_score_logbits.append(score)
                elif i == 'psnr':
                    temp_score_psnr.append(score)
                elif i == 'ssim':
                    temp_score_ssim.append(score)
                elif i == 'vmaf':
                    temp_score_vmaf.append(score)
                elif i == 'SDNdash':
                    temp_score_SDNdash.append(score)
    # save all the scores
    np.save('predictions_all_data_' + device + '/scores_FTW_' + device, temp_score_FTW)
    np.save('predictions_all_data_' + device + '/scores_bits_' + device, temp_score_bits)
    np.save('predictions_all_data_' + device + '/scores_logbits_' + device, temp_score_logbits)
    np.save('predictions_all_data_' + device + '/scores_psnr_' + device, temp_score_psnr)
    np.save('predictions_all_data_' + device + '/scores_ssim_' + device, temp_score_ssim)
    np.save('predictions_all_data_' + device + '/scores_vmaf_' + device, temp_score_vmaf)
    np.save('predictions_all_data_' + device + '/scores_SDNdash_' + device, temp_score_SDNdash)
    np.save('predictions_all_data_' + device + '/scores_videoAtlas_' + device, temp_score_videoAtlas)
    print('-----------train_test_classic_models_done-----------')

    # calculate the biLSTM scores for each experience
    filetxt = 'features_and_scores_qoes_' + device + '/filelist.txt'
    os.system('biQPS ' + filetxt)
    scoresbiqps = []
    with open('output.txt') as f:
        for line in f.readlines()[1:]:
            onefive = float(line.split('\t')[-1])
            X_std = (onefive - 1) / (5 - 1)
            X_scaled = X_std * (100 - 1) + 1
            scoresbiqps.append(X_scaled)
    np.save('predictions_all_data_' + device + '/scores_biqps_' + device, scoresbiqps)

    # P1203
    scoresp1203 = []
    for conta in range(len(mos_hdtv)):
        f = open('features_and_scores_qoes_' + device + '/exp' + str(conta) + '.json')
        data = json.load(f)
        p1203_results = P1203Standalone(data).calculate_complete()
        onefive = p1203_results['O46']
        print(onefive)
        X_std = (onefive - 1) / (5 - 1)
        X_scaled = X_std * (100 - 1) + 1
        scoresp1203.append(X_scaled)

    np.save('predictions_all_data_' + device + '/scores_p1203_' + device, scoresp1203)


#load all the scores from predictions_all_data_+device and put in df
import pandas as pd
l = ['bits', 'logbits', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas','biqps', 'p1203']
scores_array=[]
for device in ['hdtv']:
    for i in l:
        if i=='FTW':
            scores_array.append(np.load('predictions_all_data_' + device + '/scores_' + i + '_' + device + '.npy')[0])
        else:
            scores_array.append(np.load('predictions_all_data_' + device + '/scores_' + i + '_' + device + '.npy'))
    scores_array.append(mos_hdtv)
#put scores_array in df
df=pd.DataFrame(np.array(scores_array).T,columns=l+['mos'])

#best model
#for each row find the model with the lowest error compared to mos

#find the best model for each row
best_model=[]
for idx,row in df.iterrows():
    best_model.append(l[np.argmin(np.abs(row[l].values-row['mos']))])
df['best_model']=best_model
#find the best model between videoatlas and vmaf for each row
best_model=[]
diffs_va_vmaf=[]
for idx,row in df.iterrows():
    # Calculate absolute differences between 'videoatlas' and 'vmaf'
    differencesva = np.abs(row['videoAtlas'] - row['mos'])
    differencevmaf = np.abs(row['vmaf'] - row['mos'])
    if differencesva<differencevmaf:
        diff=differencevmaf-differencesva
        best_model.append('videoAtlas')
    else:
        diff=differencesva-differencevmaf
        best_model.append('vmaf')
    diffs_va_vmaf.append(diff)
    # Add the 'best_model' list as a new column in the DataFrame
df['va_vmaf'] = best_model
df['diffs_va_vmaf'] = diffs_va_vmaf

#finde the second best model for each row
# Initialize a list to store the second-best model for each row
# second_best_model = []
# Define the number of top models you want (in this case, 2 for second best)
# num_top_models = 2
# for idx, row in df.iterrows():
#     # Calculate absolute differences between the values in 'l' columns and 'mos'
#     abs_diff = np.abs(row[l].values - row['mos'])
#     # Sort the absolute differences and get the index of the second smallest difference
#     second_best_model_idx = np.argpartition(abs_diff, num_top_models - 1)[num_top_models - 1]
#     # Append the second best model to the list
#     second_best_model.append(l[second_best_model_idx])
# # Add the 'second_best_model' list as a new column in the DataFrame
# df['second_best_model'] = second_best_model

#save column best model as npy
np.save('bestmodels_' + device, df['best_model'].values)
np.save('bestmodels_va_vmaf_' + device, df['va_vmaf'].values)
