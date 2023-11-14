import json
from itu_p1203 import P1203Standalone
import numpy as np
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
os.chdir('C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/QoE_selector')

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

for device in ['hdtv','phone','uhdtv']:
#create folder trained_models_device
    if not os.path.exists('trained_models_'+device):
        os.makedirs('trained_models_'+device)
    #create folder predictions+device
    if not os.path.exists('predictions_'+device):
        os.makedirs('predictions_'+device)
    collect_sumbit=np.load('features_and_scores_qoes_'+device+'/feat_bit.npy')
    collect_sumpsnr=np.load('features_and_scores_qoes_'+device+'/feat_psnr.npy')
    collect_sumssim=np.load('features_and_scores_qoes_'+device+'/feat_ssim.npy')
    collect_sumvmaf=np.load('features_and_scores_qoes_'+device+'/feat_vmaf.npy')
    collect_logbit=np.load('features_and_scores_qoes_'+device+'/feat_logbit.npy')
    collect_FTW=np.load('features_and_scores_qoes_'+device+'/feat_ftw.npy')
    #collect_SDNdash=np.load('features_and_scores_qoes_'+device+'/feat_sdn.npy')
    collect_videoAtlas=np.load('features_and_scores_qoes_'+device+'/feat_va.npy')
    all_features_compleate=np.load('allfeat_allscores/all_feat_'+device+'.npy',allow_pickle=True)
    all_features_compleate=all_features_compleate[:,:-1]
    #remove column 3,16,29,42,55,68,81
    all_features_compleate=np.delete(all_features_compleate, [3,16,29,42,55,68,81], axis=1)
    users_scores=np.load('allfeat_allscores/users_scores_'+device+'.npy',allow_pickle=True)
    mos_hdtv=np.mean(users_scores,axis=0)

    ############################################################calculate personalized parameters#########################################################
    collect_all_features=[collect_sumbit,collect_logbit,collect_sumpsnr,collect_sumssim,collect_sumvmaf,np.array(collect_FTW),collect_videoAtlas]#collect_SDNdash,
    l = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW',  'videoAtlas']#'SDNdash',
    df_mae = pd.DataFrame(columns=['rs', l[0], l[1], l[2], l[3], l[4], l[5], l[6], 'biqps', 'p1203'])
    df_rmse = pd.DataFrame(columns=['rs', l[0], l[1], l[2], l[3], l[4], l[5], l[6], 'biqps', 'p1203'])
    for rs in range(0,99):
        collect_all = []
        temp_score_FTW = []
        temp_score_bits = []
        temp_score_logbits = []
        temp_score_psnr = []
        temp_score_ssim = []
        temp_score_vmaf = []
        #temp_score_SDNdash = []
        temp_score_videoAtlas = []
        for idx_i,i in enumerate(l):
            collect_temp = []
            all_features = collect_all_features[idx_i]
            X_train, X_test, indices_train, indices_test = train_test_split(all_features, range(len(all_features)), test_size=0.3, random_state=rs)
            X_train_mos, X_test_mos, y_train_mos, y_test_mos = train_test_split(all_features, mos_hdtv, test_size=0.3, random_state=rs)
            X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(all_features_compleate, mos_hdtv, test_size=0.3, random_state=rs)

            if i == 'FTW':
                a, b, c, d=fit_nonlinear((X_train_mos[:, 0], X_train_mos[:, 1]), y_train_mos)
                np.save('trained_models_'+device+'/FTW_'+device, [a, b, c, d])
                x1, x2 = X_test_mos[:,0], X_test_mos[:, 1]
                score = a * np.exp(-(b * x1 + c) * x2) + d
                temp_score_FTW.append(score)
            elif i == 'videoAtlas':
                pickle.dump(fit_supreg(X_train_mos, y_train_mos), open( 'trained_models_'+device+'/videoAtlas_mos'+device+'.pkl', 'wb'))
                with open('trained_models_'+device+'/videoAtlas_mos'+device+'.pkl', 'rb') as handle:
                    pickled_atlas = pickle.load(handle)
                videoAtlasregressor = pickled_atlas  # 0 there is the mdoel,
                temp_score_videoAtlas = videoAtlasregressor.predict(X_test_mos)
                temp_score_videoAtlas_for_classifier = videoAtlasregressor.predict(all_features)
            else:
                params = fit_linear(X_train_mos, y_train_mos)
                for single_X_test_mos in X_test_mos:
                    score = np.dot(params, single_X_test_mos)
                    if i=='bit':
                        np.save('trained_models_'+device+'/bit_'+device, params)
                        temp_score_bits.append(score)
                    elif i=='logbit':
                        np.save('trained_models_'+device+'/logbit_'+device, params)
                        temp_score_logbits.append(score)
                    elif i=='psnr':
                        np.save('trained_models_'+device+'/psnr_'+device, params)
                        temp_score_psnr.append(score)
                    elif i=='ssim':
                        np.save('trained_models_'+device+'/ssim_'+device, params)
                        temp_score_ssim.append(score)
                    elif i=='vmaf':
                        vmaf_model_trained=params
                        np.save('trained_models_'+device+'/vmaf_'+device, params)
                        temp_score_vmaf.append(score)
                    # elif i=='SDNdash':
                    #     np.save('trained_models_'+device+'/SDNdash_'+device, params)
                    #     temp_score_SDNdash.append(score)
                if i=='vmaf':
                    temp_score_vmaf_for_classifier = np.dot(vmaf_model_trained, all_features.T)
        #average temp_score_videoAtlas and temp_score_vmaf
        merge_va_vmaf_scores=[]
        for i in range(len(temp_score_videoAtlas)):
            merge_va_vmaf_scores.append((temp_score_videoAtlas[i]+temp_score_vmaf[i])/2)




        #save all the scores
        # np.save('predictions_'+device+'/scores_FTW_'+device, temp_score_FTW)
        # np.save('predictions_'+device+'/scores_bits_'+device, temp_score_bits)
        # np.save('predictions_'+device+'/scores_logbits_'+device, temp_score_logbits)
        # np.save('predictions_'+device+'/scores_psnr_'+device, temp_score_psnr)
        # np.save('predictions_'+device+'/scores_ssim_'+device, temp_score_ssim)
        # np.save('predictions_'+device+'/scores_vmaf_'+device, temp_score_vmaf)
        # np.save('predictions_'+device+'/scores_SDNdash_'+device, temp_score_SDNdash)
        # np.save('predictions_'+device+'/scores_videoAtlas_'+device, temp_score_videoAtlas)
        #calculate mae and rmse for each model
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score
        import pandas as pd
        #calculate mae and rmse for each model
        mae_bit=mean_absolute_error(y_test_mos, temp_score_bits)
        mae_logbit=mean_absolute_error(y_test_mos, temp_score_logbits)
        mae_psnr=mean_absolute_error(y_test_mos, temp_score_psnr)
        mae_ssim=mean_absolute_error(y_test_mos, temp_score_ssim)
        mae_vmaf=mean_absolute_error(y_test_mos, temp_score_vmaf)
        mae_ftw=mean_absolute_error(y_test_mos, temp_score_FTW[0])
        #mae_sdn=mean_absolute_error(y_test_mos, temp_score_SDNdash)
        mae_va=mean_absolute_error(y_test_mos, temp_score_videoAtlas)
        mae_merge_va_vmaf=mean_absolute_error(y_test_mos, merge_va_vmaf_scores)
        rmse_bit = sqrt(mean_squared_error(y_test_mos, temp_score_bits))
        rmse_logbit = sqrt(mean_squared_error(y_test_mos, temp_score_logbits))
        rmse_psnr = sqrt(mean_squared_error(y_test_mos, temp_score_psnr))
        rmse_ssim = sqrt(mean_squared_error(y_test_mos, temp_score_ssim))
        rmse_vmaf = sqrt(mean_squared_error(y_test_mos, temp_score_vmaf))
        rmse_ftw = sqrt(mean_squared_error(y_test_mos, temp_score_FTW[0]))
        #rmse_sdn = sqrt(mean_squared_error(y_test_mos, temp_score_SDNdash))
        rmse_va = sqrt(mean_squared_error(y_test_mos, temp_score_videoAtlas))
        rmse_merge_va_vmaf = sqrt(mean_squared_error(y_test_mos, merge_va_vmaf_scores))
        print('-----------train_test_classic_models_done-----------')
        best_model_va_vmaf=[]

        # create ground truth for training
        for i in range(len(all_features)):
            differencesva = np.abs(temp_score_videoAtlas_for_classifier[i] - mos_hdtv[i])
            differencevmaf = np.abs(temp_score_vmaf_for_classifier[i] - mos_hdtv[i])
            if differencesva < differencevmaf:
                diff = differencevmaf - differencesva
                best_model_va_vmaf.append('videoAtlas')
            else:
                diff = differencesva - differencevmaf
                best_model_va_vmaf.append('vmaf')
        #diffs_va_vmaf.append(diff)
        # train and calculate accuracy in test
        lb = LabelBinarizer()
        lb.fit(best_model_va_vmaf)
        one_hot_encoded = lb.transform(best_model_va_vmaf)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        scaler.fit(X_train_class)
        X_train_sca = scaler.transform(X_train_class)
        clf.fit(X_train_sca, one_hot_encoded[indices_train])
        X_test_sca = scaler.transform(X_test_class)
        y_pred = clf.predict(X_test_sca)
        y_test_lab = lb.inverse_transform(one_hot_encoded[indices_test])
        y_pred_lab = lb.inverse_transform(y_pred)
        print(confusion_matrix(y_test_lab, y_pred_lab))
        print(accuracy_score(y_test_lab, y_pred_lab))
        # calculate mae and rmse for the classifier plus vmaf and videoAtlas
        temp_score_classifier_model = []
        for i in range(len(y_test_lab)):
            if y_pred_lab[i] == 'videoAtlas':
                temp_score_classifier_model.append(temp_score_videoAtlas[i])
            else:
                temp_score_classifier_model.append(temp_score_vmaf[i])
        mae_classifier = mean_absolute_error(y_test_mos, temp_score_classifier_model)
        rmse_classifier = sqrt(mean_squared_error(y_test_mos, temp_score_classifier_model))
        print('-----------classifier_trained-----------')

        #####################test##############################
        #LSTM
        #calculate the biLSTM scores for each experience
        filetxt='features_and_scores_qoes_'+device+'/filelist.txt'
        filetxtest='features_and_scores_qoes_'+device+'/filelist_test.txt'
        #read each line of the file and if the number is present in indices_test then write in filelist_test.txt
        with open(filetxt) as f:
            lines = f.readlines()
        with open(filetxtest, 'w') as f:
            for line in lines:
                if int(line.split('exp')[1].split('.')[0]) in indices_test:
                    f.write(line)
        os.system('biQPS '+filetxtest)
        scoresbiqps=[]
        with open('output.txt') as f:
            for line in f.readlines()[1:]:
                onefive=float(line.split('\t')[-1])
                X_std = (onefive - 1) / (5 - 1)
                X_scaled = X_std * (100 - 1) + 1
                scoresbiqps.append(X_scaled)
        #np.save('predictions_'+device+'/scores_biqps_'+device,scoresbiqps)
        #calculate mae and rmse
        mae_biqps=mean_absolute_error(y_test_mos, scoresbiqps)
        rmse_biqps = sqrt(mean_squared_error(y_test_mos, scoresbiqps))

        #P1203
        scoresp1203=[]
        for conta in range(len(mos_hdtv)):
            if conta in indices_test:
                f = open('features_and_scores_qoes_'+device+'/exp'+str(conta)+'.json')
                data = json.load(f)
                p1203_results = P1203Standalone(data).calculate_complete()
                onefive = p1203_results['O46']
                print(onefive)
                X_std = (onefive - 1) / (5 - 1)
                X_scaled = X_std * (100 - 1) + 1
                scoresp1203.append(X_scaled)

        #np.save('predictions_'+device+'/scores_p1203_'+device, scoresp1203)
        #calculate mae and rmse
        mae_p1203=mean_absolute_error(y_test_mos, scoresp1203)
        rmse_p1203 = sqrt(mean_squared_error(y_test_mos, scoresp1203))

        #add to df with rs the idx and models the columns

        df_mae = df_mae.append({'rs': rs, 'bit': mae_bit, 'logbit': mae_logbit, 'psnr': mae_psnr, 'ssim': mae_ssim, 'vmaf': mae_vmaf,
                        'FTW': mae_ftw, 'videoAtlas': mae_va, 'biqps': mae_biqps, 'p1203': mae_p1203, 'merge_va_vmaf':mae_merge_va_vmaf, 'classifier_model':mae_classifier}, ignore_index=True)#'SDNdash': mae_sdn,
        df_rmse = df_rmse.append({'rs': rs, 'bit': rmse_bit, 'logbit': rmse_logbit, 'psnr': rmse_psnr, 'ssim': rmse_ssim, 'vmaf': rmse_vmaf,
                        'FTW': rmse_ftw,  'videoAtlas': rmse_va, 'biqps': rmse_biqps, 'p1203': rmse_p1203, 'merge_va_vmaf':rmse_merge_va_vmaf, 'classifier_model': rmse_classifier}, ignore_index=True)#'SDNdash': rmse_sdn,

    #add row with average of all rs
    df_mae = df_mae.append({'rs': 'average', 'bit': df_mae['bit'].mean(), 'logbit': df_mae['logbit'].mean(), 'psnr': df_mae['psnr'].mean(), 'ssim': df_mae['ssim'].mean(), 'vmaf': df_mae['vmaf'].mean(),
                    'FTW': df_mae['FTW'].mean(),  'videoAtlas': df_mae['videoAtlas'].mean(), 'biqps': df_mae['biqps'].mean(),#'SDNdash': df_mae['SDNdash'].mean(),
                    'p1203': df_mae['p1203'].mean(), 'merge_va_vmaf':df_mae['merge_va_vmaf'].mean(), 'classifier_model':df_mae['classifier_model'].mean()}, ignore_index=True)
    df_rmse = df_rmse.append({'rs': 'average', 'bit': df_rmse['bit'].mean(), 'logbit': df_rmse['logbit'].mean(), 'psnr': df_rmse['psnr'].mean(), 'ssim': df_rmse['ssim'].mean(), 'vmaf': df_rmse['vmaf'].mean(),
                    'FTW': df_rmse['FTW'].mean(),  'videoAtlas': df_rmse['videoAtlas'].mean(), 'biqps': df_rmse['biqps'].mean(), #'SDNdash': df_rmse['SDNdash'].mean(),
                    'p1203': df_rmse['p1203'].mean(), 'merge_va_vmaf':df_rmse['merge_va_vmaf'].mean(),'classifier_model': df_rmse['classifier_model'].mean()}, ignore_index=True)
    #for each row highlight smaller value
    #save df in excel
    df_mae.to_excel('mae_selector'+device+'.xlsx', sheet_name='sheet1', index=False)
    df_rmse.to_excel('rmse_selector'+device+'.xlsx', sheet_name='sheet1', index=False)


