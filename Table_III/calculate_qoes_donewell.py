import json
from itu_p1203 import P1203Standalone
import numpy as np
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import os
from sklearn.model_selection import train_test_split
import pandas as pd
#remove warnings
import warnings
warnings.filterwarnings("ignore")
os.chdir('C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/Table_III')

#if qoes.txt exists, delete it
if os.path.exists('qoes.txt'):
    os.remove('qoes.txt')
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
rs=42
ABRs=['Pensieve','FastMPC','BufferOccupancy','RateBased','Rdos']
df = pd.DataFrame(columns=['ABR', 'Mean_FTW', 'Mean_bits', 'Mean_logbits', 'Mean_psnr', 'Mean_ssim', 'Mean_vmaf', 'Mean_SDNdash', 'Mean_videoAtlas', 'Mean_scoresbiqps'])
Save_models_abr=[]

#select training batch from every abrs and then train the models
x_train_allabrs_bit=[]
x_train_allabrs_logbit=[]
x_train_allabrs_psnr=[]
x_train_allabrs_ssim=[]
x_train_allabrs_vmaf=[]
x_train_allabrs_FTW=[]
x_train_allabrs_SDNdash=[]
x_train_allabrs_videoAtlas=[]
y_train_allabrs_bit=[]
y_train_allabrs_logbit=[]
y_train_allabrs_psnr=[]
y_train_allabrs_ssim=[]
y_train_allabrs_vmaf=[]
y_train_allabrs_FTW=[]
y_train_allabrs_SDNdash=[]
y_train_allabrs_videoAtlas=[]
X_test_for_each_abr=[]
for nr_abr,abr in enumerate(ABRs):
    save_models=[]
    #create folder trained_models_device
    # if not os.path.exists('trained_models_'+ABRs[nr_abr]):
    #     os.makedirs('trained_models_'+ABRs[nr_abr])
    #create folder predictions+device
    if not os.path.exists('predictions_'+ABRs[nr_abr]):
        os.makedirs('predictions_'+ABRs[nr_abr])
    collect_sumbit=np.load('features_and_scores_qoes_'+ABRs[nr_abr]+'/feat_bit.npy')
    collect_sumpsnr=np.load('features_and_scores_qoes_'+ABRs[nr_abr]+'/feat_psnr.npy')
    collect_sumssim=np.load('features_and_scores_qoes_'+ABRs[nr_abr]+'/feat_ssim.npy')
    collect_sumvmaf=np.load('features_and_scores_qoes_'+ABRs[nr_abr]+'/feat_vmaf.npy')
    collect_logbit=np.load('features_and_scores_qoes_'+ABRs[nr_abr]+'/feat_logbit.npy')
    collect_FTW=np.load('features_and_scores_qoes_'+ABRs[nr_abr]+'/feat_ftw.npy')
    collect_SDNdash=np.load('features_and_scores_qoes_'+ABRs[nr_abr]+'/feat_sdn.npy')
    collect_videoAtlas=np.load('features_and_scores_qoes_'+ABRs[nr_abr]+'/feat_va.npy')
    mos_abr=np.load('allfeat_allscores_abr/mos_'+ABRs[nr_abr]+'.npy',allow_pickle=True)

    ############################################################calculate personalized parameters#########################################################
    collect_all_features=[collect_sumbit,collect_logbit,collect_sumpsnr,collect_sumssim,collect_sumvmaf,np.array(collect_FTW),collect_SDNdash,collect_videoAtlas]
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
    X_test_for_each_mod=[]
    for idx_i,i in enumerate(l):
        collect_temp = []
        all_features = collect_all_features[idx_i]
        X_train, X_test, indices_train, indices_test = train_test_split(all_features, range(len(all_features)), test_size=0.3, random_state=rs)
        X_train_mos, X_test_mos, y_train_mos, y_test_mos = train_test_split(all_features, mos_abr, test_size=0.3, random_state=rs)
        X_test_for_each_mod.append(X_test_mos)
        if i == 'FTW':
            x_train_allabrs_FTW=x_train_allabrs_FTW+[X_train_mos]
            y_train_allabrs_FTW=y_train_allabrs_FTW+ [y_train_mos]
            # np.save('trained_models_'+ABRs[nr_abr]+'/FTW_'+ABRs[nr_abr], [a, b, c, d])
            # x1, x2 = X_test_mos[:,0], X_test_mos[:, 1]
            # score = a * np.exp(-(b * x1 + c) * x2) + d
            # temp_score_FTW.append(score)
        elif i == 'videoAtlas':
            x_train_allabrs_videoAtlas=x_train_allabrs_videoAtlas+[X_train_mos]
            y_train_allabrs_videoAtlas=y_train_allabrs_videoAtlas+ [y_train_mos]
            # pickle.dump(fit_supreg(X_train_mos, y_train_mos), open( 'trained_models_'+ABRs[nr_abr]+'/videoAtlas_mos'+ABRs[nr_abr]+'.pkl', 'wb'))
            # with open('trained_models_'+ABRs[nr_abr]+'/videoAtlas_mos'+ABRs[nr_abr]+'.pkl', 'rb') as handle:
            #     pickled_atlas = pickle.load(handle)
            # videoAtlasregressor = pickled_atlas  # 0 there is the mdoel,
            # temp_score_videoAtlas = videoAtlasregressor.predict(X_test_mos)

            #params = fit_linear(X_train_mos, y_train_mos)
            # for single_X_test_mos in X_test_mos:
            #     score = np.dot(params, single_X_test_mos)
        elif i=='bit':
            x_train_allabrs_bit=x_train_allabrs_bit+[X_train_mos]
            y_train_allabrs_bit=y_train_allabrs_bit+ [y_train_mos]
            # np.save('trained_models_'+ABRs[nr_abr]+'/bit_'+ABRs[nr_abr], params)
            # temp_score_bits.append(score)
        elif i=='logbit':
            x_train_allabrs_logbit=x_train_allabrs_logbit+[X_train_mos]
            y_train_allabrs_logbit=y_train_allabrs_logbit+ [y_train_mos]
            # np.save('trained_models_'+ABRs[nr_abr]+'/logbit_'+ABRs[nr_abr], params)
            # temp_score_logbits.append(score)
        elif i=='psnr':
            x_train_allabrs_psnr=x_train_allabrs_psnr+[X_train_mos]
            y_train_allabrs_psnr=y_train_allabrs_psnr+ [y_train_mos]
            # np.save('trained_models_'+ABRs[nr_abr]+'/psnr_'+ABRs[nr_abr], params)
            # temp_score_psnr.append(score)
        elif i=='ssim':
            x_train_allabrs_ssim=x_train_allabrs_ssim+[X_train_mos]
            y_train_allabrs_ssim=y_train_allabrs_ssim+ [y_train_mos]
            # np.save('trained_models_'+ABRs[nr_abr]+'/ssim_'+ABRs[nr_abr], params)
            # temp_score_ssim.append(score)
        elif i=='vmaf':
            x_train_allabrs_vmaf=x_train_allabrs_vmaf+[X_train_mos]
            y_train_allabrs_vmaf=y_train_allabrs_vmaf+ [y_train_mos]
            # np.save('trained_models_'+ABRs[nr_abr]+'/vmaf_'+ABRs[nr_abr], params)
            # temp_score_vmaf.append(score)
        elif i=='SDNdash':
            x_train_allabrs_SDNdash=x_train_allabrs_SDNdash+[X_train_mos]
            y_train_allabrs_SDNdash=y_train_allabrs_SDNdash+ [y_train_mos]
            # np.save('trained_models_'+ABRs[nr_abr]+'/SDNdash_'+ABRs[nr_abr], params)
            # temp_score_SDNdash.append(score)
    X_test_for_each_abr.append(X_test_for_each_mod)
#merge all the training batches and y
x_train_allabrs_bit=np.concatenate(x_train_allabrs_bit)
x_train_allabrs_logbit=np.concatenate(x_train_allabrs_logbit)
x_train_allabrs_psnr=np.concatenate(x_train_allabrs_psnr)
x_train_allabrs_ssim=np.concatenate(x_train_allabrs_ssim)
x_train_allabrs_vmaf=np.concatenate(x_train_allabrs_vmaf)
x_train_allabrs_FTW=np.concatenate(x_train_allabrs_FTW)
x_train_allabrs_SDNdash=np.concatenate(x_train_allabrs_SDNdash)
x_train_allabrs_videoAtlas=np.concatenate(x_train_allabrs_videoAtlas)
y_train_allabrs_bit=np.concatenate(y_train_allabrs_bit)
y_train_allabrs_logbit=np.concatenate(y_train_allabrs_logbit)
y_train_allabrs_psnr=np.concatenate(y_train_allabrs_psnr)
y_train_allabrs_ssim=np.concatenate(y_train_allabrs_ssim)
y_train_allabrs_vmaf=np.concatenate(y_train_allabrs_vmaf)
y_train_allabrs_FTW=np.concatenate(y_train_allabrs_FTW)
y_train_allabrs_SDNdash=np.concatenate(y_train_allabrs_SDNdash)
y_train_allabrs_videoAtlas=np.concatenate(y_train_allabrs_videoAtlas)

#create if not exists trained_models_allabr
if not os.path.exists('trained_models_allabr'):
    os.makedirs('trained_models_allabr')
#train the models
#bit
params = fit_linear(x_train_allabrs_bit, y_train_allabrs_bit)
np.save('trained_models_allabr/bit_allabr', params)

#logbit
params = fit_linear(x_train_allabrs_logbit, y_train_allabrs_logbit)
np.save('trained_models_allabr/logbit_allabr', params)
#psnr
params = fit_linear(x_train_allabrs_psnr, y_train_allabrs_psnr)
np.save('trained_models_allabr/psnr_allabr', params)
#ssim
params = fit_linear(x_train_allabrs_ssim, y_train_allabrs_ssim)
np.save('trained_models_allabr/ssim_allabr', params)
#vmaf
params = fit_linear(x_train_allabrs_vmaf, y_train_allabrs_vmaf)
np.save('trained_models_allabr/vmaf_allabr', params)
#FTW
a, b, c, d=fit_nonlinear((x_train_allabrs_FTW[:, 0], x_train_allabrs_FTW[:, 1]), y_train_allabrs_FTW)
np.save('trained_models_allabr/FTW_allabr', [a, b, c, d])
#SDNdash
params = fit_linear(x_train_allabrs_SDNdash, y_train_allabrs_SDNdash)
np.save('trained_models_allabr/SDNdash_allabr', params)
#videoAtlas
pickle.dump(fit_supreg(x_train_allabrs_videoAtlas, y_train_allabrs_videoAtlas), open( 'trained_models_allabr/videoAtlas_allabr.pkl', 'wb'))

#for each ABR calculate the scores of each model
l = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']
scores_FTW=[]
scores_bits=[]
scores_logbits=[]
scores_psnr=[]
scoress_ssim=[]
scores_vmaf=[]
scores_SDNdash=[]
scores_videoAtlas=[]
scores_biqps=[]
scores_p1203=[]
for nr_abr,abr in enumerate(ABRs):
    if not os.path.exists('predictions_'+ABRs[nr_abr]):
        os.makedirs('predictions_'+ABRs[nr_abr])
    for nr_mod,mod in enumerate(l):
        if mod=='FTW':
            a, b, c, d=np.load('trained_models_allabr/FTW_allabr.npy')
            x1, x2 = X_test_for_each_abr[nr_abr][nr_mod][:,0], X_test_for_each_abr[nr_abr][nr_mod][:, 1]
            score = a * np.exp(-(b * x1 + c) * x2) + d
            np.save('predictions_'+ABRs[nr_abr]+'/scores_FTW_'+ABRs[nr_abr], score)
            scores_FTW.append(score)
        elif mod=='videoAtlas':
            with open('trained_models_allabr/videoAtlas_allabr.pkl', 'rb') as handle:
                pickled_atlas = pickle.load(handle)
            videoAtlasregressor = pickled_atlas
            score = videoAtlasregressor.predict(X_test_for_each_abr[nr_abr][nr_mod])
            np.save('predictions_'+ABRs[nr_abr]+'/scores_videoAtlas_'+ABRs[nr_abr], score)
            scores_videoAtlas.append(score)
        else:
            params=np.load('trained_models_allabr/'+mod+'_allabr.npy')
            score=np.dot(params, X_test_for_each_abr[nr_abr][nr_mod].T)
            np.save('predictions_'+ABRs[nr_abr]+'/scores_'+mod+'_'+ABRs[nr_abr], score)
            if mod=='bit':
                scores_bits.append(score)
            elif mod=='logbit':
                scores_logbits.append(score)
            elif mod=='psnr':
                scores_psnr.append(score)
            elif mod=='ssim':
                scoress_ssim.append(score)
            elif mod=='vmaf':
                scores_vmaf.append(score)
            elif mod=='SDNdash':
                scores_SDNdash.append(score)


    #LSTM
    #calculate the biLSTM scores for each experience
    filetxt='features_and_scores_qoes_'+ABRs[nr_abr]+'/filelist.txt'
    filetxtest='features_and_scores_qoes_'+ABRs[nr_abr]+'/filelist_test.txt'
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
            #X_std = (onefive - 1) / (5 - 1)
            #X_scaled = X_std * (100 - 1) + 1
            scoresbiqps.append(onefive)
    np.save('predictions_'+ABRs[nr_abr]+'/scores_biqps_'+ABRs[nr_abr],scoresbiqps)
    scores_biqps.append(scoresbiqps)

    # P1203
    scoresp1203 = []
    for conta in indices_test:
        f = open('features_and_scores_qoes_' + ABRs[nr_abr] + '/exp' + str(conta) + '.json')
        data = json.load(f)
        p1203_results = P1203Standalone(data).calculate_complete()
        onefive = p1203_results['O46']
        print(onefive)
        #X_std = (onefive - 1) / (5 - 1)
        #X_scaled = X_std * (100 - 1) + 1
        scoresp1203.append(onefive)

    np.save('predictions_' + ABRs[nr_abr] + '/scores_p1203_' + ABRs[nr_abr], scoresp1203)
    scores_p1203.append(scoresp1203)

for nr_abr,abr in enumerate(ABRs[:-1]):
    data = {
        'ABR': ABRs[nr_abr],
        'Mean_FTW': np.mean(scores_FTW[nr_abr]),
        'Mean_bits': np.mean(scores_bits[nr_abr]),
        'Mean_logbits': np.mean(scores_logbits[nr_abr]),
        'Mean_psnr': np.mean(scores_psnr[nr_abr]),
        'Mean_ssim': np.mean(scoress_ssim[nr_abr]),
        'Mean_vmaf': np.mean(scores_vmaf[nr_abr]),
        'Mean_SDNdash': np.mean(scores_SDNdash[nr_abr]),
        'Mean_videoAtlas': np.mean(scores_videoAtlas[nr_abr]),
        'Mean_scoresbiqps': np.mean(scores_biqps[nr_abr]),
        'Mean_scoresp1203': np.mean(scores_p1203[nr_abr])
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


#calculate average of every list in list
FTW=[]
SDNdash=[]
videoAtlas=[]
biqps=[]
p1203=[]
b=[]
logb=[]
psnr=[]
ssim=[]
vmaf=[]
for i in range(5):
    FTW.append(np.mean(scores_FTW[i]))
    SDNdash.append(np.mean(scores_SDNdash[i]))
    videoAtlas.append(np.mean(scores_videoAtlas[i]))
    biqps.append(np.mean(scores_biqps[i]))
    p1203.append(np.mean(scores_p1203[i]))
    b.append(np.mean(scores_bits[i]))
    logb.append(np.mean(scores_logbits[i]))
    psnr.append(np.mean(scores_psnr[i]))
    ssim.append(np.mean(scoress_ssim[i]))
    vmaf.append(np.mean(scores_vmaf[i]))
