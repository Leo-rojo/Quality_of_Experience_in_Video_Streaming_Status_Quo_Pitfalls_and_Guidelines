import numpy as np
from sklearn.model_selection import train_test_split
rs=42

names=['mae_bit','mae_logbit','mae_psnr','mae_ssim','mae_vmaf','mae_ftw','mae_va','mae_lstm','mae_p1203']#'mae_sdn'
model_best_all=[]
for device in ['hdtv','phone','uhdtv']:
    users_scores=np.load('allfeat_allscores'+'/users_scores_'+device+'.npy',allow_pickle=True)
    mos_scores=np.mean(users_scores,axis=0)
    X_train, X_test, y_train, y_test = train_test_split(mos_scores, mos_scores, test_size=0.3, random_state=rs)

    #take all the scores and put in columns with the y_test
    folder='predictions_'+device+'/'
    bit_model=np.load(folder+'scores_bits_'+device+'.npy',allow_pickle=True)
    logbit_model=np.load(folder+'scores_logbits_'+device+'.npy',allow_pickle=True)
    psnr_model=np.load(folder+'scores_psnr_'+device+'.npy',allow_pickle=True)
    ssim_model=np.load(folder+'scores_ssim_'+device+'.npy',allow_pickle=True)
    vmaf_model=np.load(folder+'scores_vmaf_'+device+'.npy',allow_pickle=True)
    ftw_model=np.load(folder+'scores_ftw_'+device+'.npy',allow_pickle=True)[0]
    #sdn_model=np.load(folder+'scores_SDNdash_'+device+'.npy',allow_pickle=True)
    va_model=np.load(folder+'scores_videoAtlas_'+device+'.npy',allow_pickle=True)
    lstm_model=np.load(folder+'scores_biqps_'+device+'.npy',allow_pickle=True)
    p1203_model=np.load(folder+'scores_p1203_'+device+'.npy',allow_pickle=True)

    #calculate mae and rmse for each model and put in df
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    import pandas as pd
    #calculate mae and rmse for each model
    mae_bit=mean_absolute_error(y_test, bit_model)
    mae_logbit=mean_absolute_error(y_test, logbit_model)
    mae_psnr=mean_absolute_error(y_test, psnr_model)
    mae_ssim=mean_absolute_error(y_test, ssim_model)
    mae_vmaf=mean_absolute_error(y_test, vmaf_model)
    mae_ftw=mean_absolute_error(y_test, ftw_model)
    #mae_sdn=mean_absolute_error(y_test, sdn_model)
    mae_va=mean_absolute_error(y_test, va_model)
    mae_lstm=mean_absolute_error(y_test, lstm_model)
    mae_p1203=mean_absolute_error(y_test, p1203_model)
    rmse_bit = sqrt(mean_squared_error(y_test, bit_model))
    rmse_logbit = sqrt(mean_squared_error(y_test, logbit_model))
    rmse_psnr = sqrt(mean_squared_error(y_test, psnr_model))
    rmse_ssim = sqrt(mean_squared_error(y_test, ssim_model))
    rmse_vmaf = sqrt(mean_squared_error(y_test, vmaf_model))
    rmse_ftw = sqrt(mean_squared_error(y_test, ftw_model))
    #rmse_sdn = sqrt(mean_squared_error(y_test, sdn_model))
    rmse_va = sqrt(mean_squared_error(y_test, va_model))
    rmse_lstm = sqrt(mean_squared_error(y_test, lstm_model))
    rmse_p1203 = sqrt(mean_squared_error(y_test, p1203_model))
    #put in df with model names in idx
    df_metrics = pd.DataFrame({'mae': [mae_bit,mae_logbit,mae_psnr,mae_ssim,mae_vmaf,mae_ftw,mae_va,mae_lstm,mae_p1203], 'rmse': [rmse_bit,rmse_logbit,rmse_psnr,rmse_ssim,rmse_vmaf,rmse_ftw,rmse_va,rmse_lstm,rmse_p1203]})
    df_metrics.index = names
    print(df_metrics)



    #save model in excel with y_test
    import pandas as pd
    #print all lengths
    print(len(y_test),len(bit_model),len(logbit_model),len(psnr_model),len(ssim_model),len(vmaf_model),len(ftw_model),len(va_model),len(lstm_model),len(p1203_model))
    df = pd.DataFrame({'y_test': y_test, 'bit': bit_model, 'logbit': logbit_model, 'psnr': psnr_model, 'ssim': ssim_model, 'vmaf': vmaf_model, 'ftw': ftw_model,'va': va_model, 'lstm': lstm_model, 'p1203': p1203_model})
    #df.to_excel('models.xlsx', sheet_name='sheet1', index=False)
    #add column with name of model which is closes to y_test
    model_names=['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'ftw', 'va', 'lstm', 'p1203']
    models_best=[]
    for i in range(len(bit_model)):
        diffs=[]
        for model in [bit_model, logbit_model, psnr_model, ssim_model, vmaf_model, ftw_model,  va_model, lstm_model, p1203_model]:
            diffs.append(abs(y_test[i] - model[i]))
        #index min value of diffs
        min_index=diffs.index(min(diffs))
        models_best.append(model_names[min_index])
    model_best_all.append(models_best)
    #add column models best
    df['models_best']=models_best
    df.to_excel('models_'+device+'.xlsx', sheet_name='sheet1', index=False)

    #plot counts for the last column of df without seaborn
    import pandas as pd
    import matplotlib.pyplot as plt
    column_counts = df["models_best"].value_counts()
    plt.figure(figsize=(10, 8))  # Optional: Set the figure size
    plt.bar(column_counts.index, column_counts.values)
    plt.xlabel("Unique Values")
    plt.ylabel("Count")
    plt.title("Histogram of Unique Values in 'column_name'")
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability if needed
    plt.savefig('histogram_'+device+'.png')
    plt.close()



#count how many times each model is the best and plot the barplot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#count how many times each model is the best
model_best_all=np.array(model_best_all)
model_names=['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'ftw', 'va', 'lstm', 'p1203']
counts=[]
for model in model_names:
    counts.append(np.count_nonzero(model_best_all==model))
#sort from highest to smallest counts and model names
counts, model_names = zip(*sorted(zip(counts, model_names), reverse=True))
#plot the barplot
plt.figure(figsize=(10, 8))  # Optional: Set the figure size
plt.bar(model_names, counts)
plt.xlabel("Unique Values")
plt.ylabel("Count")
plt.title("Histogram of Unique Values in 'column_name'")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability if needed
plt.savefig('histogram_all.png')
plt.close()















