import numpy as np
from sklearn.model_selection import train_test_split
rs=42
folder_w4='C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/features_and_scores_WIV_all_device_best_models/'
for device in ['hdtv']:#,'phone','uhdtv']:
    users_scores=np.load(folder_w4+'users_scores_'+device+'.npy',allow_pickle=True)
    mos_hdtv=np.mean(users_scores,axis=0)
    X_train, X_test, y_train, y_test = train_test_split(mos_hdtv, mos_hdtv, test_size=0.3, random_state=rs)

    #take all the scores and put in columns with the y_test
    bit_model=np.load(folder_w4+'scores_bits_'+device+'.npy',allow_pickle=True)
    logbit_model=np.load(folder_w4+'scores_logbits_'+device+'.npy',allow_pickle=True)
    psnr_model=np.load(folder_w4+'scores_psnr_'+device+'.npy',allow_pickle=True)
    ssim_model=np.load(folder_w4+'scores_ssim_'+device+'.npy',allow_pickle=True)
    vmaf_model=np.load(folder_w4+'scores_vmaf_'+device+'.npy',allow_pickle=True)
    ftw_model=np.load(folder_w4+'scores_ftw_'+device+'.npy',allow_pickle=True)[0]
    sdn_model=np.load(folder_w4+'scores_SDNdash_'+device+'.npy',allow_pickle=True)
    va_model=np.load(folder_w4+'scores_videoAtlas_'+device+'.npy',allow_pickle=True)
    lstm_model=np.load(folder_w4+'scores_biqps_'+device+'.npy',allow_pickle=True)
    p1203_model=np.load(folder_w4+'scores_p1203_'+device+'.npy',allow_pickle=True)


collect_all_models=[bit_model,logbit_model,psnr_model,ssim_model,vmaf_model,ftw_model,sdn_model,va_model,lstm_model,p1203_model]
l=['bit','logbit','psnr','ssim','vmaf','ftw','sdn','va','lstm','p1203']
for nr_m,model in enumerate(collect_all_models):
    print(l[nr_m])
    #nr elements bigger than 100 and smaller than 1
    print('nr elements bigger than 100 and smaller than 1')
    print(len([i for i in model if i>100 or i<1]))



