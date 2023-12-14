import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt
from matplotlib import cm
colori=cm.get_cmap('tab10').colors
font_axes_titles = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 60,
                        }
font_title = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 60,
                        }
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 60}
plt.rc('font', **font_general)
colori=cm.get_cmap('tab10').colors

folder_data = '../allfeat_allscores_WIV/'
#load hdtv data from allfeat_allscores
hdtv_feat = np.load(folder_data+'all_feat_hdtv.npy')
hdtv_scores = np.load(folder_data+'users_scores_hdtv.npy')

def logarithmic_model(x, a, b):
    return a * np.log(x) + b
def linear_model(x, a, b):
    return a * x + b
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def mae_log(params, x, y):
    a, b = params
    y_pred = logarithmic_model(x, a, b)
    return np.mean(np.abs(y - y_pred))
def mae_lin(params, x, y):
    a, b = params
    y_pred = linear_model(x, a, b)
    return np.mean(np.abs(y - y_pred))
def srcc_lin(params, x, y):
    a, b = params
    y_pred = linear_model(x, a, b)
    srcc, _ = spearmanr(y, y_pred)
    return -srcc
def negative_plcc_quad(params, x, y):
    a, b, c = params
    y_pred = quadratic_model(x, a, b, c)
    plcc, _ = pearsonr(y, y_pred)
    return -plcc
def rmse_lin(params, x, y):
    a, b = params
    y_pred = linear_model(x, a, b)
    return np.sqrt(np.mean((y - y_pred)**2))


nr_c = 7
#calculate mos hdtv_scores
hdtv_mos = np.mean(hdtv_scores, axis=0)


#for each row of hdtv_feat calculate the sum of rebuffering time
idxs = []
for nr,exp in enumerate(hdtv_feat):
    tot_rebuf = []
    for i in range(1, (1 + nr_c * 13 - 1), 13):
       tot_rebuf.append(float(exp[i]))
    tot_reb = np.array(tot_rebuf).sum()
    if tot_reb==0:
        idxs.append(nr)

hdtv_feat_noreb=hdtv_feat[idxs]
hdtv_mos_noreb=hdtv_mos[idxs]

#calcualte sumbit for each row of hdtv_feat_noreb
all_sum_bits= []
all_mean_bits= []
all_sum_vmaf= []
all_mean_vmaf= []
all_sum_ssim= []
all_mean_ssim= []
all_sum_psnr= []
all_mean_psnr= []
for exp in hdtv_feat_noreb:
    bit = []
    for i in range(2, (2 + nr_c * 13 - 1), 13):
        bit.append(float(exp[i]))
    # sumbit
    s_bit = np.array(bit).sum()
    m_bit = np.array(bit).mean()
    all_sum_bits.append(s_bit)
    all_mean_bits.append(m_bit)
    # sumvmaf
    vmaf = []
    for i in range(12, (12 + nr_c * 13 - 1), 13):
        vmaf.append(float(exp[i]))
    s_vmaf = np.array(vmaf).sum()
    m_vmaf = np.array(vmaf).mean()
    all_sum_vmaf.append(s_vmaf)
    all_mean_vmaf.append(m_vmaf)
    # psnr
    psnr = []
    for i in range(10, (10 + nr_c * 13 - 1), 13):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()
    m_psnr = np.array(psnr).mean()
    all_sum_psnr.append(s_psnr)
    all_mean_psnr.append(m_psnr)

    # ssim
    ssim = []
    for i in range(11, (11 + nr_c * 13 - 1), 13):
        ssim.append(float(exp[i]))
    s_ssim = np.array(ssim).sum()
    m_ssim = np.array(ssim).mean()
    all_sum_ssim.append(s_ssim)
    all_mean_ssim.append(m_ssim)

x=np.array(all_mean_vmaf)
y_plcc=np.array(hdtv_mos_noreb)

# Initial guess for parameters
initial_guess = [1, 0]
initial_guess_quad = [2, 0, -1]

# Minimize MAE (Using PLCC data, as no separate MAE data is provided)
params_mae_log = minimize(mae_log, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize MAE (Using PLCC data, as no separate MAE data is provided)
params_rmse_lin = minimize(rmse_lin, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize PLCC
params_plcc_quad = minimize(negative_plcc_quad, initial_guess_quad, args=(x, y_plcc), method='Nelder-Mead').x

# Generate fitted curves
x_fit = np.array(sorted(x))  # Adjusted x values for plotting
y_fit_mae_log = logarithmic_model(x_fit, *params_mae_log)
y_fit_mae_lin = linear_model(x_fit, *params_rmse_lin)
y_fit_plcc_quad = quadratic_model(x_fit, *params_plcc_quad)
# Calculate metrics
plcc_mae_log, _ = pearsonr(y_plcc, y_fit_mae_log)
plcc_mae_lin, _ = pearsonr(y_plcc, y_fit_mae_lin)
plcc_plcc_quad, _ = pearsonr(y_plcc, y_fit_plcc_quad)

mae_mae_log = np.mean(np.abs(y_plcc - y_fit_mae_log))
mae_mae_lin = np.mean(np.abs(y_plcc - y_fit_mae_lin))
mae_plcc_quad = np.mean(np.abs(y_plcc - y_fit_plcc_quad))

rmse_mae_log = np.sqrt(np.mean((y_plcc - y_fit_mae_log)**2))
rmse_mae_lin = np.sqrt(np.mean((y_plcc - y_fit_mae_lin)**2))
rmse_plcc_quad = np.sqrt(np.mean((y_plcc - y_fit_plcc_quad)**2))

srcc_mae_log, _ = spearmanr(y_plcc, y_fit_mae_log)
srcc_mae_lin, _ = spearmanr(y_plcc, y_fit_mae_lin)
srcc_plcc_quad, _ = spearmanr(y_plcc, y_fit_plcc_quad)

print('log','lin','quad')
print(f"PLCC: {round(plcc_mae_log, 3)} , {round(plcc_mae_lin, 3)} ,  {round(plcc_plcc_quad, 3)}")
print(f"SRCC: {round(srcc_mae_log, 3)} , {round(srcc_mae_lin, 3)} ,  {round(srcc_plcc_quad, 3)}")
print(f"MAE: {round(mae_mae_log, 3)} , {round(mae_mae_lin, 3)} ,  {round(mae_plcc_quad, 3)}")
print(f"RMSE: {round(rmse_mae_log, 3)} , {round(rmse_mae_lin, 3)} ,  {round(rmse_plcc_quad, 3)}")

# Plot the results
fig = plt.figure(figsize=(20, 10), dpi=100)
# Data and PLCC model
plt.scatter(x, y_plcc, label='Data', s=180, color='green')
plt.plot(x_fit, y_fit_mae_log, label='log', color='red',linewidth=9.0)
plt.plot(x_fit, y_fit_mae_lin, label='lin1', color='black',linewidth=9.0,linestyle='--')
plt.plot(x_fit, y_fit_plcc_quad, label='quad', color='blue',linewidth=9.0)
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
#plt.yticks([i for i in range(181) if i%30==0])
plt.yticks([1, 20, 40, 60, 80, 100])
plt.xlabel('Mean VMAF')
plt.ylabel('MOS')
#plt.legend()
#save plt
fig.savefig('hdtv_corr_mean_vmaf.pdf',bbox_inches='tight')
fig.savefig('hdtv_corr_mean_vmaf.png',bbox_inches='tight')