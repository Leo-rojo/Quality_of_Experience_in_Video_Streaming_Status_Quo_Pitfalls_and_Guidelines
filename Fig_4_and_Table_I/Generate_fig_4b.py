import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
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

nr_c = 4
#folder_data = 'C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/allfeat_allscores_WIV/'
#load hdtv data from allfeat_allscores
hdtv_feat_noreb = np.load('exp_iqoe_zero_buf.npy')
hdtv_mos_noreb = np.load('mos_iQoE_zero_buff.npy')

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
    exp=exp[0]
    bit = []
    for i in range(2, (2 + nr_c * 10), 10):
        bit.append(float(exp[i]))
    # sumbit
    s_bit = np.array(bit).sum()
    m_bit = np.array(bit).mean()
    all_sum_bits.append(s_bit)
    all_mean_bits.append(m_bit)
    # sumvmaf
    vmaf = []
    for i in range(9, (9 + nr_c * 10), 10):
        vmaf.append(float(exp[i]))
    s_vmaf = np.array(vmaf).sum()
    m_vmaf = np.array(vmaf).mean()
    all_sum_vmaf.append(s_vmaf)
    all_mean_vmaf.append(m_vmaf)
    # psnr
    psnr = []
    for i in range(7, (7 + nr_c * 10), 10):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()
    m_psnr = np.array(psnr).mean()
    all_sum_psnr.append(s_psnr)
    all_mean_psnr.append(m_psnr)

    # ssim
    ssim = []
    for i in range(8, (8 + nr_c * 10), 10):
        ssim.append(float(exp[i]))
    s_ssim = np.array(ssim).sum()
    m_ssim = np.array(ssim).mean()
    all_sum_ssim.append(s_ssim)
    all_mean_ssim.append(m_ssim)

x=np.array(all_mean_psnr)
y_plcc=np.array(hdtv_mos_noreb)

# Initial guess for parameters
initial_guess = [0, 0]
initial_guess_quad = [2, 2, 1]

# Minimize MAE (Using PLCC data, as no separate MAE data is provided)
params_mae_log = minimize(mae_log, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize MAE (Using PLCC data, as no separate MAE data is provided)
params_mae_lin = minimize(mae_lin, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize SRCC (Using PLCC data)
#params_srcc_lin = minimize(srcc_lin, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize PLCC
params_plcc_quad = minimize(negative_plcc_quad, initial_guess_quad, args=(x, y_plcc), method='Nelder-Mead').x

#
# a_paramen=[params_mae_log[0]-i for i in np.arange(0,20,0.1)]
# a_paraplus=[params_mae_log[0]+i for i in np.arange(0,20,0.1)]
# b_paramen=[params_mae_log[1]-i for i in np.arange(0,20,0.1)]
# b_paraplus=[params_mae_log[1]+i for i in np.arange(0,20,0.1)]
#
# mae=[]
# rmse=[]
# for par_a in a_paramen+b_paraplus:
#     for par_b in b_paramen+b_paraplus:
#
#         # Generate fitted curves
#         x_fit = np.array(sorted(x))  # Adjusted x values for plotting
#         y_fit_mae_log = logarithmic_model(x_fit, par_a, par_b)
#         mae_mae_log = np.mean(np.abs(y_plcc - y_fit_mae_log))
#         # mae_srcc_lin = np.mean(np.abs(y_plcc - y_fit_srcc_lin))
#         rmse_mae_log = np.sqrt(np.mean((y_plcc - y_fit_mae_log) ** 2))
#         # rmse_srcc_lin = np.sqrt(np.mean((y_plcc - y_fit_srcc_lin)**2))
#         mae.append((mae_mae_log,par_a,par_b))
#         rmse.append((rmse_mae_log,par_a,par_b))
# #print min value of rmse2 and its index
# #find if there is mae<15 and rmse>17.6
# for i in range(len(mae)):
#     if mae[i][0]<14.7 and rmse[i][0]>17.6:
#         print(mae[i],rmse[i])
# df = pd.DataFrame({'mae':mae,'rmse':rmse})
# df.to_excel('C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/Fig_4_and_Table_I/mae_rmse_log_iqoe.xlsx', index=False)

# Plot the results
# fig = plt.figure(figsize=(20, 10), dpi=100)
# # Data and PLCC model
# plt.scatter(x, y_plcc, label='Data', linewidth=5.0)
# x_fit = np.array(sorted(x))  # Adjusted x values for plotting
#
#recalculate mae and rmse of y_fit_mae_lin
# mae_mae_log = np.mean(np.abs(y_plcc - y_fit_mae_log))
# rmse_mae_log = np.sqrt(np.mean((y_plcc - y_fit_mae_log) ** 2))
# plcc = pearsonr(y_plcc, y_fit_mae_log)[0]
# srcc = spearmanr(y_plcc, y_fit_mae_log)[0]
# print('mae_log','rmse_log','plcc','srcc')
# print(mae_mae_log,rmse_mae_log,plcc,srcc)
# #y_fit_mae_log = linear_model(x_fit, [a_paramen+b_paraplus][idxs[0]], [b_paramen+b_paraplus][idxs[0]])
# plt.plot(x_fit, y_fit_mae_log, label='log', color='red',linewidth=5.0)
#
# plt.xlabel('Mean PSNR')
# plt.ylabel('MOS')
# plt.show()

# Generate fitted curves
x_fit = np.array(sorted(x))  # Adjusted x values for plotting
y_fit_mae_log = logarithmic_model(x_fit, *params_mae_log)
#y_fit_mae_log = logarithmic_model(x_fit, mae[309][1], mae[309][2])  #parames at index 1565 of lin
y_fit_mae_lin = linear_model(x_fit, *params_mae_lin)
#y_fit_srcc_lin = linear_model(x_fit, *params_srcc_lin)
y_fit_plcc_quad = quadratic_model(x_fit, *params_plcc_quad)
# Calculate metrics
plcc_mae_log, _ = pearsonr(y_plcc, y_fit_mae_log)
plcc_mae_lin, _ = pearsonr(y_plcc, y_fit_mae_lin)
#plcc_srcc_lin, _ = pearsonr(y_plcc, y_fit_srcc_lin)
plcc_plcc_quad, _ = pearsonr(y_plcc, y_fit_plcc_quad)

mae_mae_log = np.mean(np.abs(y_plcc - y_fit_mae_log))
mae_mae_lin = np.mean(np.abs(y_plcc - y_fit_mae_lin))
#mae_srcc_lin = np.mean(np.abs(y_plcc - y_fit_srcc_lin))
mae_plcc_quad = np.mean(np.abs(y_plcc - y_fit_plcc_quad))

rmse_mae_log = np.sqrt(np.mean((y_plcc - y_fit_mae_log)**2))
rmse_mae_lin = np.sqrt(np.mean((y_plcc - y_fit_mae_lin)**2))
#rmse_srcc_lin = np.sqrt(np.mean((y_plcc - y_fit_srcc_lin)**2))
rmse_plcc_quad = np.sqrt(np.mean((y_plcc - y_fit_plcc_quad)**2))

srcc_mae_log, _ = spearmanr(y_plcc, y_fit_mae_log)
srcc_mae_lin, _ = spearmanr(y_plcc, y_fit_mae_lin)
#srcc_srcc_lin, _ = spearmanr(y_plcc, y_fit_srcc_lin)
srcc_plcc_quad, _ = spearmanr(y_plcc, y_fit_plcc_quad)

print('log','lin','quad')
print(f"PLCC: {round(plcc_mae_log, 3)} , {round(plcc_mae_lin, 3)} , {round(plcc_plcc_quad, 3)}")
print(f"SRCC: {round(srcc_mae_log, 3)} , {round(srcc_mae_lin, 3)} ,  {round(srcc_plcc_quad, 3)}")
print(f"MAE: {round(mae_mae_log, 3)} , {round(mae_mae_lin, 3)} ,  {round(mae_plcc_quad, 3)}")
print(f"RMSE: {round(rmse_mae_log, 3)} , {round(rmse_mae_lin, 3)} ,  {round(rmse_plcc_quad, 3)}")
# print(f"MAE metric: {mae_plcc_log} (PLCC_opt), {mae_mae_log} (MAE_opt), {mae_rmse_log} (RMSE_opt), {mae_srcc_log} (SRCC_opt)")
# print(f"RMSE metric: {rmse_plcc_log} (PLCC_opt), {rmse_mae_log} (MAE_opt), {rmse_rmse_log} (RMSE_opt), {rmse_srcc_log} (SRCC_opt)")
# print(f"SRCC metric: {srcc_plcc_log} (PLCC_opt), {srcc_mae_log} (MAE_opt), {srcc_rmse_log} (RMSE_opt), {srcc_srcc_log} (SRCC_opt)")

# Plot the results
fig = plt.figure(figsize=(20, 10), dpi=100)
# Data and PLCC model
plt.scatter(x, y_plcc, label='Data', s=180,color='green')
plt.plot(x_fit, y_fit_mae_log, label='log', color='red',linewidth=9.0)
plt.plot(x_fit, y_fit_mae_lin, label='lin1', color='black',linewidth=9.0,linestyle='--')
#plt.plot(x_fit, y_fit_srcc_lin, label='lin2', color='blue',linewidth=5.0)
plt.plot(x_fit, y_fit_plcc_quad, label='quad', color='blue',linewidth=9.0)
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
plt.yticks([1, 20, 40, 60, 80, 100])#[i for i in range(101) if i%20==0])
#xticks need to be integer
plt.xlabel('Mean PSNR')
plt.ylabel('MOS')
#plt.legend()
#save plt
fig.savefig('C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/Fig_4_and_Table_I/hdtv_corr_mean_psnr_iqoe.pdf',bbox_inches='tight')
fig.savefig('C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/Fig_4_and_Table_I/hdtv_corr_mean_psnr_iqoe.png',bbox_inches='tight')