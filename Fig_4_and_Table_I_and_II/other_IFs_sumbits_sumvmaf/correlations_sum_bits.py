import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt

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
def rmse_log(params, x, y):
    a, b = params
    y_pred = logarithmic_model(x, a, b)
    return np.sqrt(np.mean((y - y_pred)**2))

nr_c = 7
os.chdir('/Fig_3')
folder_data = 'allfeat_allscores/'
#load hdtv data from allfeat_allscores
hdtv_feat = np.load(folder_data+'all_feat_hdtv.npy')
hdtv_scores = np.load(folder_data+'users_scores_hdtv.npy')
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

x=np.array(all_sum_bits)
y_plcc=np.array(hdtv_mos_noreb)

# Initial guess for parameters
initial_guess = [1, 0]
initial_guess_quad = [1, 1, 1]

# Minimize MAE (Using PLCC data, as no separate MAE data is provided)
params_mae_log = minimize(mae_log, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize MAE (Using PLCC data, as no separate MAE data is provided)
params_mae_lin = minimize(mae_lin, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
params_rmse_log = minimize(rmse_log, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x

# Generate fitted curves
x_fit = np.array(sorted(x))  # Adjusted x values for plotting
y_fit_mae_log = logarithmic_model(x_fit, *params_mae_log)
y_fit_mae_lin = linear_model(x_fit, *params_mae_lin)
y_fit_rmse_log = logarithmic_model(x_fit, *params_rmse_log)
# Calculate metrics
plcc_mae_log, _ = pearsonr(y_plcc, y_fit_mae_log)
plcc_mae_lin, _ = pearsonr(y_plcc, y_fit_mae_lin)
plcc_rmse_log, _ = pearsonr(y_plcc, y_fit_rmse_log)

mae_mae_log = np.mean(np.abs(y_plcc - y_fit_mae_log))
mae_mae_lin = np.mean(np.abs(y_plcc - y_fit_mae_lin))
mae_rmse_log = np.mean(np.abs(y_plcc - y_fit_rmse_log))

rmse_mae_log = np.sqrt(np.mean((y_plcc - y_fit_mae_log)**2))
rmse_mae_lin = np.sqrt(np.mean((y_plcc - y_fit_mae_lin)**2))
rmse_rmse_log = np.sqrt(np.mean((y_plcc - y_fit_rmse_log)**2))

srcc_mae_log, _ = spearmanr(y_plcc, y_fit_mae_log)
srcc_mae_lin, _ = spearmanr(y_plcc, y_fit_mae_lin)
srcc_rmse_log, _ = spearmanr(y_plcc, y_fit_rmse_log)

print(f"PLCC metric: {plcc_mae_log} (log), {plcc_mae_lin} (lin1), {plcc_rmse_log} (log2)")
print(f"SRCC metric: {srcc_mae_log} (log), {srcc_mae_lin} (lin1), {srcc_rmse_log} (log2)")
print(f"MAE metric: {mae_mae_log} (log), {mae_mae_lin} (lin1), {mae_rmse_log} (log2)")
print(f"RMSE metric: {rmse_mae_log} (log), {rmse_mae_lin} (lin1), {rmse_rmse_log} (log2)")
# print(f"MAE metric: {mae_plcc_log} (PLCC_opt), {mae_mae_log} (MAE_opt), {mae_rmse_log} (RMSE_opt), {mae_srcc_log} (SRCC_opt)")
# print(f"RMSE metric: {rmse_plcc_log} (PLCC_opt), {rmse_mae_log} (MAE_opt), {rmse_rmse_log} (RMSE_opt), {rmse_srcc_log} (SRCC_opt)")
# print(f"SRCC metric: {srcc_plcc_log} (PLCC_opt), {srcc_mae_log} (MAE_opt), {srcc_rmse_log} (RMSE_opt), {srcc_srcc_log} (SRCC_opt)")

# Plot the results
plt.figure(figsize=(10, 6))

# Data and PLCC model
plt.scatter(x, y_plcc, label='Data')
plt.plot(x_fit, y_fit_mae_log, label='log', color='green')
plt.plot(x_fit, y_fit_mae_lin, label='lin1', color='red')
plt.plot(x_fit, y_fit_rmse_log, label='log2', color='blue')
plt.legend()
plt.show()