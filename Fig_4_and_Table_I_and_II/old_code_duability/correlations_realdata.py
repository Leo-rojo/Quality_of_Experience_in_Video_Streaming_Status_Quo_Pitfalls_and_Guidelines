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
# Define custom loss functions log
def negative_plcc_log(params, x, y):
    a, b = params
    y_pred = logarithmic_model(x, a, b)
    plcc, _ = pearsonr(y, y_pred)
    return -plcc
def mae_log(params, x, y):
    a, b = params
    y_pred = logarithmic_model(x, a, b)
    return np.mean(np.abs(y - y_pred))
def rmse_log(params, x, y):
    a, b = params
    y_pred = logarithmic_model(x, a, b)
    return np.sqrt(np.mean((y - y_pred)**2))
def srcc_log(params, x, y):
    a, b = params
    y_pred = logarithmic_model(x, a, b)
    srcc, _ = spearmanr(y, y_pred)
    return -srcc
# Define custom loss functions linear
def negative_plcc_lin(params, x, y):
    a, b = params
    y_pred = linear_model(x, a, b)
    plcc, _ = pearsonr(y, y_pred)
    return -plcc
def mae_lin(params, x, y):
    a, b = params
    y_pred = linear_model(x, a, b)
    return np.mean(np.abs(y - y_pred))
def rmse_lin(params, x, y):
    a, b = params
    y_pred = linear_model(x, a, b)
    return np.sqrt(np.mean((y - y_pred)**2))
def srcc_lin(params, x, y):
    a, b = params
    y_pred = linear_model(x, a, b)
    srcc, _ = spearmanr(y, y_pred)
    return -srcc
# Define custom loss functions quadratic
def negative_plcc_quad(params, x, y):
    a, b, c = params
    y_pred = quadratic_model(x, a, b, c)
    plcc, _ = pearsonr(y, y_pred)
    return -plcc
def mae_quad(params, x, y):
    a, b, c = params
    y_pred = quadratic_model(x, a, b, c)
    return np.mean(np.abs(y - y_pred))
def rmse_quad(params, x, y):
    a, b, c = params
    y_pred = quadratic_model(x, a, b, c)
    return np.sqrt(np.mean((y - y_pred)**2))
def srcc_quad(params, x, y):
    a, b, c = params
    y_pred = quadratic_model(x, a, b, c)
    srcc, _ = spearmanr(y, y_pred)
    return -srcc


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

x=np.array(all_mean_vmaf)
y_plcc=np.array(hdtv_mos_noreb)

# def rmse(params, x, y):
#     a, b, c = params
#     y_pred = quadratic_model(x, a, b, c)
#     return np.sqrt(np.mean((y - y_pred)**2))
# def srcc(params, x, y):
#     a, b = params
#     y_pred = linear_model(x, a, b)
#     srcc, _ = spearmanr(y, y_pred)
#     return -srcc

# Initial guess for parameters
initial_guess = [1, 0]
initial_guess_quad = [1, 1, 1]

# Minimize PLCC
params_plcc_log = minimize(negative_plcc_log, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize MAE (Using PLCC data, as no separate MAE data is provided)
params_mae_log = minimize(mae_log, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize RMSE (Using PLCC data)
params_rmse_log = minimize(rmse_log, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize SRCC (Using PLCC data)
params_srcc_log = minimize(srcc_log, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x

# Minimize PLCC
params_plcc_lin = minimize(negative_plcc_lin, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize MAE (Using PLCC data, as no separate MAE data is provided)
params_mae_lin = minimize(mae_lin, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize RMSE (Using PLCC data)
params_rmse_lin = minimize(rmse_lin, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize SRCC (Using PLCC data)
params_srcc_lin = minimize(srcc_lin, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x

# Minimize PLCC
params_plcc_quad = minimize(negative_plcc_quad, initial_guess_quad, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize MAE (Using PLCC data, as no separate MAE data is provided)
params_mae_quad = minimize(mae_quad, initial_guess_quad, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize RMSE (Using PLCC data)
params_rmse_quad = minimize(rmse_quad, initial_guess_quad, args=(x, y_plcc), method='Nelder-Mead').x
# Minimize SRCC (Using PLCC data)
params_srcc_quad = minimize(srcc_quad, initial_guess_quad, args=(x, y_plcc), method='Nelder-Mead').x

# Generate fitted curves
x_fit = np.array(sorted(x))  # Adjusted x values for plotting
y_fit_plcc_log = logarithmic_model(x_fit, *params_plcc_log)
y_fit_mae_log = logarithmic_model(x_fit, *params_mae_log)
y_fit_rmse_log = logarithmic_model(x_fit, *params_rmse_log)
y_fit_srcc_log = logarithmic_model(x_fit, *params_srcc_log)
# Calculate metrics
plcc_plcc_log, _ = pearsonr(y_plcc, y_fit_plcc_log)
plcc_mae_log, _ = pearsonr(y_plcc, y_fit_mae_log)
plcc_rmse_log, _ = pearsonr(y_plcc, y_fit_rmse_log)
plcc_srcc_log, _ = pearsonr(y_plcc, y_fit_srcc_log)

mae_plcc_log = np.mean(np.abs(y_plcc - y_fit_plcc_log))
mae_mae_log = np.mean(np.abs(y_plcc - y_fit_mae_log))
mae_rmse_log = np.mean(np.abs(y_plcc - y_fit_rmse_log))
mae_srcc_log = np.mean(np.abs(y_plcc - y_fit_srcc_log))

rmse_plcc_log = np.sqrt(np.mean((y_plcc - y_fit_plcc_log)**2))
rmse_mae_log = np.sqrt(np.mean((y_plcc - y_fit_mae_log)**2))
rmse_rmse_log = np.sqrt(np.mean((y_plcc - y_fit_rmse_log)**2))
rmse_srcc_log = np.sqrt(np.mean((y_plcc - y_fit_srcc_log)**2))

srcc_plcc_log, _ = spearmanr(y_plcc, y_fit_plcc_log)
srcc_mae_log, _ = spearmanr(y_plcc, y_fit_mae_log)
srcc_rmse_log, _ = spearmanr(y_plcc, y_fit_rmse_log)
srcc_srcc_log, _ = spearmanr(y_plcc, y_fit_srcc_log)

print(f"PLCC metric: {plcc_plcc_log} (PLCC_opt), {plcc_mae_log} (MAE_opt), {plcc_rmse_log} (RMSE_opt), {plcc_srcc_log} (SRCC_opt)")
print(f"MAE metric: {mae_plcc_log} (PLCC_opt), {mae_mae_log} (MAE_opt), {mae_rmse_log} (RMSE_opt), {mae_srcc_log} (SRCC_opt)")
print(f"RMSE metric: {rmse_plcc_log} (PLCC_opt), {rmse_mae_log} (MAE_opt), {rmse_rmse_log} (RMSE_opt), {rmse_srcc_log} (SRCC_opt)")
print(f"SRCC metric: {srcc_plcc_log} (PLCC_opt), {srcc_mae_log} (MAE_opt), {srcc_rmse_log} (RMSE_opt), {srcc_srcc_log} (SRCC_opt)")

# Plot the results
plt.figure(figsize=(10, 6))

# Data and PLCC model
plt.scatter(x, y_plcc, label='Data')
plt.title('logarithmic')
plt.plot(x_fit, y_fit_plcc_log, label='PLCC Fit', color='red')
plt.plot(x_fit, y_fit_mae_log, label='MAE Fit', color='green')
plt.plot(x_fit, y_fit_rmse_log, label='RMSE Fit', color='orange')
plt.plot(x_fit, y_fit_srcc_log, label='SRCC Fit', color='purple')
plt.legend()
plt.show()

##############################same for lin
#####################################################
# Generate fitted curves
x_fit = np.array(sorted(x))  # Adjusted x values for plotting
y_fit_plcc_lin = linear_model(x_fit, *params_plcc_lin)
y_fit_mae_lin = linear_model(x_fit, *params_mae_lin)
y_fit_rmse_lin = linear_model(x_fit, *params_rmse_lin)
y_fit_srcc_lin = linear_model(x_fit, *params_srcc_lin)
# Calculate metrics
plcc_plcc_lin, _ = pearsonr(y_plcc, y_fit_plcc_lin)
plcc_mae_lin, _ = pearsonr(y_plcc, y_fit_mae_lin)
plcc_rmse_lin, _ = pearsonr(y_plcc, y_fit_rmse_lin)
plcc_srcc_lin, _ = pearsonr(y_plcc, y_fit_srcc_lin)

mae_plcc_lin = np.mean(np.abs(y_plcc - y_fit_plcc_lin))
mae_mae_lin = np.mean(np.abs(y_plcc - y_fit_mae_lin))
mae_rmse_lin = np.mean(np.abs(y_plcc - y_fit_rmse_lin))
mae_srcc_lin = np.mean(np.abs(y_plcc - y_fit_srcc_lin))

rmse_plcc_lin = np.sqrt(np.mean((y_plcc - y_fit_plcc_lin)**2))
rmse_mae_lin = np.sqrt(np.mean((y_plcc - y_fit_mae_lin)**2))
rmse_rmse_lin = np.sqrt(np.mean((y_plcc - y_fit_rmse_lin)**2))
rmse_srcc_lin = np.sqrt(np.mean((y_plcc - y_fit_srcc_lin)**2))

srcc_plcc_lin, _ = spearmanr(y_plcc, y_fit_plcc_lin)
srcc_mae_lin, _ = spearmanr(y_plcc, y_fit_mae_lin)
srcc_rmse_lin, _ = spearmanr(y_plcc, y_fit_rmse_lin)
srcc_srcc_lin, _ = spearmanr(y_plcc, y_fit_srcc_lin)

print('#############lin#################')
print(f"PLCC metric: {plcc_plcc_lin} (PLCC_opt), {plcc_mae_lin} (MAE_opt), {plcc_rmse_lin} (RMSE_opt), {plcc_srcc_lin} (SRCC_opt)")
print(f"MAE metric: {mae_plcc_lin} (PLCC_opt), {mae_mae_lin} (MAE_opt), {mae_rmse_lin} (RMSE_opt), {mae_srcc_lin} (SRCC_opt)")
print(f"RMSE metric: {rmse_plcc_lin} (PLCC_opt), {rmse_mae_lin} (MAE_opt), {rmse_rmse_lin} (RMSE_opt), {rmse_srcc_lin} (SRCC_opt)")
print(f"SRCC metric: {srcc_plcc_lin} (PLCC_opt), {srcc_mae_lin} (MAE_opt), {srcc_rmse_lin} (RMSE_opt), {srcc_srcc_lin} (SRCC_opt)")

# Plot the results
plt.figure(figsize=(10, 6))

# Data and PLCC model
plt.scatter(x, y_plcc, label='Data')
plt.title('linear')
plt.plot(x_fit, y_fit_plcc_lin, label='PLCC Fit', color='red')
plt.plot(x_fit, y_fit_mae_lin, label='MAE Fit', color='green')
plt.plot(x_fit, y_fit_rmse_lin, label='RMSE Fit', color='orange')
plt.plot(x_fit, y_fit_srcc_lin, label='SRCC Fit', color='purple')
plt.legend()
plt.show()

##############################same for quad
#####################################################
# Generate fitted curves
x_fit = np.array(sorted(x))  # Adjusted x values for plotting
y_fit_plcc_quad = quadratic_model(x_fit, *params_plcc_quad)
y_fit_mae_quad = quadratic_model(x_fit, *params_mae_quad)
y_fit_rmse_quad = quadratic_model(x_fit, *params_rmse_quad)
y_fit_srcc_quad = quadratic_model(x_fit, *params_srcc_quad)
# Calculate metrics
plcc_plcc_quad, _ = pearsonr(y_plcc, y_fit_plcc_quad)
plcc_mae_quad, _ = pearsonr(y_plcc, y_fit_mae_quad)
plcc_rmse_quad, _ = pearsonr(y_plcc, y_fit_rmse_quad)
plcc_srcc_quad, _ = pearsonr(y_plcc, y_fit_srcc_quad)

mae_plcc_quad = np.mean(np.abs(y_plcc - y_fit_plcc_quad))
mae_mae_quad = np.mean(np.abs(y_plcc - y_fit_mae_quad))
mae_rmse_quad = np.mean(np.abs(y_plcc - y_fit_rmse_quad))
mae_srcc_quad = np.mean(np.abs(y_plcc - y_fit_srcc_quad))

rmse_plcc_quad = np.sqrt(np.mean((y_plcc - y_fit_plcc_quad)**2))
rmse_mae_quad = np.sqrt(np.mean((y_plcc - y_fit_mae_quad)**2))
rmse_rmse_quad = np.sqrt(np.mean((y_plcc - y_fit_rmse_quad)**2))
rmse_srcc_quad = np.sqrt(np.mean((y_plcc - y_fit_srcc_quad)**2))

srcc_plcc_quad, _ = spearmanr(y_plcc, y_fit_plcc_quad)
srcc_mae_quad, _ = spearmanr(y_plcc, y_fit_mae_quad)
srcc_rmse_quad, _ = spearmanr(y_plcc, y_fit_rmse_quad)
srcc_srcc_quad, _ = spearmanr(y_plcc, y_fit_srcc_quad)

print('#############quad#################')
print(f"PLCC metric: {plcc_plcc_quad} (PLCC_opt), {plcc_mae_quad} (MAE_opt), {plcc_rmse_quad} (RMSE_opt), {plcc_srcc_quad} (SRCC_opt)")
print(f"MAE metric: {mae_plcc_quad} (PLCC_opt), {mae_mae_quad} (MAE_opt), {mae_rmse_quad} (RMSE_opt), {mae_srcc_quad} (SRCC_opt)")
print(f"RMSE metric: {rmse_plcc_quad} (PLCC_opt), {rmse_mae_quad} (MAE_opt), {rmse_rmse_quad} (RMSE_opt), {rmse_srcc_quad} (SRCC_opt)")
print(f"SRCC metric: {srcc_plcc_quad} (PLCC_opt), {srcc_mae_quad} (MAE_opt), {srcc_rmse_quad} (RMSE_opt), {srcc_srcc_quad} (SRCC_opt)")

# Plot the results
plt.figure(figsize=(10, 6))

# Data and PLCC model
plt.scatter(x, y_plcc, label='Data')
plt.title('quadratic')
plt.plot(x_fit, y_fit_plcc_quad, label='PLCC Fit', color='red')
plt.plot(x_fit, y_fit_mae_quad, label='MAE Fit', color='green')
plt.plot(x_fit, y_fit_rmse_quad, label='RMSE Fit', color='orange')
plt.plot(x_fit, y_fit_srcc_quad, label='SRCC Fit', color='purple')
plt.legend()
plt.show()
