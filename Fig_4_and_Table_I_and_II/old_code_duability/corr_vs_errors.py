import numpy as np

# # Generate some random data
# np.random.seed(0)
# x = np.linspace(0, 10, 100)
# y = 2 * x + np.random.normal(0, 2, 100)  # Linear relationship with some random noise
#
# # Add some outliers
# y[10] = 30  # Outlier
#
# # Create a non-linear relationship
# x_nonlinear = np.linspace(0, 10, 100)
# y_nonlinear = 2 * x_nonlinear**2 + np.random.normal(0, 5, 100)  # Quadratic relationship with noise
#
# # Combine the datasets
# x_combined = np.concatenate((x, x_nonlinear))
# y_combined = np.concatenate((y, y_nonlinear))
#
# # Calculate PLCC and SRCC
# plcc = np.corrcoef(x_combined, y_combined)[0, 1]
# from scipy.stats import spearmanr
# srcc, _ = spearmanr(x_combined, y_combined)
#
#
# # Calculate RMSE and MAE
# rmse = np.sqrt(np.mean((y_combined - (2 * x_combined))**2))
# mae = np.mean(np.abs(y_combined - (2 * x_combined)))
#
# print(f"PLCC: {plcc}")
# print(f"SRCC: {srcc}")
# print(f"RMSE: {rmse}")
# print(f"MAE: {mae}")
#
# #plot the data
# import matplotlib.pyplot as plt
# plt.scatter(x_combined,y_combined)
# plt.show()



import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt

# Generate example data
np.random.seed(0)
x = np.linspace(1, 10, 100)  # Adjusted x values to avoid log(0)
y_plcc = 2 * np.log(x) + np.random.normal(0, 0.2, 100)  # Logarithmic relationship with noise for PLCC optimization

# Define logarithmic model function
def logarithmic_model(x, a, b):
    return a * np.log(x) + b
def linear_model(x, a, b):
    return a * x + b
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# Define custom loss functions
def negative_plcc(params, x, y):
    a, b = params
    y_pred = logarithmic_model(x, a, b)
    plcc, _ = pearsonr(y, y_pred)
    return -plcc

def mae(params, x, y):
    a, b = params
    y_pred = logarithmic_model(x, a, b)
    return np.mean(np.abs(y - y_pred))

def rmse(params, x, y):
    a, b, c = params
    y_pred = quadratic_model(x, a, b, c)
    return np.sqrt(np.mean((y - y_pred)**2))

def srcc(params, x, y):
    a, b = params
    y_pred = linear_model(x, a, b)
    srcc, _ = spearmanr(y, y_pred)
    return -srcc

# Initial guess for parameters
initial_guess = [1, 0]
initial_guess_quad = [1, 1, 1]

# Minimize PLCC
params_plcc = minimize(negative_plcc, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x

# Minimize MAE (Using PLCC data, as no separate MAE data is provided)
params_mae = minimize(mae, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x

# Minimize RMSE (Using PLCC data)
params_rmse = minimize(rmse, initial_guess_quad, args=(x, y_plcc), method='Nelder-Mead').x

# Minimize SRCC (Using PLCC data)
params_srcc = minimize(srcc, initial_guess, args=(x, y_plcc), method='Nelder-Mead').x

# Generate fitted curves
x_fit = np.linspace(1, 10, 100)  # Adjusted x values for plotting
y_fit_plcc = logarithmic_model(x_fit, *params_plcc)
y_fit_mae = logarithmic_model(x_fit, *params_mae)
y_fit_rmse = quadratic_model(x_fit, *params_rmse)
y_fit_srcc = linear_model(x_fit, *params_srcc)

# Calculate metrics
plcc_plcc, _ = pearsonr(y_plcc, y_fit_plcc)
plcc_mae, _ = pearsonr(y_plcc, y_fit_mae)
plcc_rmse, _ = pearsonr(y_plcc, y_fit_rmse)
plcc_srcc, _ = pearsonr(y_plcc, y_fit_srcc)

mae_plcc = np.mean(np.abs(y_plcc - y_fit_plcc))
mae_mae = np.mean(np.abs(y_plcc - y_fit_mae))
mae_rmse = np.mean(np.abs(y_plcc - y_fit_rmse))
mae_srcc = np.mean(np.abs(y_plcc - y_fit_srcc))

rmse_plcc = np.sqrt(np.mean((y_plcc - y_fit_plcc)**2))
rmse_mae = np.sqrt(np.mean((y_plcc - y_fit_mae)**2))
rmse_rmse = np.sqrt(np.mean((y_plcc - y_fit_rmse)**2))
rmse_srcc = np.sqrt(np.mean((y_plcc - y_fit_srcc)**2))

srcc_plcc, _ = spearmanr(y_plcc, y_fit_plcc)
srcc_mae, _ = spearmanr(y_plcc, y_fit_mae)
srcc_rmse, _ = spearmanr(y_plcc, y_fit_rmse)
srcc_srcc, _ = spearmanr(y_plcc, y_fit_srcc)

print(f"PLCC metric: {plcc_plcc} (PLCC_opt), {plcc_mae} (MAE_opt), {plcc_rmse} (RMSE_opt), {plcc_srcc} (SRCC_opt)")
print(f"MAE metric: {mae_plcc} (PLCC_opt), {mae_mae} (MAE_opt), {mae_rmse} (RMSE_opt), {mae_srcc} (SRCC_opt)")
print(f"RMSE metric: {rmse_plcc} (PLCC_opt), {rmse_mae} (MAE_opt), {rmse_rmse} (RMSE_opt), {rmse_srcc} (SRCC_opt)")
print(f"SRCC metric: {srcc_plcc} (PLCC_opt), {srcc_mae} (MAE_opt), {srcc_rmse} (RMSE_opt), {srcc_srcc} (SRCC_opt)")

# Plot the results
plt.figure(figsize=(10, 6))

# Data and PLCC model
plt.scatter(x, y_plcc, label='Data')
plt.plot(x_fit, y_fit_plcc, label='PLCC Fit', color='red')
plt.plot(x_fit, y_fit_mae, label='MAE Fit', color='green')
plt.plot(x_fit, y_fit_rmse, label='RMSE Fit', color='orange')
plt.plot(x_fit, y_fit_srcc, label='SRCC Fit', color='purple')
plt.legend()
plt.show()


