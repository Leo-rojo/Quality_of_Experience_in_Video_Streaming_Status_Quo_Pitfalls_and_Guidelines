import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib import cm
nr_c = 7
folder_data = 'C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/allfeat_allscores_WIV'
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

#load hdtv data from allfeat_allscores
hdtv_feat = np.load(folder_data+'/all_feat_hdtv.npy')
hdtv_scores = np.load(folder_data+'/users_scores_hdtv.npy')
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
all_sum_vmaf= []
all_sum_ssim= []
all_sum_psnr= []
for exp in hdtv_feat_noreb:
    bit = []
    for i in range(2, (2 + nr_c * 13 - 1), 13):
        bit.append(float(exp[i]))
    # sumbit
    s_bit = np.array(bit).mean()
    all_sum_bits.append(s_bit)
    # sumvmaf
    vmaf = []
    for i in range(12, (12 + nr_c * 13 - 1), 13):
        vmaf.append(float(exp[i]))
    s_vmaf = np.array(vmaf).mean()
    all_sum_vmaf.append(s_vmaf)
    # psnr
    psnr = []
    for i in range(10, (10 + nr_c * 13 - 1), 13):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()
    all_sum_psnr.append(s_psnr)

    # ssim
    ssim = []
    for i in range(11, (11 + nr_c * 13 - 1), 13):
        ssim.append(float(exp[i]))
    s_ssim = np.array(ssim).mean()
    all_sum_ssim.append(s_ssim)

#scatterplot of all_sum_psnr vs mos
# Create and fit a linear regression model
reg = LinearRegression().fit(np.array(all_sum_psnr).reshape(-1, 1), hdtv_mos_noreb)

fig = plt.figure(figsize=(20, 10), dpi=100)
# Scatter plot of data points
plt.scatter(all_sum_psnr, hdtv_mos_noreb,s=180)

# Predicted values using the regression model
predicted_mos = reg.predict(np.array(all_sum_psnr).reshape(-1, 1))
#count predicted_mos bigger than 100
count=0
values_bigger_than_100=[]
psnrvalues_bigger_than_100=[]
for nr,i in enumerate(predicted_mos):
    if i>100:
        count+=1
        values_bigger_than_100.append(i)
        psnrvalues_bigger_than_100.append(all_sum_psnr[nr])
print('predicted_mos bigger than 100: ', count)
print('values bigger than 100: ', values_bigger_than_100)
print('psnr values bigger than 100: ', psnrvalues_bigger_than_100)

zerovalue= -reg.intercept_/reg.coef_
value100= (100-reg.intercept_)/reg.coef_
#plot solid red regression line between zerovalue and value100
plt.plot([zerovalue+1, value100], [1, 100], color='red', linestyle='-', linewidth=9)
#plot dashed red regression line between value100 and value100+50
plt.plot([value100, value100+50], [100, reg.predict(np.array(value100+50).reshape(-1, 1))], color='red', linestyle=':', linewidth=9)
#plot dashed red regression line between zerovalue-50 and zerovalue
plt.plot([zerovalue-50, zerovalue+1], [reg.predict(np.array(zerovalue-50).reshape(-1, 1)), 0], color='red', linestyle=':', linewidth=9)
#plot vertical dashed line at max(all_sum_psnr)
plt.plot([value100, value100], [reg.predict(np.array(value100).reshape(-1, 1)), reg.predict(np.array(zerovalue+1).reshape(-1, 1))], color='black', linestyle=':', linewidth=7)
#plot horizontal dashed line in reg.predict(np.array(max(all_sum_psnr)).reshape(-1, 1))
plt.plot([zerovalue+1, value100], [reg.predict(np.array(value100).reshape(-1, 1)), reg.predict(np.array(value100).reshape(-1, 1))], color='black', linestyle=':', linewidth=7)
# Create a plot
plt.plot([zerovalue+1, value100], [1, 1], color='black', linestyle=':', linewidth=7)
#plot vertical dashed line at max(all_sum_psnr)
plt.plot([zerovalue+1, zerovalue+1], [reg.predict(np.array(value100).reshape(-1, 1)), reg.predict(np.array(zerovalue+1).reshape(-1, 1))], color='black', linestyle=':', linewidth=7)
#plt.xticks([zerovalue, value100], ['0', '100'])
plt.xlim(zerovalue-50, value100+50)
plt.ylim(-20, 130)
plt.xlabel('sum_psnr')
plt.ylabel('mos')
#plt.title('Distribution of scores '+['mydata','w4data'][c])
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
ax = plt.gca()
#plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])
plt.ylabel('MOS', fontdict=font_axes_titles)
plt.xlabel('PSNR sum', fontdict=font_axes_titles)
ax.tick_params(axis='x', which='major', width=5, length=20)
ax.tick_params(axis='y', which='major', width=5, length=20, pad=20)
#plt.ylim(0, 6.1)
plt.yticks([0, 20, 40, 60, 80, 100, 120], ['0', '20', '40', '60', '80', '100','120'])
plt.savefig('hdtv_sum_psnr.pdf',bbox_inches='tight')
plt.close()

#remove this balck dashed lines
#plot vertical dashed line at min(all_sum_psnr)

points_iqoe=np.load('points_iqoe_dataset.npy')
predicted_mos_iqoe=reg.predict(points_iqoe[:,0].reshape(-1, 1))
print('min predicted mos iqoe: ', np.min(predicted_mos_iqoe))
#calculate how many points are prediction less than 1 and what are their values
count=0
values_less_than_1=[]
psnrvalues_less_than_1=[]
for nr,i in enumerate(predicted_mos_iqoe):
    if i>=1:
        count+=1
        values_less_than_1.append(i)
        psnrvalues_less_than_1.append(points_iqoe[:,0][nr])
print('predicted_mos less than 1: ', count)
print('values less than 1: ', values_less_than_1)
print('psnr values less than 1: ', psnrvalues_less_than_1)
fig = plt.figure(figsize=(20, 10), dpi=100)
# Scatter plot of data points
plt.scatter(all_sum_psnr, hdtv_mos_noreb, s=180)
plt.scatter(points_iqoe[:,0], points_iqoe[:,1], color='green', marker='o', s=180)


# Predicted values using the regression model
predicted_mos = reg.predict(np.array(all_sum_psnr).reshape(-1, 1))
zerovalue = -reg.intercept_ / reg.coef_
value100 = (100 - reg.intercept_) / reg.coef_
#plot solid red regression line between zerovalue and value100
plt.plot([zerovalue+1, value100], [1, 100], color='red', linestyle='-', linewidth=9)
#plot dashed red regression line between value100 and value100+50
plt.plot([value100, value100+50], [100, reg.predict(np.array(value100+50).reshape(-1, 1))], color='red', linestyle=':', linewidth=9)
#plot dashed red regression line between zerovalue-50 and zerovalue
plt.plot([zerovalue-50, zerovalue+1], [reg.predict(np.array(zerovalue-50).reshape(-1, 1)), 0], color='red', linestyle=':', linewidth=9)
#plot vertical dashed line at max(all_sum_psnr)
plt.plot([value100, value100], [reg.predict(np.array(value100).reshape(-1, 1)), reg.predict(np.array(zerovalue+1).reshape(-1, 1))], color='black', linestyle=':', linewidth=7)
#plot horizontal dashed line in reg.predict(np.array(max(all_sum_psnr)).reshape(-1, 1))
plt.plot([zerovalue+1, value100], [reg.predict(np.array(value100).reshape(-1, 1)), reg.predict(np.array(value100).reshape(-1, 1))], color='black', linestyle=':', linewidth=7)
# Create a plot
plt.plot([zerovalue+1, value100], [1, 1], color='black', linestyle=':', linewidth=7)
#plot vertical dashed line at max(all_sum_psnr)
plt.plot([zerovalue+1, zerovalue+1], [reg.predict(np.array(value100).reshape(-1, 1)), reg.predict(np.array(zerovalue+1).reshape(-1, 1))], color='black', linestyle=':', linewidth=7)
#plot vertical dashed line from reg.predict(min(points_iqoe[:,0])) to points_iqoe[:,1]
argm=np.argmin(points_iqoe[:,0])
#plt.plot([np.min(points_iqoe[:,0]), np.min(points_iqoe[:,0])], [reg.predict(np.array(np.min(points_iqoe[:,0])).reshape(-1, 1)), points_iqoe[:,1][argm]], color='black', linestyle=':', linewidth=3)

# plt.xticks([zerovalue, value100], ['0', '100'])
plt.xlim(zerovalue - 50, value100 + 50)
plt.ylim(-20, 130)
plt.yticks([0, 20, 40, 60, 80, 100, 120], ['0', '20', '40', '60', '80', '100','120'])
plt.xlabel('sum_psnr')
plt.ylabel('mos')
# plt.title('Distribution of scores '+['mydata','w4data'][c])
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
ax = plt.gca()
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])
plt.ylabel('MOS', fontdict=font_axes_titles)
plt.xlabel('PSNR sum', fontdict=font_axes_titles)
ax.tick_params(axis='x', which='major', width=5, length=20)
ax.tick_params(axis='y', which='major', width=5, length=20, pad=20)
# plt.ylim(0, 6.1)
plt.savefig('hdtv_sum_psnr_iqoe.pdf',bbox_inches='tight')
plt.close()


#plt.plot([np.min(all_sum_psnr), np.min(points_iqoe[:,0])], [reg.predict(np.array(np.min(points_iqoe[:,0])).reshape(-1, 1)), reg.predict(np.array(zerovalue-70).reshape(-1, 1))], color='black', linestyle=':', linewidth=3)
#plot horizontal dashed line in reg.predict(np.array(max(all_sum_psnr)).reshape(-1, 1))




#print min and max of sum_psnr
print('min sum_psnr: ', np.min(all_sum_psnr))
print('max sum_psnr: ', np.max(all_sum_psnr))
#print value of regression for max(all_sum_psnr)
print('regression value for max sum_psnr: ', reg.predict(np.array(np.max(all_sum_psnr)).reshape(-1, 1)))
#print zero value
print('zero value: ', zerovalue)
#print value100
print('value100: ', value100)








