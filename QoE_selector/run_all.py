import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")
os.chdir('C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/QoE_selector')

os.system('python Save_features_and_user_scores_W4.py')
print('step0 done')
os.system('python collect_feat_and_scores.py')
print('step1 done')
#run Elaborate_results_group_qoes.py
os.system('python calculate_predictions_various_rs.py')
print('step2 done')
#run Elaborate_results_iQoE_test.py
os.system('python mae_rmse_of_scores.py')
print('step3 done')


