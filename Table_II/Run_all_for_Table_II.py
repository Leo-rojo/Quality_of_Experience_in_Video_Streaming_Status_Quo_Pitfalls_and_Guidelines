import os
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.system('python extract_data_for_ABRs.py')
print('step1 done')
os.system('python calculate_qoes_donewell.py')
print('step2 done')