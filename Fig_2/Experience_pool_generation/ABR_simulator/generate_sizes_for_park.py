import os
import numpy as np
import pandas as pd


sizes=[]
c=0
order_of_qualities=[6,9,10,11,12,13,1,2,3,4,5,7,8]

for file in order_of_qualities:
    print(file)
    rep_feat=np.load('./features_calculation/chunk_features/'+str(file)+'_.npy')
    rep=[]
    for i in range(len(rep_feat)):
        rep.append(float(rep_feat[i][0]))
    sizes.append(rep)
#save new size for simulator
new_sz=np.array(sizes)
np.save('video_sizes_ToS.npy',new_sz)