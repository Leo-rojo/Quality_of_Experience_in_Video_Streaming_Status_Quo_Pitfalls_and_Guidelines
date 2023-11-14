import os
import numpy as np
from Subjective_assessment_iQoE_v2_parallel import generate_video
import random

maps = list([str(i) for i in [6, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 7, 8]])
nr_chunks=4
chunks_nr = ["%.3d" % i for i in range(1, 295)]

synthetic_experiences_ = np.load('iQoE_synth_exp.npy')
synthetic_experiences = np.delete(synthetic_experiences_, -1, axis=1)

Reference_experience=synthetic_experiences[0]
for i in range(1,len(Reference_experience),10): #zero all stalls
    Reference_experience[i]=0
for i in range(0,len(Reference_experience),10):
    Reference_experience[i] = len(maps)-1 #highest representation=12
#representation_index,rebuffering_duration,video_bitrate,chunk_size,width,height,is_best,psnr,ssim,vmaf
#kbit/s video bitrate
#chunk size in bytes
folder_mkv='' #add path to mkvfiles folder, which need to stay outside the main folder due to its name, in my case it was: 'C:\\Users\\leona\\Desktop'
for i in random.choices(range(295), k=10):
    initial_chunk_nr=i
    generate_video(Reference_experience, initial_chunk_nr, maps, chunks_nr, nr_chunks,i,folder_mkv,'reference_video')
for f in os.listdir('reference_video'):
    os.rename('reference_video/'+f,'reference_video/'+f.split('.')[0]+'_ref'+'.mp4')