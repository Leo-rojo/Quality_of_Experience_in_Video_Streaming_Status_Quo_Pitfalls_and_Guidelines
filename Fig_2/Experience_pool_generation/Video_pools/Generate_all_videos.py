import numpy as np
from Subjective_assessment_iQoE_v2_parallel import generate_video

maps = list([str(i) for i in [6, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 7, 8]])
nr_chunks=4
chunks_nr = ["%.3d" % i for i in range(1, 295)]

synthetic_experiences_ = np.load('iQoE_synth_exp.npy')
synthetic_experiences = np.delete(synthetic_experiences_, -1, axis=1)

folder_mkv='' #add path to mkvfiles folder, which need to stay outside the main folder due to its name, in my case it was: 'C:\\Users\\leona\\Desktop'
for i in range(len(synthetic_experiences)):
    initial_chunk_nr=int(synthetic_experiences_[i][-1])
    generate_video(synthetic_experiences_[i], initial_chunk_nr, maps, chunks_nr, nr_chunks,i,folder_mkv,'all_videos')
