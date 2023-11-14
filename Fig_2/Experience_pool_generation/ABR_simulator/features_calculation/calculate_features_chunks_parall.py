import os
import numpy as np
import shutil
from pydub.utils import mediainfo
from ffmpeg_quality_metrics import FfmpegQualityMetrics as ffqm
import multiprocessing
vmaf_4k_model='./ffmpeg-quality-metrics-master/ffmpeg_quality_metrics/vmaf_models/vmaf_4k_v0.6.1.json'

def calc_feat(folder_distorted_chunk):
    folder=folder_distorted_chunk.split('_')[0]+'_'
    distorted_chunk=folder_distorted_chunk
    chunks_features = []
    pathdis = './mkvfiles/'+folder + '/' + distorted_chunk+'.mkv'
    for original_chunk in os.listdir('./mkvfiles_original'):
        pathor = './mkvfiles_original/' + original_chunk
        if original_chunk.split('_')[1] == distorted_chunk.split('_')[1]+'.mkv':
            print(pathor+'_____'+ pathdis)
            #mediainfo(pathdis)
            ffqm_videos = ffqm(pathor,pathdis)
            ffqm_videos.calc(["ssim", "psnr", "vmaf"], vmaf_options={'n_threads': 0, 'phone_model': False,
                                                                     'model_path': vmaf_4k_model})
            gs = ffqm_videos.get_global_stats()
            psnr = gs['psnr']['average']
            ssim = gs['ssim']['average']
            vmaf = gs['vmaf']['average']
            a = mediainfo(pathdis)['size']  # in bytes
            w = mediainfo(pathdis)['width']
            h = mediainfo(pathdis)['height']
            br = float(mediainfo(pathdis)['bit_rate'])
            #bit_rate = os.popen('ffprobe -v quiet -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1 ' + pathdis).read()
            #br = float(bit_rate.split('=')[1])  # in bit/s
            chunks_features.append([a, w, h, br, psnr, ssim, vmaf])
            with open('save_info.txt', 'a') as f:
               f.write(pathdis+' '+str(psnr)+' '+str(ssim)+' '+str(vmaf))
               f.write('\n')
    if not os.path.exists('./chunk_features/'+folder):
        os.makedirs('./chunk_features/'+folder)
    np.save('./chunk_features/'+folder+'/'+folder_distorted_chunk+'.npy', chunks_features)


if __name__ == "__main__":
    import time
    from multiprocessing import Pool
    time_1 = time.time()
    params_fold=[folder for folder in os.listdir('./mkvfiles')]
    params_dist=["%.3d" % i for i in range(1, 295)]
    params=[]
    for i in params_fold:
        for k in params_dist:
            #se non esistono gia li appendi
            if not os.path.exists('chunk_features/'+str(i)+'/'+str(i+k)+'.npy'):
                params.append(i+k)
    print('total parameters='+str(len(params)))
    #params=['dim_1280x720__bit_2350k','dim_2560x1440__bit_8100k']
    nr_cpu=multiprocessing.cpu_count()
    print(nr_cpu)
    with Pool(nr_cpu-1) as p:
        p.map(calc_feat, params)
    p.close()
    time_2 = time.time()
    time_interval = time_2 - time_1
    print(time_interval)