import numpy as np, os, subprocess, shutil, time
exps = np.load('iQoE_synth_exp.npy')

def empty_folder(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))

        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def convert_experience(one_exp, nr_chunks, nr_features_per_chunk):
    temp = []
    for k in range(nr_chunks * nr_features_per_chunk):
        if k % 10 == 0:
            temp.append((one_exp[k], one_exp[k + 1]))

    return temp


def helper_format_result_string(result):
    result = result.replace('\n', '=')
    result = result.replace('\r', '')
    x = result.split('=')
    return x


def create_stalled_video(path, pathout, namefile, path_to_gif, stall_duration, duration, video_idx):
    newname = str(namefile).split('.mkv')[0] + '.jpg'
    jpg_path = os.path.join(pathout, newname)
    file_path = os.path.join(path, namefile)
    komanda = 'ffmpeg\\ffmpeg-2021\\bin\\ffmpeg.exe -sseof -3 -i ' + file_path + ' -update 1 -q:v 1 ' + jpg_path
    os.system(komanda)
    path_mp4 = os.path.join(path, namefile)
    print(path_mp4)
    result = subprocess.run(['ffmpeg\\ffmpeg-2021\\bin\\ffprobe.exe','-v','error','-select_streams','v:0','-show_entries',
     'stream=width,height,avg_frame_rate,duration','-of','default=noprint_wrappers=1',
     path_mp4],
      stdout=(subprocess.PIPE)).stdout.decode('utf-8')
    x = helper_format_result_string(result)
    result2 = subprocess.run(['ffmpeg\\ffmpeg-2021\\bin\\ffprobe.exe','-v','error','-select_streams','a:0','-show_entries',
     'stream=sample_rate,channel_layout,codec_name','-of','default=noprint_wrappers=1',
     path_mp4],
      stdout=(subprocess.PIPE)).stdout.decode('utf-8')
    y = helper_format_result_string(result2)
    name_file_for_parallel = str(namefile).split('.mkv')[0] + str(video_idx) + '.mkv'
    path_mp4s = os.path.join(pathout, 's' + name_file_for_parallel)
    command = 'ffmpeg\\ffmpeg-2021\\bin\\ffmpeg.exe -loop 1 -i ' + jpg_path + ' -f lavfi -i anullsrc=channel_layout=' + y[5].split('(')[0] + ':sample_rate=' + y[3] + ' -t ' + str(stall_duration) + ' -c:a ' + y[1] + ' -c:v libx264 -t ' + str(stall_duration) + ' -pix_fmt yuv420p -vf scale=' + x[1] + ':' + x[3] + ' -r ' + x[5].split('/')[0] + ' -y ' + path_mp4s
    os.system(command)
    temp_path = 'temporaryList' + str(video_idx) + '.txt'
    open(temp_path, 'w').close()
    komanda = " echo file '" + path_mp4 + "'" + '  >>  ' + temp_path
    os.system(komanda)
    komanda = " echo file '" + path_mp4s + "'" + '  >>  ' + temp_path
    os.system(komanda)
    path_mp4ss = os.path.join(pathout, 'ss' + name_file_for_parallel)
    komanda = 'ffmpeg\\ffmpeg-2021\\bin\\ffmpeg.exe -f concat -safe 0 -i ' + temp_path + ' -c copy ' + path_mp4ss
    os.system(komanda)
    ss_path = os.path.join(pathout, 'sss' + name_file_for_parallel)
    subkomanda = "'gte(t," + str(duration) + ")'"
    scale_gif = 295
    up_path = os.path.join(pathout, 'ssup' + name_file_for_parallel)
    up = 'ffmpeg -i ' + path_mp4ss + ' -vf scale=3840x2160:flags=bilinear ' + up_path
    os.system(up)
    komanda = 'ffmpeg\\ffmpeg-2021\\bin\\ffmpeg.exe -i ' + up_path + ' -ignore_loop 0 -i ' + path_to_gif + ' -filter_complex "[1:v]scale=%d:%d,rotate=PI/6:c=black@0:ow=rotw(PI/6):oh=roth(PI/6) [rotate];[0:v][rotate] overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2:format=auto:shortest=1:enable=' % (scale_gif, scale_gif) + subkomanda + '" -codec:a copy -y ' + ss_path
    os.system(komanda)
    seg_path = os.path.join(pathout, 'segmentList' + str(video_idx) + '.txt')
    komanda = " echo file '" + ss_path + "'" + '  >>  ' + seg_path
    os.system(komanda)
    return x


def generate_video(features_experience, init_c, maps, chunks_nr, chunks_per_exp, video_idx, folder_mkv, folder_out):
    nr_features_per_chunk = 10
    pathout = 'temp' + str(video_idx)
    if not os.path.exists(pathout):
        os.makedirs(pathout)
    if not os.path.exists('stitch' + str(video_idx)):
        os.makedirs('stitch' + str(video_idx))
    for f in os.listdir(pathout):
        extension = f.split('.')[1]
        if extension != 'css':
            if extension != 'js':
                os.remove(os.path.join(pathout, f))

    empty_folder('stitch' + str(video_idx))
    justone = features_experience
    initial_chunk_nr = init_c
    pathcopy = 'stitch' + str(video_idx)
    for conta in range(chunks_per_exp):
        path = folder_mkv + '/mkvfiles'
        repnr_stall = convert_experience(justone, chunks_per_exp, nr_features_per_chunk)
        print(repnr_stall)
        translated = maps[int(repnr_stall[conta][0])] + '_' + chunks_nr[initial_chunk_nr + conta] + '.mkv'
        gif = 'gif.gif'
        st_dur = repnr_stall[conta][1]
        dur = 2
        if st_dur != 0:
            create_stalled_video(path, pathout, translated, gif, st_dur, dur, video_idx)
            #make number video_idx to a three digit number
            original_name = str(translated).split('.mkv')[0] + str(video_idx) + '.mkv'
            formatted_video_idx = str(video_idx).zfill(3)
            translated_parallel = str(translated).split('.mkv')[0] + formatted_video_idx + '.mkv'
            shutil.copy(pathout + '\\sss' + original_name, pathcopy + '\\sss' + translated_parallel)
            for f in os.listdir(pathout):
                extension = f.split('.')[1]
                if extension != 'css':
                    if extension != 'js':
                        os.remove(os.path.join(pathout, f))

        else:
            formatted_video_idx = str(video_idx).zfill(3)
            translated_parallel = str(translated).split('.mkv')[0] + formatted_video_idx + '.mkv'
            up = 'ffmpeg -i ' + path + '\\' + translated + ' -vf scale=3840x2160:flags=bilinear ' + pathcopy + '\\sss' + translated_parallel
            os.system(up)

    full_path = 'segmentlist' + str(video_idx) + '.txt'
    name_final_video = str(video_idx) + '.mp4'
    final_path = pathout + '\\' + name_final_video
    open('segmentlist' + str(video_idx) + '.txt', 'w').close()
    print(video_idx)
    folder = sorted((os.listdir('stitch' + str(video_idx))), key=(lambda x: int(x[-10] + x[-9] + x[-8])))
    print('------------------------------')
    print(folder)
    with open('segmentlist' + str(video_idx) + '.txt', 'w') as f:
        for files in folder:
            f.write('file stitch' + str(video_idx) + '/' + files)
            f.write('\n')

    komanda = 'ffmpeg\\ffmpeg-2021\\bin\\ffmpeg.exe -f concat -safe 0 -i ' + full_path + ' -filter:v "format=yuv420p" ' + final_path
    os.system(komanda)
    shutil.copy(final_path, folder_out + '\\' + name_final_video)
