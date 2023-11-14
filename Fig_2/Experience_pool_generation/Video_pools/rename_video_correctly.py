import os
video_folder='all_videos'
for videoname in os.listdir(video_folder):
    print(videoname)
    old_name = video_folder+'/'+videoname
    new_name_int=int(videoname.split('.')[0])
    new_name = video_folder+'/'+str(new_name_int)+'.mp4'
    # Renaming the file
    os.rename(old_name, new_name)
