## Generate actual impaired video to be added in the iQoE web application:
* input files: "iQoE_synth_exp.npy" and "ffmpeg" folder containing the ffmpeg version used by the script (you can download the version used in https://drive.google.com/file/d/123nyQa9SWzqiJ4Zd9oVmak5Gcx9aWkU-/view?usp=share_link), "gif.gif" for creating the waiting wheel
* Run 'Generate_all_videos.py' to generate 1000 videos experiences described by iQoE_synth_exp.npy, it needs a lot of time. The final videos will be saved in all_videos folder
* Run 'rename_video_correctly.py' to rename the videos in order to make them compatible with the structure of iQoE web application
* Run 'Generate_reference_videos.py' to generate 10 reference videos that will be saved in reference_videos folder
NB: in Generate_all_videos.py and Generate_reference_video.py you need to specify the path to the mkvfiles folder containing all the chunks for different encodings,
due to naming reason of this folder mkvfiles need to be copied outside the main folder "Empowerment of Atypical Viewers via Low-Effort Personalized Modeling of Video Streaming Quality"