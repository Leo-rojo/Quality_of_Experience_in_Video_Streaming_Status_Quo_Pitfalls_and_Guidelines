# Calculate features idx, vmaf, ssim, psnr, isbest, width, height, bitrate, stall, size
* folder ffmpeg-quality_metrics-master contain the vmaf.json file needed to calculate the vmaf. It can be downloaded online.
* run 'create_folders_mkvfiles.py' to create the folders with the right structure and copy mkvfiles_original
* copy the content from preceiding generated mkvfiles to the corresponding folder with the new structure
* run 'calculate_features_chunks_parall.py' to calculate the features for every chunks and save them in chunk_features folder
* in chunk_features folder run the file Organize_features.py to generate xls files with the features for every chunks


