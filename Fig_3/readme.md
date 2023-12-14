## Folder description

- Run calculate_features_for_qoes.py inside generate_points_iqoe folder in order to generate the file points_iqoe_dataset.npy
- Run the file exemplify_capping.py in order to generate Figure 3. It uses as input data the file points_iqoe_dataset.npy and the folder dataset_120.

## Input file description

points_iqoe_dataset.npy contains 43 experiences from the 120 experiences used in the iQoE evaluation which do not contain any stall.
The experiences are described by the PSNR sum of the 4 chunks composing them and the MOS score.  The file is generated from the code in folder generate_points_iqoe.

## Generate_points_iqoe
it contains:
- users: folder from iQoE dataset
- calculate_features_for_qoes.py: it generates the file points_iqoe_dataset.npy, the file exp_iqoe_zero_buf.npy and mos_iQoE_zero_buf.npy which are used as input for the code in Fig_4_and_Table_I folder.

