## Generate realistic videostreaming experiences:
* in ABR_simulator:
  * The folder 'park' contains the used simulator, the original code can be found in 'https://github.com/park-project/park'. We have already inserted the elaborated throughput traces and video_sizes. 
  The necessary files and data to generate the used traces can be downloaded from: https://drive.google.com/file/d/1ejsq95UllfzejXWS0jrRcWz5XnYf2cjd/view?usp=sharing
  * 'agent_class.py' contains implementations of various ABRs.
  * Generate_sizes_for_park.py generates video_sizes_ToS.npy to be included in park. It contanins the sizes of the different chunks considered
  * Instruction to generate the videostreaming experiences: 
    * Follow the instruction in features_calculation folder to generate the 10 used features for all the chunks. This will produce a folder called chunk_features that need to be copied in experience_collection folder.
    * Run 'Simulation.py' to generate three npy files 'exp_bb.npy', 'exp_mpc.npy', 'exp_th.npy' containing videostreaming experience description for different ABRs. 
    * Put the three previous generated files in 'experience_collection' and  run 'map_bit_to_chunk_features.py' to produce the videostreaming experience with features information for each chunk. The final files is called 'experiences_with_features.npy'.
    * Run 'generate_1000_shorter_exps.py' to generate the description of 1000 shorter experiences of 4 chunks each. The final file is called 'iQoE_synth_exp.npy' and it needs to be copied in Video_pools folder.