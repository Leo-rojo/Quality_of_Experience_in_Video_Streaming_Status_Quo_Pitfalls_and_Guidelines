## Intruction in order to generate the experience description of video pool for iQoE framework (composed of 1000 experiences)
1. Run Simulation.py to produce the set of experiences divided by abr and put them in experience collection
2. Follow the instruction in features_calculation to generate chunk features folder and put it in experience collection
3. in experience_collection run 'map_bit_to_chunk_features.py' and then run 'generate_1000_shorter_expes.py' to generate the 1000 experiences that will interact with iQoE. The final file is 'iQoE_synth_exp.npy' and 
it needs to be copied in Video_pools folder.


