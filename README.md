#### README ####



Instructions on running the code

The code was tested with TensorFlow 1.13

First download the dataset and place it in the data data_folder

Run ML_MUSIC_AOA.py
This will construct an ML-MUSIC compute graph to estimate the AoA for the signals.
Next it trains the NN to calibrate the MUSIC algorithm for different training set sizes.
In the end the results are saved for training the positioning model and plotting the results.
