# Sound Source Localization
Voice signals are monitored by deployed microphone array in a low-power state. Time-delay features of each microphone pair are extracted to estimate the direction of the sound source, which can be combined with the mobility of autonomous robot. Before usage, we first train a NN model using supervised learning, on dataset collected from GSound simulator (Schissler and Manocha, 2011). Trained NN model is then fine-tuned through online RL, in daily usage.

- Dirs:

*baseline*: traditional TDOA methos based on Geometric Features

*gcc*: extracted GCC features

*main_ssl*: filed test codes for walker

*map*: point cloud generated by mapping module

*online_wav*: ssl temp storage

*save*: pre-trained model

*wakeup*: keyword spotting, choose 1 second of raw audio as input signal. In particular, 40 MFCC features are extracted from a frame of length 40ms with a stride of 20ms, which gives 1960 (40×49) features for each 1-second audio clip

*wav*: raw audio data

- Files:

*game_multi*: simulated environment to perform online tuning

*train*: pre-training code
