1. Introduction

1.1 ./features	     
# Acoustic feature files are saved in this folder. The part in related to eNTERFACE is in ./features/238_eNTERFACE, and the other part in related to IEMOCAP is in ./feature/238_IEMOCAP.

1.2 ./models         
# Trained model files are provided in this folder. The partition strategy for training and test is "Leave-One-Speaker-Out", which is consistent with other state-of-the-art research works.

In ./models/IEMOCAP, there are 10 folders named as CVi, where i = 0, 1, 2, ..., 9. In each CVi, there saved the model trained with data from all speakers in IEMOCAP except speaker i.

1.3 ./testing_script 
# Scripts for testing the performance of trained models on test sets.

1.4 ./training_script
#Scripts for training models.

2. Runtime environment
    Python: 3.6.0
    Tensorflow: 1.4.0-rc1

3. Network configurarion:
    corpus:     IEMOCAP   (5531 utterances, 4 emotions, 10 subjects)
    fea_dim:    238
    network:    Encoder(CNN+LSTM(256))+ Decoder(LSTM(256)+Dense) + Attention
                CNN(pooling over time dimension, 只在时间维度上进行池化操作)
                kernel size [10,1,1,4] stride[1,1,1,1] + pooling size [1,3,1,1] stride[1,2,1,1]
                kernel size [5,1,4,8]  stride[1,1,1,1] + pooling size [1,3,1,1] stride[1,2,1,1]
                kernel size [3,1,8,16] stride[1,1,1,1] + pooling size [1,3,1,1] stride[1,2,1,1]
                
    batch_size: 64  learning_rate: 0.001   epoch_num: 100   maximum_iterations: 32 
    
    trainning_time (IEMOCAP): 17000s(epoch=100, sample_num≈5000, sample_duration=3~8s)
    validation_time(IEMOCAP): 6s (sample_num≈500, sample_duration=3~8s)

4. Accuracy
    Corpus:             IEMOCAP (10 CV Average Class accuracy = 65.7%)
    CV ID:              CV0     CV1     CV2     CV3     CV4     CV5     CV6     CV7     CV8     CV9
    Class accuracy:    70.76%  66.79%  65.90%  70.07%  61.01%  61.74%  66.41%  63.74%  65.07%  65.95%

5. General usage
    5.1 If you want to run the training scripts, just execute train.py in ./training_script/IEMOCAP.
    
    5.2 The scripts of testing trained model are in ./testing_script. 
        if you want to test the IEMOCAP models, just go to the folder ./testing_script/IEMOCAP run ./IEMOCAP_testing.py and specify a model path and utterance's features(python IEMOCAP_testing.py model_path feature_path ),
        like "python IEMOCAP_testing.py /data/mm0105.chen/wjhan/speech-emotion-recognition/Attention/models/IEMOCAP/CV0/ /data/mm0105.chen/wjhan/speech-emotion-recognition/CTC/features/238_IEMOCAP/Ses01F_impro01_F012_ang.csv"

        
        
