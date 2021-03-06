#!/usr/bin/env python
# encoding: utf-8

import os

emo_num = 4
class_num = emo_num

fea_dim = 238
batch_size = 64
num_epoches = 100

initial_learning_rate = 0.001

##### label set ######
shuffle = True
label_add_blank = False
label_repeat_factor= 0.03


##### model cofig #####
lr_decay = False
do_dropout = True
do_batchnorm = False
do_visualization = False


lstm_hidden_size = [256]
full_connect_layer_unit = 128

##### output config #####
out_log = 1
out_model = 1

rootdir = './'

log_dir = os.path.join(rootdir, 'log')
model_dir = os.path.join(rootdir, 'model')
