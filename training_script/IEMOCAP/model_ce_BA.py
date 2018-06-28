#!/usr/bin/env python
# encoding: utf-8

import os
import data
import time
import config
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf) 

classes={0:'ang',1:'hap',2:'neu',3:'sad'}

class RecognitionNN(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('weights'):
                self.weights = {
                    'W_conv1': tf.get_variable('W_conv1', [10,1,1,4],                               # [10,1,1,4],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1)),
                    'W_conv2': tf.get_variable('W_conv2', [5,1,4,8],                                # [5,1,4,8]
                                                initializer=tf.truncated_normal_initializer(stddev=0.1)),
                    'W_conv3': tf.get_variable('W_conv3', [3,1,8,16],                               # [3,1,8,16],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
                }
                
            with tf.variable_scope('biases'):
                self.biases = {
                    'b_conv1':tf.get_variable('b_conv1', [4],
                                                initializer=tf.constant_initializer(0, dtype=tf.float32)),
                    'b_conv2':tf.get_variable('b_conv2', [8],
                                                initializer=tf.constant_initializer(0, dtype=tf.float32)),
                    'b_conv3':tf.get_variable('b_conv3', [16],
                                                initializer=tf.constant_initializer(0, dtype=tf.float32))
                }
                
            # input_x.shape: [batch_size, max_step, fea_dim]
            self.input_x = tf.placeholder(tf.float32,shape=[None,None,config.fea_dim],name="inputs_x")
            # seq_len: [batch_size]
            self.seq_len = tf.placeholder(tf.int32, shape=[None], name="feature_len")
            # input_y.shape:[batch_size, emo_num]
            self.input_y = tf.placeholder(tf.int32,shape=[None,None],name="labels_y")
            self.lab_len = tf.placeholder(tf.int32, shape=[None], name="label_len")
            # batch_szie 
            self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
            # training or testing label
            self.is_train = tf.placeholder(tf.bool, None, name="is_train")
            self.keep_prob=tf.placeholder(tf.float32,name="keep_prob")
            
            self.mu=tf.placeholder(tf.float32,shape=[config.fea_dim],name="mu")
            self.var=tf.placeholder(tf.float32,shape=[config.fea_dim],name="var")
            
            fea_norm=tf.nn.batch_normalization(self.input_x, self.mu, self.var, 0, 2, 0.001, name="normalize")
            print("fea_norm after BN:", fea_norm)
            self.input_x_bn = fea_norm
            
            with tf.name_scope('cnn_net'):
                # x_data.shape:[batch_size, max_step, fea_dim, 1]
                self.x_data = tf.reshape(self.input_x_bn, [self.batch_size, -1, config.fea_dim, 1])
                print('self.x_data:', self.x_data)
                # first convolution and pooling
                with tf.name_scope('conv1'):
                    conv1 = tf.nn.conv2d(self.x_data, self.weights['W_conv1'], strides=[1,1,1,1], padding='SAME')
                    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, self.biases['b_conv1']))
                    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,3,1,1], strides=[1,2,1,1], padding='SAME')
                    print("h_pool1:", h_pool1)
                # second convolution and pooling
                with tf.name_scope('conv2'):
                    conv2 = tf.nn.conv2d(h_pool1, self.weights['W_conv2'], strides=[1,1,1,1], padding='SAME')
                    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, self.biases['b_conv2']))
                    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,3,1,1], strides=[1,2,1,1], padding='SAME')
                    print("h_pool2:", h_pool2)
                # third convolution and pooling
                with tf.name_scope('conv3'):
                    conv3 = tf.nn.conv2d(h_pool2, self.weights['W_conv3'], strides=[1,1,1,1], padding='SAME')
                    h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3, self.biases['b_conv3']))
                    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,3,1,1], strides=[1,2,1,1], padding='SAME')
                    print("h_pool3:", h_pool3)
                self.cnn_result = h_pool3
                print("self.cnn_result:", self.cnn_result)
            # self.cnn_result.shape:[batch_size, new_max_step, fea_dim*16]
            shape=self.cnn_result.get_shape().as_list()
            print('shape:', shape)
            self.cnn_result = tf.reshape(self.cnn_result, [self.batch_size, -1, shape[2]*16])
            print("self.cnn_result:", self.cnn_result)
            
            with tf.name_scope('encoder'):
                with tf.name_scope('lstm_net'):
                    count = -1
                    hidden_layer = []
                    with tf.name_scope('lstm_layer'):
                        for unit_num in config.lstm_hidden_size:
                            count = count+1
                            with tf.name_scope('lstm_cell_'+str(count)):
                                lstm_cell = tf.contrib.rnn.LSTMCell(unit_num)
                            hidden_layer.append(lstm_cell)
                            
                    stack = tf.contrib.rnn.MultiRNNCell(hidden_layer, state_is_tuple=True)
                    
                    self.new_seq_len = tf.ceil((tf.to_float(self.seq_len))/8)
                    self.new_seq_len = tf.cast(self.new_seq_len, tf.int32)
                    
                    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(stack, self.cnn_result, self.new_seq_len, dtype=tf.float32, time_major=False) # tf.ceil(tf.to_float(self.seq_len))
                    
                    print('encoder_outputs:', encoder_outputs)
                    print('encoder_states:', encoder_states)
                    self.encoder_outputs = encoder_outputs
                    print("self.encoder_outputs:", self.encoder_outputs)
            
            with tf.name_scope('decoder'):
                # define attention mechanism
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.lstm_hidden_size[0], memory=encoder_outputs,
                                                                        memory_sequence_length=self.new_seq_len, normalize=True)
                def single_lstm_cell():
                    single_cell = tf.contrib.rnn.LSTMCell(config.lstm_hidden_size[0])
                    single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
                    return single_cell
                    
                decoder_cell = tf.contrib.rnn.MultiRNNCell([single_lstm_cell() for _ in range(1)], state_is_tuple=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism, 
                                                                attention_layer_size=None, cell_input_fn=None, output_attention=False, name='Attention_Wrapper')
                decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(cell_state=encoder_states)
                output_layer = tf.layers.Dense(config.emo_num, name="decoder_dense_layer")
                
                GO_SYMBOL = 4
                END_SYMBOL =5
                start_tokens = tf.tile([GO_SYMBOL], [self.batch_size])
                end_token = END_SYMBOL
                with tf.variable_scope('embedding'):
                    embedding_dim = 4
                    decoder_vocab_size = 3
                    decoder_embedding = tf.Variable(tf.truncated_normal(shape=[decoder_vocab_size, embedding_dim], stddev=0.1), name='decoder_embedding')
                
                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=decoder_embedding, start_tokens=start_tokens, end_token=end_token)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                        initial_state=decoder_initial_state, output_layer=output_layer)
                decoder_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=True,
                                                                                                        maximum_iterations=32)
                self.decoder_outputs = decoder_outputs
                self.decoder_rnn_output = decoder_outputs.rnn_output
                self.decoder_sample_id = decoder_outputs.sample_id
                self.final_state = final_state
                self.final_sequence_lengths = final_sequence_lengths
                print('self.decoder_rnn_output:', self.decoder_rnn_output)
                
            with tf.name_scope('accuracy'):
                self.global_step = tf.Variable(0, trainable=False)
                self.logits = self.decoder_rnn_output[:, -1]
                print('self.logits:', self.logits)
                self.softmax = tf.nn.softmax(self.logits)
                att_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))
                tf.add_to_collection('losses', att_loss)
                
                self.cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
                self.optimizer = tf.train.AdamOptimizer(config.initial_learning_rate).minimize(self.cost, self.global_step)
                correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.input_y, 1))
                self.acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
                
            if config.out_model:
                saver = tf.train.Saver(max_to_keep=30)
                self.saver = saver
                
    def m_print(self, str, log_file):
        print(str) 
        if config.out_log:
            with open(log_file,'a') as fout:
                fout.write(str+"\n")

    def calculate_ua(self,labels, logits):
        total_array = np.zeros(config.class_num)
        right_array = np.zeros(config.class_num)
        
        maxindex_labels = labels
        maxindex_logits = logits 
        for index in range(len(maxindex_labels)):
            total_array[maxindex_labels[index]] += 1
            if maxindex_logits[index] == maxindex_labels[index]:
                right_array[maxindex_labels[index]] += 1
        acc_ua = right_array/total_array
                
        print('right_array:', right_array)
        print('total_array:', total_array)
        acc_ua[np.isnan(acc_ua)] = 0
        print('acc_ua:',acc_ua)
        return np.mean(acc_ua)
        
    def test_model(self, sess, val_set, mu_, var_, log_file, curr_epoch, log_dir):
        vali_total_true_labels = []
        vali_total_pre_logits = []
        vali_total_softmax = []
        
        key = []
        vali_num = 0
        vali_time = time.time()
        while True:
            keys1,val_features,seq_len1,val_labels,lab_len1,batch_num=val_set.next_batch_onehot(config.batch_size)
            
            if batch_num==0:
                break
            if batch_num < config.batch_size:
                print("vali batch_num:", batch_num)
            
            vali_num += batch_num
            val_logits, softmax= sess.run([self.logits, self.softmax], feed_dict={self.input_x:val_features,
                                                            self.seq_len:seq_len1,
                                                            self.input_y:val_labels,
                                                            self.batch_size:len(val_labels),
                                                            self.lab_len:lab_len1,
                                                            self.mu:mu_,
                                                            self.var:var_,
                                                            self.keep_prob:1.0})
            
            
            vali_total_true_labels.extend(val_labels)
            vali_total_pre_logits.extend(val_logits)
        predict_logits = np.argmax(vali_total_pre_logits, axis=1)
        true_labels = np.argmax(vali_total_true_labels, axis=1)
        
        val_acc_wa = float(np.sum(np.equal(predict_logits,true_labels))) /  len(true_labels)
        val_acc_ua = self.calculate_ua(true_labels, predict_logits)
        
        self.m_print("vali_num:%d"%vali_num, log_file)
        self.m_print("vali_time:%fs"%(time.time()-vali_time), log_file)
        return val_acc_wa, val_acc_ua
        
    def train(self, train_data_path, vali_data_path, log_file=None, model_dir=None, log_dir=None):
        cv_time_start = time.time()
        
        train_set = data.CDataSet(train_data_path, "train", shuffle=True)
        val_set = data.CDataSet(vali_data_path, "vali", shuffle=False)
        
        self.m_print("###train num:%d"%train_set.sample_num,log_file)
        self.m_print("###val num:%d"%val_set.sample_num,log_file)
        
        val_max_ua = -1
        
        configs = tf.ConfigProto()
        configs.gpu_options.allow_growth=True
        with tf.Session(config=configs, graph=self.graph) as sess:
            if config.do_visualization:
                tf.summary.scalar('Loss', self.cost)
                tf.summary.scalar('acc',self.acc)
                merged_summary_op = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter('/data/mm0105.chen/wjhan/dzy/LSTM+CTC/CCLDNN/photo/', tf.get_default_graph())
                
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            
            val_acc_cv_wa = []
            val_acc_cv_ua = []
            train_acc_cv_wa = []
            train_acc_cv_ua = []
            
            for curr_epoch in range(config.num_epoches):
                self.m_print("###Epoch %d begin"%curr_epoch, log_file)
                self.m_print("---", log_file)
                
                epoch_time_start = time.time()
                
                i = 0
                epoch_loss = 0
                epoch_right_num = 0
                epoch_total_num = 0
                # train 
                while True:
                    batch_start_time = time.time()
                    keys1,features1,seq_len1,labels1,lab_len1,batch_num=train_set.next_batch_onehot(config.batch_size)
                    if batch_num == 0:
                        break
                    
                    if batch_num < config.batch_size:
                        print("train batch_num:", batch_num)
                        
                    train_feed = {self.input_x: features1, 
                                    self.seq_len:seq_len1,
                                    self.input_y: labels1,
                                    self.batch_size:len(labels1),
                                    self.lab_len:lab_len1,
                                    self.mu:train_set.mu,
                                    self.var:train_set.var,
                                    self.keep_prob:0.5}
                    
                    if config.do_visualization:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, batch_cost, batch_acc, step, summary= sess.run([self.optimizer, self.cost, self.acc, self.global_step, merged_summary_op], 
                                                                            feed_dict=train_feed, options=run_options,run_metadata=run_metadata)
                    
                        train_writer.add_run_metadata(run_metadata, 'step%05d' % step)
                        train_writer.add_summary(summary,step)
                    else:
                        _, batch_cost, batch_acc, step= sess.run([self.optimizer, self.cost, self.acc, self.global_step], feed_dict=train_feed)
                    batch_right_num = int(batch_num*batch_acc)
                    
                    epoch_loss = (epoch_loss*epoch_total_num + batch_cost) / (epoch_total_num + batch_num)
                    epoch_right_num = epoch_right_num+batch_right_num
                    epoch_total_num = epoch_total_num+batch_num
                    
                    self.m_print("###epoch %d"%curr_epoch, log_file)
                    self.m_print("###global_step:%d"%step, log_file)
                    self.m_print("###batch_cost:%f"%batch_cost, log_file)
                    self.m_print("###batch_acc:%f"%batch_acc, log_file)
                    self.m_print("###batch_num:%d"%batch_num, log_file)
                    self.m_print("###batch_right_num:%d"%batch_right_num, log_file)
                    self.m_print("###batch_time:%fs"%(time.time()-batch_start_time), log_file)
                    
                self.m_print("****************************", log_file)
                
                # test 
                val_wa, val_ua = self.test_model(sess, val_set, train_set.mu, train_set.var, log_file, curr_epoch, log_dir)
                if val_ua>val_max_ua:
                    val_max_ua = val_ua
                    if config.out_model:
                        model_file = os.path.join(model_dir, 'model.ckpt')
                        rt = self.saver.save(sess, model_file.replace('.ckpt', '_'+str(curr_epoch)+'.ckpt'))
                        self.m_print("model saved in %s"% rt, log_file)
                
                self.m_print('epoch_recognition_correct_nums:%d'%epoch_right_num, log_file)
                self.m_print('epoch_total_nums:%d'%epoch_total_num, log_file)
                train_acc_wa = (float(epoch_right_num) / float(epoch_total_num))
                
                # wa: epoch_correct_num / epoch_total_num
                val_acc_cv_wa.append(val_wa)
                val_acc_cv_ua.append(val_ua)
                train_acc_cv_wa.append(train_acc_wa)
                
                # 打印每个epoch的信息
                self.m_print('\nEpoch %d finished'%curr_epoch, log_file)
                self.m_print('Epoch loss %f'%epoch_loss, log_file)
                self.m_print('Epoch train_acc_wa %f'%(train_acc_wa), log_file)
                self.m_print('Epoch %d val_acc_wa %f'%(curr_epoch, val_wa), log_file)
                self.m_print('Epoch %d val_acc_ua %f'%(curr_epoch, val_ua), log_file)
                self.m_print('Epoch_time_cost:%fs'%(time.time()-epoch_time_start), log_file)
                
            self.m_print('-------------------------------------------------------------------------', log_file)
            self.m_print("###cv finished", log_file)
            self.m_print('--------UA------------', log_file)
            self.m_print('val_acc_cv_ua_max:%f in epoch %d'%(val_acc_cv_ua[np.argmax(val_acc_cv_ua)], np.argmax(val_acc_cv_ua)), log_file)
            self.m_print('val_acc_cv_wa_max:%f'%(val_acc_cv_ua[np.argmax(val_acc_cv_wa)]), log_file)
            
            self.m_print('--------WA------------',log_file)
            self.m_print('val_acc_cv_wa_max:%f in epoch %d'%(val_acc_cv_wa[np.argmax(val_acc_cv_wa)], np.argmax(val_acc_cv_wa)), log_file)
            self.m_print('train_acc_cv_wa_max:%f in epoch %d'%(train_acc_cv_wa[np.argmax(val_acc_cv_wa)], np.argmax(val_acc_cv_wa)), log_file)
            self.m_print('val_acc_cv_ua_max:%f'%(val_acc_cv_ua[np.argmax(val_acc_cv_wa)]), log_file)
            
            self.m_print('cv_time_cost:%fs'%(time.time()-cv_time_start), log_file)
            if config.do_visualization:
                train_writer.close()
    

