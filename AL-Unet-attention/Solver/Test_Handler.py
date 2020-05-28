import tensorflow as tf
import numpy as np
import yaml
import os
import h5py
import time
import sys
import math

from Lib.Test_Processing import *
from Lib.Utility import *
from Solver.CUP_Model import CUP_Model
from Solver.Base_Handler import Basement_Handler


class Decoder_Handler(Basement_Handler):
    def __init__(self, dataset_name, model_config, sess, is_training=True):
        
        # Initialization of Configuration, Parameter and Datasets
        super(Decoder_Handler, self).__init__(sess=sess, model_config=model_config, is_training=is_training)
        self.initial_parameter() #lr
        self.data_assignment(dataset_name)

        # Data Generator
        self.gen_train = Data_Generator_File(dataset_name,self.set_train,self.sense_mask,self.batch_size,is_training=True)
        self.gen_valid = Data_Generator_File(dataset_name,self.set_valid,self.sense_mask,self.batch_size,is_training=False)
        self.gen_test  = Data_Generator_File(dataset_name,self.set_test,self.sense_mask,self.batch_size,is_training=False)
        
        # Define the general model and the corresponding input
        shape_meas = (self.batch_size,) + self.sense_mask.shape[:2] + (1,)
        shape_sense1 = self.sense_mask.shape        
        shape_sense= (self.batch_size,)+shape_sense1
        shape_truth=shape_sense
        print (shape_meas,shape_sense,shape_truth)
        
        self.meas_sample = tf.placeholder(tf.float32, shape=shape_meas, name='input_meas')
        self.initial_net = tf.placeholder(tf.float32, shape=shape_truth,name='input_init')
       # self.sense_cross = tf.placeholder(tf.float32, shape=shape_cross,name='matrix_cross')
        self.sense_matrix = tf.placeholder(tf.float32, shape=shape_sense, name='input_mat')
        self.truth_sample = tf.placeholder(tf.float32, shape=shape_truth, name='output_truth')
        
        # Initialization for the model training procedure.
        self.learning_rate = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(self.lr_init),
                                             trainable=False)
        self.lr_new = tf.placeholder(tf.float32, shape=(), name='lr_new')
        self.lr_update = tf.assign(self.learning_rate, self.lr_new, name='lr_update')
        self.train_test_valid_assignment()
        self.trainable_parameter_info()
        self.saver = tf.train.Saver(tf.global_variables())

    def initial_parameter(self):
        # Configuration Set
        config = self.model_config
        
        # Model Input Initialization
        self.batch_size = int(config.get('batch_size',1))
        self.upbound = float(config.get('upbound',1))
        
        # Initialization for Training Controler
        self.epochs = int(config.get('epochs',100))
        self.patience = int(config.get('patience',30))
        self.lr_init = float(config.get('learning_rate',0.001))
        self.lr_decay_coe = float(config.get('lr_decay',0.1))
        self.lr_decay_epoch = int(config.get('lr_decay_epoch',20))
        self.lr_decay_interval = int(config.get('lr_decay_interval',10))

    def data_assignment(self,dataset_name):
        # Division for train, test and validation
        model_config = self.model_config
        set_train, set_test, set_valid, self.sense_mask, sample = Data_Division(dataset_name)
        
        # The value of the position is normalized (the value of lat and lon are all limited in range(0,1))
        scalar = limit_scalar(self.sense_mask)
        self.phi_cross = scalar.phi_cross
        
        self.set_test, disp_test  = scalar.seperate_normalization(set_test)
        self.test_size  = int(np.ceil(float(disp_test[0]) /self.batch_size))
        
        self.set_train,disp_train = scalar.overlap_normalization(set_train)
        self.set_valid,disp_valid = scalar.overlap_normalization(set_valid)
        self.train_size = int(np.ceil(float(disp_train[0])/self.batch_size))
        self.valid_size = int(np.ceil(float(disp_valid[0])/self.batch_size))
        
    def train_test_valid_assignment(self):
        
        value_set = (self.meas_sample,self.initial_net,self.truth_sample,self.sense_matrix)
        
        with tf.name_scope('Train'):
            with tf.variable_scope('Depth_Decoder', reuse=False):
                self.Decoder_train = CUP_Model(value_set,self.learning_rate,self.sess,self.model_config,is_training=True)
        with tf.name_scope('Val'):
            with tf.variable_scope('Depth_Decoder', reuse=True):
                self.Decoder_valid = CUP_Model(value_set,self.learning_rate,self.sess,self.model_config,is_training=False)
                
    
    def test(self):
        
        print ("Testing Started")
        self.restore()
        
        test_fetches = {'pred_orig':   self.Decoder_valid.decoded_image}
        time_list = []
        for tested_batch in range(0,self.test_size):
            (measure_test1,ground_test,mask_test,netinit_test,index_test) =next(self.gen_test) #self.gen_test._next_()
            print (index_test)
            measure_test=np.expand_dims(measure_test1,-1)
            #phi_phi=np.matmul(np.transpose(mask_test,[1,2,3,0]),np.transpose(mask_test,[1,2,0,3]))
            feed_dict_test = {self.meas_sample:measure_test, 
                              self.truth_sample:ground_test,                             
                              self.initial_net:netinit_test,
                              self.sense_matrix:mask_test}                              
            start_time = time.time()
            test_output = self.sess.run(test_fetches,feed_dict=feed_dict_test)
            end_time = time.time()
            time_list.append(end_time-start_time)
            message = "Test [%d/%d] time %s"%(tested_batch+1,self.test_size,time_list[-1])
            matcontent = {}
            matcontent[u'truth'],matcontent[u'pred'],matcontent[u'meas'] = ground_test,test_output['pred_orig'],measure_test
            hdf5storage.write(matcontent, '.', self.log_dir+'/Data_Visualization_%d.mat' % (tested_batch), 
                              store_python_metadata=False, matlab_compatible=True)
            print (message)
            
        
    def calculate_scheduled_lr(self, epoch, min_lr=1e-10):
        decay_factor = int(math.ceil((epoch - self.lr_decay_epoch)/float(self.lr_decay_interval)))
        new_lr = self.lr_init * (self.lr_decay_coe ** max(0, decay_factor))
        new_lr = max(min_lr, new_lr)
        
        self.logger.info('Current learning rate to: %.6f' % new_lr)
        sys.stdout.flush()
        
        self.sess.run(self.lr_update, feed_dict={self.lr_new: new_lr})
        self.Decoder_train.set_lr(self.learning_rate) 
        return new_lr
