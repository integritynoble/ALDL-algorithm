import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import numpy as np
import math
import os
import json

from Lib.Utility import *
from Model.Base_TFModel import Basement_TFModel

class Depth_Decoder(Basement_TFModel):
    
    def __init__(self, value_sets, init_learning_rate, sess, config, is_training=True, *args, **kwargs):
        
        super(Depth_Decoder, self).__init__(sess=sess, config=config, learning_rate=init_learning_rate,is_training=is_training)
        '''
        Arguments:
            measurement: [batch, height, width, 1] compressed measurement
            initial_net: [batch, height, width, depth] phi_T_y 
            groundtruth: [batch, height, width, depth] 
            sense_mat: [1, height, width, depth] sensing matris phi 
            sense_cross: [height, width, depth, depth] phi_T_phi         
        '''
        (measurement,initial_net,groundtruth,sense_mat,sense_cross) = value_sets
        self.height,self.width,self.ratio = sense_mat.get_shape().as_list()[1:]
        #self.height,self.width,self.ratio = sense_mat.get_shape()
        
        # Initialization of the model hyperparameter, enc-dec structure, evaluation metric & Optimizier
        self.initial_parameter()
        #with tf.device("GPU:1"):
        self.decoded_image = self.encdec_handler(measurement,initial_net,groundtruth,sense_mat,sense_cross, 0.8)
        self.metric_opt(self.decoded_image, groundtruth)
        
            
    def encdec_handler(self, measurement, initial_net, groundtruth, sense_mat, sense_cross, keep_probability, 
                       phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            'updates_collections': None,
            'scale':True,
            'is_training':self.is_training,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],}
        with slim.arg_scope([slim.conv2d, slim.fully_connected,slim.conv2d_transpose],
                            weights_initializer=slim.initializers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params):
            network_inputs = (measurement, initial_net, groundtruth, sense_cross, sense_mat)
            return self.encoder_decoder(network_inputs,is_training=self.is_training,dropout_keep_prob=self.keep_prob,reuse=reuse)
        
    def encoder_decoder(self, inputs, is_training=True, dropout_keep_prob=0.8, reuse=None, scope='generator'):
        (measurement, initial_net, groundtruth, sense_cross, sense_mat) = inputs
        self.LineAggre1,self.LineAggre2,self.ShrinkOpers,self.Multiplier1,self.Multiplier2 = [],[],[],[],[]
        with tf.variable_scope(scope, 'generator', [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],stride=1, padding='SAME'):
                    #self.LineAggre.append(self.LinearProj_orig(sense_cross, initial_net))
                    #self.FreqModes.append(self.Domain_Transform(self.LineAggre[0], 0))
                   
                    #self.Multipliers.append(self.Multiplier_orig(self.FreqModes[0], self.ShrinkOpers[0]))
                    self.LineAggre1.append(initial_net)
                    self.ShrinkOpers.append(self.ShrinkOper_orig(initial_net))
                    self.LineAggre2.append(self.ShrinkOper_orig(initial_net))
                    self.Multiplier1.append(self.Multiplier_orig(initial_net))
                    self.Multiplier2.append(0*measurement)
                    #self.ShrinkOpers.append(initial_net)             
                    for stage in range(1,int(self.stages/2)):
                        result1,result2, multiplier1,multiplier2=self.LinearProj_mid(measurement, initial_net, sense_mat, self.LineAggre1[-1], 
                                                              self.ShrinkOpers[-1],self.Multiplier1[-1],self.Multiplier2[-1], stage)
                        self.LineAggre1.append(result1)
                        self.LineAggre2.append(result2) 
                        self.Multiplier1.append(multiplier1)  
                        self.Multiplier2.append(multiplier2)                      
                        #self.FreqModes.append(self.Domain_Transform(self.LineAggre[stage], stage))
                        self.ShrinkOpers.append(self.ShrinkOper_mid(self.LineAggre2[stage],stage))
                        #self.Multipliers.append(self.Multiplier_mid(self.FreqModes[stage],self.ShrinkOpers[stage],
                                                                 #   self.Multipliers[-1],stage))
                    for stage in range(int(self.stages/2),self.stages):
                        result1,result2, multiplier1,multiplier2=self.LinearProj_mid(measurement, initial_net, sense_mat, self.LineAggre1[-1], 
                                                              self.ShrinkOpers[-1],self.Multiplier1[-1],self.Multiplier2[-1],stage)
                        self.LineAggre1.append(result1)
                        self.LineAggre2.append(result2)
                        self.Multiplier1.append(multiplier1)  
                        self.Multiplier2.append(multiplier2)                        
                        #self.FreqModes.append(self.Domain_Transform(self.LineAggre[stage], stage))
                        self.ShrinkOpers.append(self.ShrinkOper_mid(self.LineAggre2[stage],stage))
                        
                    result1,result2, multiplier1,multiplier2=self.LinearProj_mid(measurement, initial_net, sense_mat, self.LineAggre1[-1], 
                                                              self.ShrinkOpers[-1],self.Multiplier1[-1],self.Multiplier2[-1],stage+1)
                    self.LineAggre1.append(result1)                                           
                    output = self.U_net(initial_net, self.LineAggre1[-1], stage+1)
                    return output
                
    def metric_opt(self, model_output, ground_truth):
        mask=None
        if self.loss_func == 'MSE':
            self.loss = loss_mse(model_output, ground_truth, mask)
        elif self.loss_func == 'RMSE':
            self.loss = loss_rmse(model_output, ground_truth, mask)
        elif self.loss_func == 'MAE':
            self.loss = loss_mae(model_output, ground_truth, mask)
        elif self.loss_func == 'SSIM':
            self.loss = loss_SSIM(model_output, ground_truth, self.mask)
        else:
            self.loss = loss_rmse(model_output, ground_truth, mask)
            
        self.metrics = calculate_metrics(model_output, ground_truth, mask)
        global_step = tf.train.get_or_create_global_step()
            
        if self.is_training:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')
        self.info_merge = tf.summary.merge_all()
             
    
    
    def ShrinkOper_orig(self, pattern):
        '''
        Input Argument:
            [#Patterns][batch height, width, depth]
        Return:
            [#Patterns][batch height, width, depth]
        '''
        shrinkage = []
        with tf.variable_scope('Shrinkage_init'):
            for ind_pattern in range(self.num_pattern):               
                shrinkage.append(pattern)
        return shrinkage

    def Multiplier_orig(self, pattern):
        '''
        Input Argument:
            [#Patterns][batch height, width, depth]
        Return:
            [#Patterns][batch height, width, depth]
        '''
        multiplier = []
        with tf.variable_scope('Shrinkage_init'):
            for ind_pattern in range(self.num_pattern):               
                multiplier.append(0*pattern)
        return multiplier
    
    
    def LinearProj_mid(self, measurement, phi_T_y, mask, linearer,shrinkage, multiplier1,multiplier2, stage):
        '''
        Input Argument:
            [batch, height, width, depth] phi_T_y: One step reconstruction initialization
            [height, width, depth, depth] phi_cross: phi_T_Phi the inner product of each tube
        Return:
            [batch, height, width, depth] Reconstruction Result
        '''
        gamma,rho,lamda,result2,multipliernew1,multipliernew2,multiplier= [],[],[],[],[],[],[] 
        with tf.variable_scope('LinearProj_%d'%(stage),reuse=None):
            with slim.arg_scope([slim.variable],dtype=tf.float32,initializer=slim.initializers.xavier_initializer(),
                                regularizer=slim.l2_regularizer(0.0),trainable=self.is_training):
                
                auxli_v1=0 
                
                for ind_pattern in range(self.num_pattern):                                  
                    gamma.append(slim.variable(name='gamma_%d'%(ind_pattern),shape=[]))                     
                    
                    multiplier=multiplier1[ind_pattern]-gamma[ind_pattern]*(linearer - shrinkage[ind_pattern])
                    aux_mid= linearer - shrinkage[ind_pattern]-multiplier/gamma[ind_pattern]                                   
                    #aux_mid = slim.fully_connected(aux_mid ,self.ratio,scope='LiPro_%d_1'%(ind_pattern))             
                    auxli_v1 += gamma[ind_pattern]*aux_mid  
                    multipliernew1.append(multiplier) 

                rho=slim.variable(name='rho_%d'%(1),shape=[])
                lamda=slim.variable(name='lamda_%d'%(1),shape=[])
                phi_f=tf.reduce_sum(tf.multiply(mask,linearer),axis=-1,keepdims=True)
                multipliernew2=multiplier2-lamda*(measurement-phi_f)
                auxli_v2=lamda*(tf.multiply(mask,(phi_f+multipliernew2/lamda))-phi_T_y)
                
                auxli_v2 =auxli_v2 + auxli_v1
                #auxli_v2 = slim.fully_connected(auxli_v2,self.ratio,activation_fn=None,scope='LiPro_%d_End'%(1))
                auxli_v=rho*auxli_v2
                
                result1=linearer - auxli_v 
                for ind_pattern in range(self.num_pattern): 
                    result2.append(result1-multipliernew1[ind_pattern]/gamma[ind_pattern])
        return result1, result2,multipliernew1,multipliernew2    
    
    def Multiplier_mid(self, linearer, shrinkage, multiplier_past, stage):
        '''
        Input Arguments:
            freq_mode [#Patterns][batch, height, width, depth]
            shrinkage [#Patterns][batch, height, width, depth]
        Return:
            multiplier[#Patterns][batch, height, width, depth]
        '''
        eta,multiplier = [],[]
        with tf.variable_scope('multiplier_%d'%(stage),reuse=None):
            with slim.arg_scope([slim.variable],dtype=tf.float32,initializer=slim.initializers.xavier_initializer(),
                                regularizer=slim.l2_regularizer(0.0),trainable=self.is_training):
                for ind_pattern in range(self.num_pattern):
                    eta.append(slim.variable(name='eta_%d'%(ind_pattern),shape=[]))
                    temp = tf.multiply(eta[ind_pattern], (linearer-shrinkage[ind_pattern]))
                    multiplier.append(multiplier_past[ind_pattern] + temp)
        return multiplier

    

    def ShrinkOper_mid(self, pattern, stage):
        '''
        Input Argument:
            [#Patterns][batch height, width, depth]
        Return:
            [#Patterns][batch height, width, depth]
        '''
        shrinkage = []
        with tf.variable_scope('Shrinkage_%d'%(stage),reuse=None):
            for ind_pattern in range(self.num_pattern):
                #pattern = freq_mode[ind_pattern]+multiplier[ind_pattern]
                pattern1 = slim.conv2d(pattern[ind_pattern],self.num_kernel,5,scope='shrink_%d_0'%(ind_pattern))
                pattern1 = slim.conv2d(pattern1,self.num_kernel,3,scope='shrink_%d_1'%(ind_pattern))
                pattern1 = slim.conv2d(pattern1,self.ratio,3,scope='shrink_%d_2'%(ind_pattern),activation_fn=None)
                shrinkage.append(pattern1)
        return shrinkage
   
    
        
    def U_net(self, phi_T_y, shrinkage, stage):
        '''
        Input Argument:
            [batch, height, width, depth] phi_T_y: One step reconstruction initialization
        Return:
            [batch, height, width, depth] Reconstruction Result
        '''
        end_points = {}  
        lamda = [] 
        with tf.variable_scope('U_net_%d'%(stage),reuse=None):
            with slim.arg_scope([slim.variable],dtype=tf.float32,initializer=slim.initializers.xavier_initializer(),
                                regularizer=slim.l2_regularizer(0.0),trainable=self.is_training):
                
                ##################### encoder ##############################################

                net = slim.conv2d(shrinkage, 32, 3, stride=1, padding='SAME',scope='en_1_1')
                net=slim.conv2d(net, 32, 3, stride=1, padding='SAME',scope='en_1_2')

                end_points['encode_1'] = net 
                net=slim.max_pool2d(net,2,stride=2,padding='SAME',scope='Pool1')
                


                net = slim.conv2d(net, 64, 3, stride=1, padding='SAME', scope='en_2_1')
                net = slim.conv2d(net,64, 3, stride=1, padding='SAME', scope='en_2_2')

                end_points['encode_2'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='Pool2')
                

                net = slim.conv2d(net, 128, 3, stride=1, padding='SAME', scope='en_3_1')
                net = slim.conv2d(net,128, 3, stride=1, padding='SAME', scope='en_3_2')
                end_points['encode_3'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='VALID', scope='Pool3')
              

                #
                net = slim.conv2d(net, 256, 3, stride=1, padding='SAME', scope='en_4_1')
                net = slim.conv2d(net,256, 3, stride=1, padding='SAME', scope='en_4_2')
                end_points['encode_4'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='Pool4')
               

                net=slim.conv2d(net, 512, 3, stride=1, padding='SAME', scope='en_5_1')
                net=slim.conv2d(net, 512, 3, stride=1, padding='SAME', scope='en_5_2')
                end_points['encode_5'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='Pool5')


                net=slim.conv2d(net, 1024, 3, stride=1, padding='SAME', scope='en_6')
                #
                net = slim.conv2d(net, 1024, 3, stride=1, padding='SAME', scope='en_7')

                # ##################### decoder ##############################################
                net = slim.conv2d_transpose(net, 512, 2, 2, padding='VALID')
                net=tf.concat([net,end_points['encode_5']],3)
                net = slim.conv2d(net, 512, 3, stride=1)
                net = slim.conv2d(net, 512, 3, stride=1)



                net=slim.conv2d_transpose(net,256,2,2,padding='VALID')
                net=tf.concat([net,end_points['encode_4']],3)
                net=slim.conv2d(net,256,3,stride=1)
                net=slim.conv2d(net,256,3,stride=1)

                

                net=slim.conv2d_transpose(net,128,2,2,padding='VALID')
                net = tf.concat([net, end_points['encode_3']], 3)
                net = slim.conv2d(net, 128, 3, stride=1)
                net = slim.conv2d(net, 128, 3, stride=1)
                net = self.attention(net, 128, 8, scope='att1')


                
                net=slim.conv2d_transpose(net,64,2,2,padding='SAME')
                net = tf.concat([net, end_points['encode_2']], 3)
                net = slim.conv2d(net, 64, 3, stride=1)
                net = slim.conv2d(net, 64, 3, stride=1)

               
                net = slim.conv2d_transpose(net, 32, 2, 2, padding='SAME')
                net = tf.concat([net, end_points['encode_1']], 3)
                net = slim.conv2d(net, 32, 3, stride=1)
                net = slim.conv2d(net, 32, 3, stride=1)
                

                net=slim.conv2d(net,self.ratio,1,stride=1,activation_fn=None)
                lamda.append(slim.variable(name='lamda_%d'%(1),shape=[self.height,self.width,self.ratio,self.batch_size]))
                net=tf.transpose(tf.multiply(tf.transpose(net,[1,2,3,0]),lamda[-1]),[3,0,1,2])
                net+=shrinkage
        return net

    def attention(self, x, ch, bs, scope='attention', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            f = slim.conv2d(x, ch // 8, 1, stride=1, scope='f_conv')
            g = slim.conv2d(x, ch // 8, 1, stride=1, scope='g_conv')
            h = slim.conv2d(x, ch, 1, stride=1, scope='h_conv') 
            
            s = tf.matmul(tf.reshape(f, shape=[self.batch_size,bs, -1, ch // 8]), tf.reshape(g, shape=[self.batch_size,bs, ch // 8, -1]))  # 
            beta = tf.nn.softmax(s, dim=-1)  # attention map

            o = tf.matmul(beta, tf.reshape(h, shape=[self.batch_size,bs,-1, ch]))  #
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape)  # 
            x = gamma * o + x

        return x
    
    
    def initial_parameter(self):
        config = self.config
        self.keep_prob = float(config.get('keep_rate_forward',0.8))
        # Parameter Initialization of Data Assignment
        self.stages = int(config.get('num_stages',2))
        self.num_pattern = int(config.get('num_pattern',10))
        self.batch_size = int(config.get('batch_size',12))
        self.trans_dim = int(config.get('trans_dim',16))
        self.num_kernel = int(config.get('num_kernel',16))
        self.num_endlayer = int(config.get('num_endlayer',3))
        noise_coe = float(config.get('noise',0.000001))*np.identity(self.ratio)
        self.noise = tf.expand_dims(tf.expand_dims(tf.constant(noise_coe,dtype=tf.float32),0),0)