from __future__ import absolute_import

import tensorflow as tf
import os
import yaml
import h5py

from Solver.Test_Handler import Decoder_Handler

config_filename = './Solver/Config.yaml'
scenario = 'NBA'

def main():
   
    folder_id, config_id = 'Boatman-AL-T0518165807-K0.80L0.008-RMSE', 'config_250.yaml'
    with open(config_filename) as handle:
        model_config = yaml.load(handle)
    log_dir = os.path.join(os.path.abspath('.'), model_config['result_dir'], model_config['result_model'], folder_id)

    with open(os.path.join(log_dir, config_id)) as handle:
        model_config = yaml.load(handle)
    data_name = os.path.join(os.path.abspath('.'), 'Test_Data', model_config['category'], model_config['data_name'])
    
    mask_name = os.path.join(os.path.abspath('.'),'Test_Data',model_config['category'],model_config['code_name'])
        
    dataset_name = (data_name,mask_name)
    
    tf_config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Please change the id of GPU in your local server accordingly
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        Cube_Decoder = Decoder_Handler(dataset_name=dataset_name, model_config=model_config, sess = sess, is_training=False)
        Cube_Decoder.test()

if __name__ == '__main__':
    main()
