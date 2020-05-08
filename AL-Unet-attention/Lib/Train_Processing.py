import tensorflow as tf
import numpy as np
import h5py
import scipy.io as sio
import hdf5storage


def Data_Division(dataset_name,experiment=True):
    """
    :param dataset_name: the common part in the name of the dataset
    :return: 
        Dataset tuple: pair_train, pair_test, pair_valid (measurement, ground-truth)
        Sensing Model: mask pre-modeled according the optical structure
    """
    
    (data_name,mask_name),file_id_list,file_cnt_list = dataset_name,[],[]
    
    seg_direct = sio.loadmat(data_name+'_%d.mat'%(0))
    #seg_direct =  h5py.File(data_name+'_%d.mat'%(0))
    
   # feature=h5py.File('C:\integrity\deeplearning\1\AL-Unet-attention\2\AL-Unet-attention\Train_Data\Boatman\Frame_0.mat')               #读取mat文件
    #seg_direct = feature['feature_data'][:] 

    sample_truth,sample_meas = seg_direct['orig'],seg_direct['meas']
    file_id_list.append(0)
    file_cnt_list.append((seg_direct['orig'].shape[-1],seg_direct['step'][0][0]))
    print ('Test Group with %d samples step %d is recorded' % (file_cnt_list[-1][0],file_cnt_list[-1][1]))
    pair_train = (file_id_list,file_cnt_list)
    pair_valid = (file_id_list,file_cnt_list)
    pair_test  = (file_id_list,file_cnt_list)
    
    mask_file = sio.loadmat(mask_name+'.mat')
    mask = mask_file['mask']
    
    return pair_train, pair_test, pair_valid, mask, (sample_meas.transpose([2,0,1]),sample_truth)

def Data_Generator_File(dataset_name, sample_path, mask, batch_size, is_training=True):
    """
    :param dataset: the raw data, containing pairs measurements & groundtruth for train/test/valid 
    :param mask: tuple, contrain the hyperparameter of [hcube, hstride, wcube, wstride]
    :param batch_size: 
    :return: 
    """
    (data_name,mask_name),(file_ind, file_cnt) = dataset_name,sample_path
    num_file,folder_id = len(file_ind),0
    (height,width,ratio) = mask.shape
    valid_ratio=8
    
    addmeasure,(num_frame,step_max) = sio.loadmat(data_name+'_%d.mat'%(file_ind[folder_id])),file_cnt[folder_id]
    truth,measurement=addmeasure['orig'],addmeasure['meas']
    addmask=sio.loadmat(mask_name+'.mat')
    mask2=addmask['mask2']
    step = np.random.choice(np.linspace(1,step_max,step_max),1,replace=False).astype(np.int16)[0]
    #ind_end = truth.shape[-1]-(ratio-1)*step
    #ind_end = measurement.shape[-1]-1
    count1,count2=0,0
    ind_end = int(truth.shape[-1]/ratio) 
    for i in range(ind_end):
        if (i%valid_ratio==0):
            count2+= 1
        else:
            count1+= 1
    ind_train=np.zeros(count1)        
    ind_valid=np.zeros(count2)
    count_train=count1
    count_valid=count2
    count1,count2=0,0
    for i in range(ind_end):
        if (i%valid_ratio==0):
            ind_valid[count2]=i
            count2+= 1
        else:
            ind_train[count1]=i
            count1+= 1
    index1 = np.random.choice(ind_train, size=ind_train, replace=False).astype(np.int16)
    index2 = np.random.choice(ind_valid, size=ind_valid, replace=False).astype(np.int16)
    print ('File %d Imported with Step %d Samples Group %d' % (file_ind[folder_id],step,ind_end))
    count1,count2,sample_cnt,batch_cnt,list_measure,list_ground,list_mask,list_netinit,list_index = 0,0,0,0,[],[],[],[],[]
    
    while True:
        if (sample_cnt==0):
            if is_training is True:
                ind_set = index1[count1]
                count1 += 1
                if (count1==count_train):
                    sample_cnt=1
            else:
                ind_set = index2[count2]
                count2 += 1
                if (count2==count_valid):
                    sample_cnt=1
                
            ind_seq = np.linspace(ind_set*ratio,ind_set*ratio+ratio-1,ratio).astype(np.int16)            
            ground = truth[:,:,ind_seq]
            #mask1=mask2[:,:,ind_seq1]
            mask1=mask2
            phi_cross = np.matmul(np.expand_dims(mask1,-1),np.expand_dims(mask1,-2))
            measurement1=measurement[:,:,ind_set] 
            measurement2=measurement1.reshape(height,width,1)           
            net_init = np.multiply(mask1,measurement2)
                
            list_measure.append(measurement1)
            list_ground.append(ground)
            list_mask.append(mask1)
            list_netinit.append(net_init)
            list_index.append(ind_set)
            batch_cnt += 1
            
            
            if batch_cnt == batch_size:
                yield np.stack(list_measure,0),np.stack(list_ground,0),np.stack(list_mask,0),np.stack(list_netinit,0),list_index
                batch_cnt,list_measure,list_ground,list_netinit,list_mask,list_index = 0,[],[],[],[],[]
        else:            
            if folder_id == num_file-1:
                folder_id = 0
            else:
                folder_id += 1
            sample_cnt = 0
            truth,(num_frame,step_max) = sio.loadmat(data_name+'_%d.mat'%(file_ind[folder_id]))['orig'],file_cnt[folder_id]
            step = np.random.choice(np.linspace(1,step_max,step_max),1,replace=False).astype(np.int16)[0]
            ind_end = int(truth.shape[-1]/ratio) 
            index = np.random.choice(ind_end, size=ind_end, replace=False).astype(np.int16)
            print ('File %d Imported with Step %d Samples Group %d' % (file_ind[folder_id],step,ind_end))
