# coding:utf8
import xlsx_operation
import feature_extraction
import numpy as np
import os


def preprocessData(root):
    data_name = [os.path.join(root, sample) for sample in os.listdir(root)]
    data_num = len(data_name)
    # sort the file
    data_name = sorted(data_name, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

    all_data = [];
    min_track_length = 200
    
    for i in range(data_num):
        data = xlsx_operation.read_excel_xlsx(data_name[i],'Sheet1')
        data_len = data.shape[0]
        if data_len  < min_track_length:
              min_track_length = data_len
              
        data_len = data_len-1
        delta_r = np.zeros([data_len, 2])
        vi = np.zeros([data_len, 1])
        ai = np.zeros([data_len, 1])
        sample = np.zeros([data_len, 4])
        label = np.zeros([data_len, 1])

        for j in range(0, data_len):

           # delta_r=[x(t+1)-x(t),y(t+1)-y(t)],x and y (pixel)
            delta_r[j, 0] = data[j + 1, 1] - data[j, 1]
            delta_r[j, 1] = data[j + 1, 2] - data[j, 2]

            # vi:instantaneous speed, not um/s
            vi[j] = np.sqrt(np.square(delta_r[j, 0])+np.square(delta_r[j, 1]))

            # ai:instantaneous angle
            if delta_r[j ,0] == 0:
                ai[j] = np.pi / 2
            else:
                ai[j] = np.arctan(delta_r[j, 1] / delta_r[j, 0])

            # pay attentino: label start from 0
            label[j, 0] = data[j + 1, 3] - 1

        sample = np.concatenate([np.concatenate([delta_r,vi],1),ai],1)
        data_process = np.concatenate([sample,label],1)
        all_data.append(data_process)
    
    min_state_length = min_track_length -1
    return all_data,min_state_length

def getFeatVec(root,addFeat,shift,n):

    data_name = [os.path.join(root, sample) for sample in os.listdir(root)]
    data_num = len(data_name)
    # sort the file
    data_name = sorted(data_name, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

    all_data = [];
    min_track_length = 200
    
    for i in range(data_num):
        data = xlsx_operation.read_excel_xlsx(data_name[i],'Sheet1')
        data_len = data.shape[0]
        if data_len  < min_track_length:
              min_track_length = data_len
        
              
        data_len = data_len-1 
        sample = np.zeros([data_len, 1])
        label = np.zeros([data_len, 1])
        
        x = data[:,1]
        y = data[:,2]
        # ML:label start from 0,from the second frame(1)
        label = data[1:,3]-1
        label = label.reshape(-1,1)
        if 'deltaX' in addFeat:
            deltaX = feature_extraction.getDeltaX(x)
            sample[:,0] = deltaX
        if 'deltaY' in addFeat:
            deltaY = feature_extraction.getDeltaY(y)
            sample = np.column_stack((sample,deltaY.reshape(-1,1)))
        if 'vi' in addFeat:
            vi = feature_extraction.getVi(x,y) 
            sample = np.column_stack((sample,vi.reshape(-1,1)))
        if 'ai' in addFeat:
            ai = feature_extraction.getAi(x,y) 
            sample = np.column_stack((sample,ai.reshape(-1,1)))
        if 'dtot' in addFeat:
            dtot = feature_extraction.getDtot(x,y,shift) 
            sample = np.column_stack((sample,dtot.reshape(-1,1)))
        if 'dnet' in addFeat:
            dnet = feature_extraction.getDnet(x,y,shift)
            sample = np.column_stack((sample,dnet.reshape(-1,1)))
        if 'rcon' in addFeat:
            rcon = feature_extraction.getRcon(x,y,shift)
            sample = np.column_stack((sample,rcon.reshape(-1,1)))
        if 'msd' in addFeat:
            for jj in range(1,n+1):
                msd = feature_extraction.getMSD(x,y,shift,jj) 
                sample = np.column_stack((sample,msd.reshape(-1,1)))
        if 'msdRatio' in addFeat:
            for jj in range(1,n):
                msdRatio = feature_extraction.getMSDRatio(x,y,shift,jj,jj+1)
                sample = np.column_stack((sample,msdRatio.reshape(-1,1)))
        if 'A' in addFeat:
            A = feature_extraction.getA(x,y,shift) 
            sample = np.column_stack((sample,A.reshape(-1,1)))
        if 'K' in addFeat:
            K = feature_extraction.getK(x,y,shift) 
            sample = np.column_stack((sample,K.reshape(-1,1)))
        if 'E' in addFeat:
            E = feature_extraction.getE(x,y,shift) 
            sample = np.column_stack((sample,E.reshape(-1,1)))           
        if 'Df' in addFeat:
            Df = feature_extraction.getDf(x,y,shift) 
            sample = np.column_stack((sample,Df.reshape(-1,1))) 
        
        data_process = np.concatenate([sample,label],1)
        all_data.append(data_process)
    
    min_state_length = min_track_length -1
    return all_data,min_state_length  
    
def getSmallTrainSet(root, train_data, min_length, window_size, interval_size):

    if min_length <= window_size:
        window_size = min_length

    folder_name = str(window_size)+'_'+str(interval_size)
    train_process_folder = os.path.join(root, folder_name)
    if not os.path.exists(train_process_folder):
        os.makedirs(train_process_folder)      

    file_num = 0
    for i in range(0,len(train_data)):
        
        for j in range(0,int((train_data[i].shape[0]-window_size) / interval_size)+1):
            small_data = train_data[i][j*interval_size:j*interval_size+window_size]
            file_num += 1
            file_name = str(file_num) + '.xlsx'
            path = os.path.join(train_process_folder,file_name)
            xlsx_operation.write_excel_xlsx(path, 'Sheet1', small_data.tolist())

def getOriginSet(root,origin_root, data):
    if not os.path.exists(origin_root):
        os.makedirs(origin_root)

    #get the xlsx file name
    data_name = os.listdir(root)
    # sort the file
    data_name = sorted(data_name, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

    file_num = 0
    for i in range(0,len(data)):
        
            origin_data = data[i]
            file_name = data_name[i]
            path = os.path.join(origin_root,file_name)
            xlsx_operation.write_excel_xlsx(path, 'Sheet1', origin_data.tolist())

        
#train_root = './data_20200313/train'
#train_process_root = './data_20200304/train_process'
#train_feat_root = './data_20200313/train_feat2'
#train_origin_root = './data_20200304/train_origin'
#train_originFeat_root = './data_20200420/train_originFeat'
#test_root = './data_20200510/test'
#test_origin_root = './data_20200304/test_origin'
#test_originFeat_root = './data_20200510/test_originFeat2'
#test_root = './dex647_lyso_RFPer_int2s002_crop'
#test_origin_root = './dex647_lyso_RFPer_int2s002_crop_origin'
#test_root = './200316 L48 gfprab4 bfpsec61b mchrab7 exp100 int2s 003'
#test_origin_root = './200316 L48 gfprab4 bfpsec61b mchrab7 exp100 int2s 003_origin'

#test_root = './200428 91bfpkdel 20mchrab7 exp200ms int2s 009 good'
#test_originFeat_root = './200428 91bfpkdel 20mchrab7 exp200ms int2s 009 good_originFeat4'

#test_root = './200502  105 93gfpsec61btf lyso647 exp200ms in2s 057'
#test_originFeat_root = './200502  105 93gfpsec61btf lyso647 exp200ms in2s 057_originFeat4'

#test_root = './200424 90+92+0 exp200 int0 009good'
#test_originFeat_root = './200424 90+92+0 exp200 int0 009good_originFeat4'


train_root = './data_20200611/minLength20/train'
train_feat_root = './data_20200611/minLength20/train_feat2'
test_root = './data_20200611/minLength20/test'
test_originFeat_root = './data_20200611/minLength20/test_originFeat2'


import ipdb 
ipdb.set_trace()
#addFeat = ['deltaX','deltaY','vi','ai','dtot','dnet','rcon','msd','msdRatio','A','K','E','Df']
addFeat = ['deltaX','deltaY']
shift = 3
n = 3
train_data, train_min_state = getFeatVec(train_root,addFeat,shift,n)
getSmallTrainSet(train_feat_root, train_data, train_min_state, 50,20)
#getOriginSet(train_root,train_originFeat_root,train_data)

#train_data, train_min_state = preprocessData(train_root)
#getSmallTrainSet(train_process_root, train_data, train_min_state, 50,20)
#getOriginSet(train_root,train_origin_root,train_data)

#test_data, test_min_state = preprocessData(test_root)
#getOriginSet(test_root,test_origin_root,test_data)

test_data, test_min_state = getFeatVec(test_root,addFeat,shift,n)
getOriginSet(test_root,test_originFeat_root,test_data)


