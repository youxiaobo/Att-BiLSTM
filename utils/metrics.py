# coding:utf8
import torch as t
import numpy as np

def getFrameAcc(label_num,pred_num):
    frame_acc = pred_num.sum()/label_num.sum()
    return frame_acc

def getTemporalSeg(label_seq):
    length = label_seq.shape[0]
    temporal_seg = t.zeros(length,3)
    # initial label and start time: label,start,end
    temporal_seg[0] = t.tensor([label_seq[0],0,0])
    k = 1
    for i in range(1,length):
        if label_seq[i] != label_seq[i-1]:
            # record last end time
            temporal_seg[k-1,2] = i-1
            # record new label
            temporal_seg[k,0] = label_seq[i]
            # record new start time
            temporal_seg[k,1] = i
            k = k+1
        if i == length-1:
           temporal_seg[k-1,2] = i
    # remove extra row
    temporal_seg = temporal_seg[:k,:]
    return temporal_seg
            
def getTrackletPredNum(label_seq,pred_seq):
    label_seg = getTemporalSeg(label_seq)
    pred_seg = getTemporalSeg(pred_seq)
    tracklet_num = label_seg.shape[0]
    trackletPred_num = 0
    for i in range(tracklet_num):
        for j in range(pred_seg.shape[0]):
            # state,start time,end time all equal
            if label_seg[i].equal(pred_seg[j]):
                trackletPred_num +=1
                break
    return tracklet_num,trackletPred_num
 
           
def getTrackletAcc(tracklet_num,trackletPred_num):
    tracklet_acc = trackletPred_num.sum()/tracklet_num.sum()
    return tracklet_acc

def findTP(labelClass_seg,predClass_seg,iou_threshold):
    TP_count = 0
    for i in range(labelClass_seg.shape[0]):
        for j in range(predClass_seg.shape[0]):
            union = t.max(labelClass_seg[i,2],predClass_seg[j,2])-t.min(labelClass_seg[i,1],predClass_seg[j,1])+1

            intersection  = t.min(labelClass_seg[i,2],predClass_seg[j,2])-t.max(labelClass_seg[i,1],predClass_seg[j,1])+1
            if intersection <=0:
                continue
            iou = intersection / union
            if iou >= iou_threshold:
                TP_count += 1
                # if find true positive,then delete this record,ensure every tracklet has only one true positive prediction at most
                if predClass_seg.shape[0] > 1:
                   if j == 0:
                       predClass_seg = predClass_seg[j+1:,:]
                   elif j == predClass_seg.shape[0]-1:
                       predClass_seg = predClass_seg[:j,:]
                   else:
                       predClass_seg = t.cat((predClass_seg[:j,:],predClass_seg[j+1:,:]),0)
                break
    return TP_count

def getPrecision(label_seq,pred_seq,class_num,iou_threshold):
   
    label_seg = getTemporalSeg(label_seq)
    pred_seg = getTemporalSeg(pred_seq)
    labelClass_seg = []
    predClass_seg = []
    # devide label_seg and pred_seg into different class
    for i in range(class_num):
       # groundtruth does not have this class
       if len((label_seg[:,0]==i).nonzero())==0:
          labelClass_seg.append(t.tensor([float('nan')]))
       else:
          labelClass_seg.append(label_seg[(label_seg[:,0]==i).nonzero(),:].squeeze(-2))

       # prediction does not have this class
       if len((pred_seg[:,0]==i).nonzero())==0:
          predClass_seg.append(t.tensor([float('nan')]))
       else:
          predClass_seg.append(pred_seg[(pred_seg[:,0]==i).nonzero(),:].squeeze(-2))

    precision_class = t.zeros([1,class_num])
    # calculate precision for every class
    for i in range(class_num):
       # set precision to nan if groundtruth doesn't have this class 
       if t.isnan(labelClass_seg[i]).any():
           precision_class[0,i] = t.tensor([float('nan')])
       # set precision to 0 if prediction doesn't have this class
       elif t.isnan(predClass_seg[i]).any():
           precision_class[0,i] = 0.0
       # true positive num / all tracklet num in this class
       else: 
           precision_class[0,i] = findTP(labelClass_seg[i],predClass_seg[i],iou_threshold)/labelClass_seg[i].shape[0]

    return precision_class

def getmAP(precision_class):
    class_num = precision_class.shape[1]
    ap = t.zeros([1,class_num])
    for i in range(class_num):
        ap[0,i] = precision_class[~t.isnan(precision_class[:,i]),i].mean()
    
    mAP = ap.mean()
    return mAP
