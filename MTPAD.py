from ctypes import *
import math
import random
import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2
import os
import pickle
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res
    
if __name__ == "__main__":
    # load the MTPAD model   
    net = load_net("cfg/resnet152-yolo-spoof.cfg", "MTPAD-ResNet152/resnet152-yolo-spoof_final.weights", 0) 
    meta = load_meta("data/MT-PAD.data")
   
    
    gt_labels=[]
    pd_labels=[]
    pd_scores=[]
    missed_det=0
    textfile_label=open('label_output.txt','w')
    textfile_score=open('score_output.txt','w')
    for filename in glob.glob('processed/*.png'):
        # determine the GT label; the filename is prefixed with 'Live' or 'Spoof' for ROC calculation
        if 'Live' in filename: # Hence, the directory name must not contain "Live"
            class_label=0 
        elif 'live' in filename:
            class_label=0
        else:
            class_label=1
        gt_labels.append(class_label)
        textfile_label.write("%s\n" % class_label)

        # perform PAD detection
        r = detect(net, meta, filename)  

        if not r:
            pd_label=1
            pd_labels.append(pd_label)
            pd_scores.append(1) # spoof
            missed_det=missed_det+1
            textfile_score.write("%s\n" % 1)
            continue
      
        if r[0][0]=='Live':      
            pd_label=0
            pd_scores.append(1-r[0][1])
            textfile_score.write("%s\n" % (1-r[0][1]))
        else:
            pd_label=1
            pd_scores.append(r[0][1])
            textfile_score.write("%s\n" % r[0][1])
        pd_labels.append(pd_label)

        print filename, r[0][0], r[0][1]
  
    textfile_label.close()
    textfile_score.close()   
        
    # compute the accuracy
    #import numpy
    #unique,counts=numpy.unique(gt_labels,return_counts=True)
    #print dict(zip(unique,counts))
    print confusion_matrix(gt_labels, pd_labels)
    tn, fp, fn, tp = confusion_matrix(gt_labels, pd_labels).ravel()
    # calculate error rate
    apcer=1.0*fn/(fn+tp)
    bpcer=1.0*fp/(tn+fp)
    print apcer, bpcer

    print accuracy_score(gt_labels, pd_labels)
    print missed_det
    fpr,tpr,thresholds=metrics.roc_curve(gt_labels,pd_scores,pos_label=1)
    plt.semilogx(fpr*100,tpr*100,'r')
    plt.axvline(x=0.2, color='r', linestyle='--')
    plt.grid()
    plt.show()
    #print fpr,tpr,thresholds
    #with open('Evaluation/MT-PAD-ResNet/roc.pkl', 'w') as f: 
        #pickle.dump([fpr, tpr, pd_scores, pd_labels, gt_labels], f)    

