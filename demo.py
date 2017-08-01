#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import xml.etree.cElementTree as ET

FP = 0# there is not an object but the net detectes one
NC = 0# there is an object but the detection is wrong 
CLASSES = ('__background__',  
            'leftAtrial')  

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def vis_detections(im, image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    IoU_score = []
    bbox = []
    score = []
    for i in inds:
        bbox.append(dets[i, :4])
        score.append(dets[i, -1])
	IoU_score.append(-1.0)
        xml_path = find_xml(image_name, xml_find_path)		# find xml consistent with image_name by LYS
    	if xml_path:		# if find, caculate the IoU score. else this is a false detection by LYS
	    gt_box = get_gt_box(xml_path)
	    IoU_score[i] = float(calcIoU(bbox[i], gt_box))
        ax.add_patch(
            plt.Rectangle((bbox[i][0], bbox[i][1]),
                          bbox[i][2] - bbox[i][0],
                          bbox[i][3] - bbox[i][1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[i][0], bbox[i][1] - 2,
                '{:s} {:.3f} IoU:{:.3f}'.format(class_name, score[i], IoU_score[i]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    IoU = max(IoU_score)
    max_rec_index = IoU_score.index(IoU)
    ax.add_patch(
	plt.Rectangle((bbox[max_rec_index][0], bbox[max_rec_index][1]),
                      bbox[max_rec_index][2] - bbox[max_rec_index][0],
                      bbox[max_rec_index][3] - bbox[max_rec_index][1], fill=False,
                      edgecolor='yellow', linewidth=3.5)
            )
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f} image_name:{}').format(class_name, class_name,
                                                  thresh, image_name),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    #save result
    result_path = '/home/lys/lys/test/001grb_result/'
    plt.savefig(os.path.join(result_path, image_name.split('.')[0] + '_result.jpg'))
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    print image_name
    # Load the demo image
#    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
#    im = cv2.imread(im_file)
    #load the test image LYS
    im_file = os.path.join(image_path,image_name)
    im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, image_name, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def calcIoU(pre_box,gt_box):
    global NC
    calc_IoU = -1.0
    pre_w = pre_box[2] - pre_box[0]
    pre_h = pre_box[3] - pre_box[1]
    pre_area = pre_w * pre_h
    pre_core_x = pre_box[0] + pre_w/2.0
    pre_core_y = pre_box[1] + pre_h/2.0
    gt_w = gt_box[2] - gt_box[0]
    gt_h = gt_box[3] - gt_box[1]
    gt_area = gt_w * gt_h
    gt_core_x = gt_box[0] + gt_w/2.0
    gt_core_y = gt_box[1] + gt_h/2.0
    if(abs(pre_core_x - gt_core_x) < ((pre_w + gt_w)/2.0)) and (abs(pre_core_y - gt_core_y) < ((pre_h + gt_h)/2.0)):
	
        coin_x_min = max(pre_box[0], gt_box[0])
	coin_y_min = max(pre_box[1], gt_box[1])
	coin_x_max = min(pre_box[2], gt_box[2])
	coin_y_max = min(pre_box[3], gt_box[3])
	coincide_box = [coin_x_min, coin_y_min, coin_x_max, coin_y_max]
	#print coincide_box
	coin_w = coincide_box[2] - coincide_box[0]
	coin_h = coincide_box[3] - coincide_box[1]
	coin_area = coin_w * coin_h
        calc_IoU = coin_area/float(pre_area + gt_area - coin_area)
	#print "IoU = ", calc_IoU
    else: 
	NC = NC + 1
	print setColor.UseStyle("no coincide", fore = 'green')
	
    return calc_IoU

def find_xml(image_name, xml_find_path):
    global FP
    xml_find_name = image_name.split('.')[0] + '.xml'
    xml_list = os.listdir(xml_find_path)
    xml_path = ''
    if xml_find_name in xml_list:
        xml_path = os.path.join(xml_find_path,xml_find_name)
	#print xml_path
    else: 
	FP = FP + 1
	print setColor.UseStyle("not find the xml",fore = 'blue')
    return xml_path

def get_gt_box(xml_path):
    gt_box = np.zeros((4,1))
    tree = ET.ElementTree(file= xml_path)
    for elem in tree.iter(tag = 'xmin'):
	gt_box[0] = int(elem.text)
    for elem in tree.iter(tag = 'ymin'):
	gt_box[1] = int(elem.text)
    for elem in tree.iter(tag = 'xmax'):
	gt_box[2] = int(elem.text)
    for elem in tree.iter(tag = 'ymax'):
	gt_box[3] = int(elem.text)
    return gt_box

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

##load image by path LYS
    image_path = '/home/lys/lys/test/001grb/'
    im_names = os.listdir(image_path)  
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    print 'FP = {}'.format(FP)
    print 'NC = {}'.format(NC)
    plt.show()
