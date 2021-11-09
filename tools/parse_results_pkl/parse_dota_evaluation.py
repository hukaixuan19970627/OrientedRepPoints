# -*- coding: utf-8 -*-
import pickle
import cv2
import json
import os 
import shutil
import numpy as np 
import argparse

parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
parser.add_argument('--detection_pkl_path', default='', help='!')
parser.add_argument('--val_json', default='', help='!')
parser.add_argument('--outpath', default='', help='Task1_classname.txt where to save')
args = parser.parse_args()

detection_pkl_path = args.detection_pkl_path
val_json = args.val_json
outpath = args.outpath

if os.path.exists(outpath):
    shutil.rmtree(outpath)  # delete output folderX
    
os.makedirs(outpath)

conf_thresh = 0.05
def transfer_to_order_point(corner_points):
    a_x = corner_points[0][0]
    a_y = corner_points[0][1]

    b_x = corner_points[1][0]
    b_y = corner_points[1][1]

    c_x = corner_points[2][0]
    c_y = corner_points[2][1]

    d_x = corner_points[3][0]
    d_y = corner_points[3][1]

    # top x0 y0
    if  a_y == min(a_y, b_y, c_y, d_y) :
        x0 = a_x
        y0 = a_y
    if  b_y == min(a_y, b_y, c_y, d_y) :
        x0 = b_x
        y0 = b_y
    if  c_y == min(a_y, b_y, c_y, d_y) :
        x0 = c_x
        y0 = c_y
    if  d_y == min(a_y, b_y, c_y, d_y) :
        x0 = d_x
        y0 = d_y

    # right x1 y1
    if  a_x == max(a_x, b_x, c_x, d_x) :
        x1 = a_x
        y1 = a_y
    if  b_x == max(a_x, b_x, c_x, d_x) :
        x1 = b_x
        y1 = b_y
    if  c_x == max(a_x, b_x, c_x, d_x) :
        x1 = c_x
        y1 = c_y
    if  d_x == max(a_x, b_x, c_x, d_x) :
        x1 = d_x
        y1 = d_y

    # bottom x2 y2
    if  a_y == max(a_y, b_y, c_y, d_y) :
        x2 = a_x
        y2 = a_y
    if  b_y == max(a_y, b_y, c_y, d_y) :
        x2 = b_x
        y2 = b_y
    if  c_y == max(a_y, b_y, c_y, d_y) :
        x2 = c_x
        y2 = c_y
    if  d_y == max(a_y, b_y, c_y, d_y) :
        x2 = d_x
        y2 = d_y

    # left x3 y3
    if  a_x == min(a_x, b_x, c_x, d_x) :
        x3 = a_x
        y3 = a_y
    if  b_x == min(a_x, b_x, c_x, d_x) :
        x3 = b_x
        y3 = b_y
    if  c_x == min(a_x, b_x, c_x, d_x) :
        x3 = c_x
        y3 = c_y
    if  d_x == min(a_x, b_x, c_x, d_x) :
        x3 = d_x
        y3 = d_y

    order_points = np.array([(x0,y0),(x1,y1),(x2,y2),(x3,y3)], np.int32)
    return order_points


#open result pkl
with open(detection_pkl_path, 'rb') as file:
    while True:
        try:
            data = pickle.load(file)  # list[num_imgs/batch_size * list[batch_size * list[num_classes * array(n, [18, 8, score])]][0] ]
        except EOFError:
            break

num_img = len(data)

#open json 
with open(val_json) as f_json:
    ann=json.load(f_json)

    for img_item in ann['images']:  # 遍历每张图的检测结果
        index_img = img_item['id'] - 1
        img_results = data[index_img]

        img_name = img_item['file_name'] # 'P0003__1.0__0___0.png'
        #img_base_name = img_name.split('.png')[0]   # 'P0003__1.0__0___0'
        bboxes = img_results  # bboxes： list[num_classes * array(n, [18, 8, score])]
        num_bboxes = len(bboxes)

        for iter_box in range(num_bboxes): # 遍历每个类的检测结果
            if len(bboxes[iter_box])>0:
                if iter_box == 0:
                    class_name = 'plane'
                elif iter_box == 1:
                    class_name = 'baseball-diamond'
                elif iter_box == 2:
                    class_name = 'bridge'
                elif iter_box == 3:
                    class_name = 'ground-track-field' 
                elif iter_box == 4:
                    class_name = 'small-vehicle'
                elif iter_box == 5:
                    class_name = 'large-vehicle'
                elif iter_box == 6:
                    class_name = 'ship'
                elif iter_box == 7:
                    class_name = 'tennis-court'
                elif iter_box == 8:
                    class_name = 'basketball-court'
                elif iter_box == 9:
                    class_name = 'storage-tank'
                elif iter_box == 10:
                    class_name = 'soccer-ball-field'
                elif iter_box == 11:
                    class_name = 'roundabout' 
                elif iter_box == 12:
                    class_name = 'harbor'
                elif iter_box == 13:
                    class_name = 'swimming-pool'
                elif iter_box == 14:
                    class_name = 'helicopter'  
                elif iter_box == 15:
                    class_name = 'container-crane'  
                elif iter_box == 16:
                    class_name = 'airport'  
                elif iter_box == 17:
                    class_name = 'helipad'  
                else:
                    class_name = None
                    print('Unknown class_name!!!!!!!!!!!!!!!!!!!!!!! ')
                
                class_dstname = os.path.join(outpath, 'Task1_' + class_name + '.txt')  # eg: .../Task1_plane.txt
                with open(class_dstname, 'a') as f:
                    for bbox in bboxes[iter_box]:  # bboxes[iter_box]：  array(n, [18, 8, score]) x_first                      
                        confidence = float(bbox[-1])
                        #confidence threshold
                        if confidence > conf_thresh:    
                            confidence = '%.2f' % confidence                       
                            # visulize the oriented boxes     
                            box_list = []
                            box_list.append((float(bbox[-9]), float(bbox[-8])))
                            box_list.append((float(bbox[-7]), float(bbox[-6])))
                            box_list.append((float(bbox[-5]), float(bbox[-4])))
                            box_list.append((float(bbox[-3]), float(bbox[-2])))  # box_list.size(4, 2)
                            box_order = transfer_to_order_point(box_list)  # 给四个点排序
                            box_order = box_order.reshape(-1)  # box_order.size(8)
                            
                            lines = img_name + ' ' + confidence + ' ' + ' '.join(list(map(str, box_order)))     # 目标所属原始图片名称 置信度 poly
                            f.writelines(lines + '\n')
            #print(f'{class_dstname} has been writen! ')
                        
print('Done !')
                                