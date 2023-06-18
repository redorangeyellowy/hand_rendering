import cv2
import json
import copy
from pycocotools.coco import COCO
import os.path as osp
from tqdm import tqdm
import torch
import smplx
from glob import glob
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt

COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": "g"},
    "index": {"ids": [0, 5, 6, 7, 8], "color": "c"},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": "b"},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": "m"},
    "little": {"ids": [0, 17, 18, 19, 20], "color": "r"},
}

def load_data():
    
    data_split = 'train'
    data_path = '/root/dataset/freihand'
    #human_bbox_root_dir = osp.join('..', 'data', 'FreiHAND', 'rootnet_output', 'bbox_root_freihand_output.json')
    
    if data_split == 'train':
        db = COCO(osp.join(data_path, 'moon/freihand_train_coco.json'))
        with open(osp.join(data_path, 'moon/freihand_train_data.json')) as f:
            data = json.load(f)
        
    else:
        db = COCO(osp.join(data_path, 'moon/freihand_eval_coco.json'))
        with open(osp.join(data_path, 'moon/freihand_eval_data.json')) as f:
            data = json.load(f)
        #print("Get bounding box and root from " + human_bbox_root_dir)
        bbox_root_result = {}
        #with open(human_bbox_root_dir) as f:
        #    annot = json.load(f)
        #for i in range(len(annot)):
        #    bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']), 'root': np.array(annot[i]['root_cam'])}

    datalist = []
    for aid in tqdm(db.anns.keys()):
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]
        img_path = osp.join(data_path, img['file_name'])
        img_shape = (img['height'], img['width'])
        db_idx = str(img['db_idx'])

        if data_split == 'train':
            cam_param, mano_param = data[db_idx]['cam_param'], data[db_idx]['mano_param']
            mano_param['hand_type'] = 'right' # FreiHAND only contains right hand
            #bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            #if bbox is None: continue
            #render(mano_param, cam_param, img, img_path)
            joint_3d = data[db_idx]['joint_3d']
            joint_3d = np.array(joint_3d)
            joint_2d = joint_3d[:, :2]
            

            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                #'bbox': bbox,
                'cam_param': cam_param,
                'mano_param': mano_param})
        else:
            cam_param = data[db_idx]['cam_param']
            #bbox = bbox_root_result[str(image_id)]['bbox'] # bbox should be aspect ratio preserved-extended. It is done in RootNet.
            root_joint_depth = bbox_root_result[str(image_id)]['root'][2]

            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                #'bbox': bbox,
                'root_depth': root_joint_depth,
                'cam_param': cam_param})

def projectPoints(xyz, K):
    """
    Projects 3D coordinates into image space.
    Function taken from https://github.com/lmb-freiburg/freihand
    """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def load_data2():
    
    data_split = 'train'
    data_path = '/root/dataset/freihand'
    xyz_path = 'training_xyz.json'
    k_path = 'training_K.json'
    #human_bbox_root_dir = osp.join('..', 'data', 'FreiHAND', 'rootnet_output', 'bbox_root_freihand_output.json')
    
    if data_split == 'train':
        with open(osp.join(data_path, xyz_path)) as f:
            xyz = json.load(f)
        with open(osp.join(data_path, k_path)) as f:
            k = json.load(f)
    
    for idx in tqdm(range(len(xyz))):
        keypoints = projectPoints(xyz[idx], k[idx])
        #plt.figure(figsize=(2.24, 2.24), dpi=100)
        empty_image = np.zeros([224, 224, 3])
        plt.imshow(empty_image)
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c="k", alpha=1)
        for finger, params in COLORMAP.items():
            plt.plot(
                keypoints[params["ids"], 0],
                keypoints[params["ids"], 1],
                params["color"],
            )
        plt.axis('off')
        save_folder = '/root/dataset/freihand/training/pose'
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, str(idx).zfill(8) + '.jpg')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=60.7)
        plt.close()
        #plt.savefig('test_2dpose.png',  pad_inches=0)
    
        #fig = plt.gcf()
        #fig.canvas.draw()
        #image_array = np.array(fig.canvas.renderer.buffer_rgba())
        #image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        #cv2.imwrite('test_2dpose.png', image_cv2)
        
    
if __name__ == "__main__":
    #load_data()
    load_data2()
