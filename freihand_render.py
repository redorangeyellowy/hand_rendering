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

def render(mano_param, cam_param, img, img_path):
    smplx_path = '/root/dataset/mano_v1_2/models'
    hand_type = mano_param['hand_type']
    mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
    prev_depth = None
    
    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
    root_pose = mano_pose[0].view(1,3)
    hand_pose = mano_pose[1:,:].view(1,-1)
    shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
    trans = torch.FloatTensor(mano_param['trans']).view(1,3)
    output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    mesh = output.vertices[0].numpy() * 1000 # meter to milimeter
    
    # apply camera extrinsics
    #t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3), np.array(cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3,3)
    #t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
    #mesh = np.dot(R, mesh.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    
    # save mesh to obj files
    #save_obj(mesh, mano_layer[hand_type].faces, osp.join(save_path, img_path.split('/')[-1][:-4] + '_' + hand_type + '.obj'))
    
    # mesh
    mesh = mesh / 1000 # milimeter to meter
    mesh = trimesh.Trimesh(mesh, mano_layer[hand_type].faces)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3), bg_color=[0.0, 0.0, 0.0]) # background color
    scene.add(mesh, 'mesh')
    # add camera intrinsics
    focal = np.array(cam_param['focal'], dtype=np.float32).reshape(2)
    princpt = np.array(cam_param['princpt'], dtype=np.float32).reshape(2)
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
    
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img['width'], viewport_height=img['height'], point_size=1.0)
    
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)
    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    depth = depth[:,:,None]
    valid_mask = (depth > 0)
    img_ori = cv2.imread(img_path)
    '''
    if prev_depth is None:
        render_mask = valid_mask
        img_ren = rgb * render_mask + img_ren * (1 - render_mask)
        prev_depth = depth
    else:
        render_mask = valid_mask * np.logical_or(depth < prev_depth, prev_depth==0)
        img_ren = rgb * render_mask + img_ren * (1 - render_mask)
        prev_depth = depth * render_mask + prev_depth * (1 - render_mask)
    '''
    # save image
    path_list = img_path.split('/')
    new_path_folder = os.path.join('/', path_list[0], path_list[1], path_list[2], path_list[3], path_list[4], 'render')
    os.makedirs(new_path_folder, exist_ok=True)
    new_path_file = os.path.join(new_path_folder, path_list[6])
    cv2.imwrite(new_path_file, rgb)
    #cv2.imwrite('freihand/test_origin.png', img_ori)
    
    return

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
            render(mano_param, cam_param, img, img_path)

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

if __name__ == "__main__":
    load_data()
