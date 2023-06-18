import os
import os.path as osp
import numpy as np
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import smplx
os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch
import pyrender
import trimesh
import cv2
# from pytorch3d.structures import Meshes
# from pytorch3d.renderer import (
# PointLights,
# PerspectiveCameras,
# OrthographicCameras,
# Materials,
# SoftPhongShader,
# RasterizationSettings,
# MeshRendererWithFragments,
# MeshRasterizer,
# TexturesVertex)
from config import cfg

smplx_path = '/root/dataset/mano_v1_2/models'
mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}

# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1

def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        bbox = None

    return bbox

def process_bbox(bbox, img_width, img_height, do_sanitize=True, extend_ratio=1.25):
    if do_sanitize:
        bbox = sanitize_bbox(bbox, img_width, img_height)
        if bbox is None:
            return bbox

   # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*extend_ratio
    bbox[3] = h*extend_ratio
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def render_mesh_orthogonal(mesh, face, cam_param, render_shape, hand_type):
    batch_size, vertex_num = mesh.shape[:2]

    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)
    
    cameras = OrthographicCameras(focal_length=cam_param['focal'], 
                                principal_point=cam_param['princpt'], 
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1)#, perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    if hand_type == 'right':
        color = ((1.0, 0.0, 0.0),)
    else:
        color = ((0.0, 1.0, 0.0),)
    materials = Materials(
	device='cuda',
        ambient_color=((0.5,0.5,0.5),),
        diffuse_color=((1.0,1.0,1.0),),
        specular_color=color,
	shininess=0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
        images = images[:,:,:,:3] * 255
        depthmaps = fragments.zbuf

    return images, depthmaps

def render_mesh_perspective(mesh, face, cam_param, render_shape, hand_type):
    batch_size, vertex_num = mesh.shape[:2]

    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)
    
    cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                principal_point=cam_param['princpt'], 
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1)#, perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    if hand_type == 'right':
        color = ((1.0, 0.0, 0.0),)
    else:
        color = ((0.0, 1.0, 0.0),)
    materials = Materials(
	device='cuda',
        ambient_color=((0.5,0.5,0.5),),
        diffuse_color=((1.0,1.0,1.0),),
        specular_color=color,
	shininess=0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
        images = images[:,:,:,:3] * 255
        depthmaps = fragments.zbuf

    return images, depthmaps

def render_mesh_origin(mesh, face, cam_param, img):
    mesh = mesh / 1000 # milimeter to meter
    mesh = trimesh.Trimesh(mesh, face)
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
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
    
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
    return rgb

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
    # path_list = img_path.split('/')
    # new_path_folder = os.path.join('/', path_list[0], path_list[1], path_list[2], path_list[3], path_list[4], 'render')
    # os.makedirs(new_path_folder, exist_ok=True)
    # new_path_file = os.path.join(new_path_folder, path_list[6])
    # cv2.imwrite(new_path_file, rgb)

def mesh_transformation(param, hand_type):
    mano_param = param[hand_type]['mano_param']
    cam_param = param[hand_type]['cam_param']
    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
    root_pose = mano_pose[0].view(1,3)
    hand_pose = mano_pose[1:,:].view(1,-1)
    shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
    trans = torch.FloatTensor(mano_param['trans']).view(1,3)
    output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    mesh = output.vertices[0].numpy() * 1000 # meter to milimeter
    
    # apply camera extrinsics
    #t, R = np.array(cam_param['campos'], dtype=np.float32).reshape(3), np.array(cam_param['camrot'], dtype=np.float32).reshape(3,3)
    #t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
    #mesh = np.dot(R, mesh.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    
    # render
    # mesh = torch.from_numpy(mesh).float().cuda()[None,:,:]
    # face = torch.from_numpy(mano_layer[hand_type].faces.astype(np.int32)).cuda()[None,:,:]
    # focal = torch.FloatTensor(cam_param['focal']).cuda()[None,:]
    # princpt = torch.FloatTensor(cam_param['princpt']).cuda()[None,:]

    face = mano_layer[hand_type].faces.astype(np.int32)
    focal = cam_param['focal']
    princpt = cam_param['princpt']

    return mesh, face, focal, princpt

def load_data():
    root_path = '/root/dataset/mscoco/images'
    annot_path = '/root/dataset/mscoco/annotations'
    joint_set = {
                'joint_num': 42,
                'joints_name': ('L_Wrist', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', 'R_Wrist', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4'),
                'flip_pairs': [ (i,i+21) for i in range(21)],
                }
    joint_set['joint_type'] = {'left': np.arange(0,joint_set['joint_num']//2), 'right': np.arange(joint_set['joint_num']//2,joint_set['joint_num'])}
    joint_set['root_joint_idx'] = {'left': joint_set['joints_name'].index('L_Wrist'), 'right': joint_set['joints_name'].index('R_Wrist')}
    
    data_split = 'train'
    if data_split == 'train':
        with open(osp.join(annot_path, 'MSCOCO_train_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)
        db = COCO(osp.join(annot_path, 'coco_wholebody_train_v1.0.json'))
    else:
        db = COCO(osp.join(annot_path, 'coco_wholebody_val_v1.0.json'))

    for aid in tqdm(db.anns.keys()):
        ann = db.anns[aid]
        img = db.loadImgs(ann['image_id'])[0]
        if data_split == 'train':
            imgname = osp.join('train2017', img['file_name'])
        else:
            imgname = osp.join('val2017', img['file_name'])
        img_path = osp.join(root_path, imgname)
        img_origin = cv2.imread(img_path)

        if ann['iscrowd'] or (ann['num_keypoints'] == 0):
            continue
        if ann['lefthand_valid'] is False and ann['righthand_valid'] is False:
            continue
        
        # body bbox
        body_bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
        if body_bbox is None: continue
        
        # left hand bbox
        if ann['lefthand_valid'] is False:
            lhand_bbox = None
        else:
            lhand_bbox = np.array(ann['lefthand_box'], dtype=np.float32)
            lhand_bbox = sanitize_bbox(lhand_bbox, img['width'], img['height'])
        if lhand_bbox is not None:
            lhand_bbox[2:] += lhand_bbox[:2] # xywh -> xyxy

        # right hand bbox
        if ann['righthand_valid'] is False:
            rhand_bbox = None
        else:
            rhand_bbox = np.array(ann['righthand_box'], dtype=np.float32)
            rhand_bbox = sanitize_bbox(rhand_bbox, img['width'], img['height'])
        if rhand_bbox is not None:
            rhand_bbox[2:] += rhand_bbox[:2] # xywh -> xyxy

        joint_img = np.concatenate((
                            np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1,3),
                            np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1,3)))
        joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)
        joint_img[:,2] = 0
        
        if data_split == 'train' and str(aid) in mano_params:
            mano_param = mano_params[str(aid)]
        else:
            mano_param = {'right': None, 'left': None}
        
        # render
        if ann['lefthand_valid'] and not ann['righthand_valid']:
            hand_types = ['left']
        elif ann['righthand_valid'] and not ann['lefthand_valid']:
            hand_types = ['right']
        elif ann['righthand_valid'] and ann['lefthand_valid']:
            hand_types = ['right', 'left']
        if len(hand_types) == 2:
            mesh, face, focal, princpt = mesh_transformation(mano_param, hand_types[0])
            img_render_right = render_mesh_origin(mesh, face, {'focal': focal, 'princpt': princpt}, img_origin)
            mesh, face, focal, princpt = mesh_transformation(mano_param, hand_types[1])
            img_render_left = render_mesh_origin(mesh, face, {'focal': focal, 'princpt': princpt}, img_origin)
            img_render = img_render_right + img_render_left
        else:
            mesh, face, focal, princpt = mesh_transformation(mano_param, hand_types[0])
            img_render = render_mesh_origin(mesh, face, {'focal': focal, 'princpt': princpt}, img_origin)
        save_folder = '/root/dataset/mscoco/images_render/train2017'
        save_num = img_path.split('/')[-1]
        save_file = os.path.join(save_folder, save_num)
        cv2.imwrite(save_file, img_render)
        #cv2.imwrite('mscoco/test_render.png', img_render)
        #cv2.imwrite('mscoco/test_origin.png', img_origin)
    print('rendering finish!')
    



if __name__ == "__main__":
    load_data()
