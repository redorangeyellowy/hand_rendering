import cv2
import os
from glob import glob
from tqdm import tqdm

root_path = '/root/dataset/freihand/training/render'
path_list = sorted(glob(os.path.join(root_path, '*')))
path_list = path_list[:int(len(path_list)/4)]

for path in tqdm(path_list):
    src = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    folder_path = '/root/dataset/freihand/training/edge'
    num = path.split('/')[-1]
    file_path = os.path.join(folder_path, num)
    cv2.imwrite(file_path, laplacian)
    