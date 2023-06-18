'''
import cv2
import numpy as np

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    # 컬러 이미지와 같은 크기의 빈 캔버스 생성
    overlay = image.copy()

    # mask 값을 이용하여 컬러를 적용한 뒤 캔버스에 합성
    output = cv2.addWeighted(overlay, 1 - alpha, color, alpha, 0)

    # mask를 이진화하여 컨투어를 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 추출된 컨투어를 캔버스에 그림
    cv2.drawContours(output, contours, -1, color, 2)

    return output

def overlay_manual(image, mask):
    img_size = image.shape[0]
    img_channel = image.shape[2]
    for x in range(img_size):
        for y in range(img_size):
            for c in range(img_channel):
                
    print(mask)
    print(image)
    
    return

# RGB 영상 로드
image = cv2.imread('/root/dataset/freihand/training/rgb/00000000.jpg')

# Segmentation mask 로드
mask = cv2.imread('/root/dataset/freihand/training/mask/00000000.jpg', cv2.IMREAD_GRAYSCALE)

# Segmentation mask를 컬러로 변환
mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# 영상과 mask를 합치기 위한 overlay 작업
#output = overlay_mask(image, mask_color)
#output = cv2.absdiff(mask_color, image)
output = overlay_manual(image, mask_color)

# 결과를 화면에 출력
cv2.imwrite('/root/dataset/freihand/test_overlay.png', output)
'''
import cv2
import os
from glob import glob
from tqdm import tqdm

root_path = sorted(glob('/root/dataset/freihand/training/mask/*'))
save_path = '/root/dataset/freihand/training/overlay'
for path in tqdm(root_path):
    num = path.split('/')[-1]
    src = cv2.imread('/root/dataset/freihand/training/rgb/' + num)
    mask = cv2.imread('/root/dataset/freihand/training/mask/' + num, cv2.IMREAD_GRAYSCALE)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    dst = cv2.bitwise_and(src, mask)
    save = os.path.join(save_path, num)
    cv2.imwrite(save, dst)
