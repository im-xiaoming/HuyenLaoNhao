import numpy as np
import math
from PIL import Image

def findEuclideanDistance(a, b):
    return np.linalg.norm(a - b)

def alignment_procedure(img, keypoints):
    
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # xác định hướng xoay
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1
    
    # tính khoảng cách
    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))
    
    angle = 0
    
    if b != 0 and c != 0:
        cos_a = (b*b + c*c - a*a) / (2*b*c)
        
        # fix numerical issue
        cos_a = np.clip(cos_a, -1.0, 1.0)
        
        angle = np.degrees(np.arccos(cos_a))
    
        if direction == -1:
            angle = 90 - angle
    
        # rotate image
        img_pil = Image.fromarray(img)
        img_rotated = img_pil.rotate(direction * angle)
        img = np.array(img_rotated)
    
    return img, angle