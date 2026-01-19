import numpy as np
import cv2
import math

scale_subregion = float(3) / 4
scale_mask = float(1)/(scale_subregion*4)

def closet_distance_ggtt(gt_masks, region_mask, classes_gt_objects, class_object):
    _, _, n = gt_masks.shape 
    min_distance = float('inf')
    closest_box = None
    for k in range(n): #모든 gt 마스크 n개에 대하여 반복
        if classes_gt_objects[k] != class_object:
            continue
        gt_mask = gt_masks[:,:,k]
        x_diff = region_mask[0] - gt_mask[0]
        y_diff = region_mask[1] - gt_mask[1]
        distance = np.sqrt(x_diff**2 + y_diff**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_box = gt_mask
    return closest_box
        

def closet_distance_gt(gt_masks, region_mask, classes_gt_objects, class_object, annotation, size_mask):
    _, _, n = gt_masks.shape 
    min_distance = float('inf')
    closest_box = None
    for k in range(n): #모든 gt 마스크 n개에 대하여 반복
        if classes_gt_objects[k] != class_object:
            continue
        x_diff = ((size_mask[0])/2) - ((annotation[k][2]+annotation[k][1])/2)
        y_diff = ((size_mask[1])/2) - ((annotation[k][4]+annotation[k][3])/2)
        distance = np.sqrt(x_diff**2 + y_diff**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_box = gt_masks[:,:,k]
    return closest_box

def calculate_iou(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j)/float(i))
    return iou


def calculate_overlapping(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gt_mask)
    overlap = float(float(j)/float(i))
    return overlap


def follow_iou(gt_masks, region_mask, classes_gt_objects, class_object, last_matrix):
    results = np.zeros([np.size(array_classes_gt_objects), 1])
    for k in range(np.size(classes_gt_objects)):
        if classes_gt_objects[k] == class_object:
            gt_mask = gt_masks[:, :, k]
            iou = calculate_iou(region_mask, gt_mask)
            results[k] = iou
    index = np.argmax(results)
    new_iou = results[index]
    iou = last_matrix[index]
    return iou, new_iou, results, index

# Auto find the max bounding box in the region image
def find_max_bounding_box_and_gt(gt_masks, region_mask, classes_gt_objects, class_object, annotation):
    global got
    _, _, n = gt_masks.shape 
    max_iou = 0.0
    for k in range(n): #모든 gt 마스크 n개에 대하여 반복
        if classes_gt_objects[k] != class_object:
            continue
        gt_mask = gt_masks[:,:,k]
        iou = calculate_iou(region_mask, gt_mask)
        if max_iou < iou:
            max_iou = iou
            got = sum(annotation[k]) #k번째 주석(좌표들)을 합친게 got -> 다른 주석일 경우 합도 다를 것이다.
    return max_iou, got #하나의 객체가 여러 개 있을 때 중 iou가 가장 높은 객체와의 예측 iou뽑아냄

def find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, class_object):
    
    _, _, n = gt_masks.shape 
    max_iou = 0.0
    for k in range(n): #모든 gt 마스크 n개에 대하여 반복
        if classes_gt_objects[k] != class_object:
            continue
        gt_mask = gt_masks[:,:,k]
        iou = calculate_iou(region_mask, gt_mask)
        if max_iou < iou:
            max_iou = iou
            
    return max_iou #하나의 객체가 여러 개 있을 때 중 iou가 가장 높은 객체와의 예측 iou뽑아냄

def find_iou_list(gt_masks, region_mask, classes_gt_objects, class_object): #모든 gt iou list 뽑아온다

    _, _, n = gt_masks.shape 
    iou_list = []
    
    for k in range(n): #모든 gt 마스크 n개에 대하여 반복
        if classes_gt_objects[k] != class_object:
            continue
        
        gt_mask = gt_masks[:,:,k]
        iou = calculate_iou(region_mask, gt_mask)
        iou_list.append(iou)
        
    return iou_list
    

def get_crop_image_and_mask(original_shape, offset, region_image, size_mask, action):
    r"""crop the the image according to action
    
    Args:
        original_shape: shape of original image (H x W)
        offset: the current image's left-top coordinate base on the original image
        region_image: the image to be cropped
        size_mask: the size of region_image
        action: the action choose by agent. can be 1,2,3,4,5.
        
    Returns:
        offset: the cropped image's left-top coordinate base on original image
        region_image: the cropped image
        size_mask: the size of the cropped image
        region_mask: the masked image which mask cropped region and has same size with original image
    
    """
    
    
    region_mask = np.zeros(original_shape) # mask at original image 
    size_mask = (int(size_mask[0] * scale_subregion), int(size_mask[1] * scale_subregion)) # the size of croped image
    if action == 1:
        offset_aux = (0, 0)
    elif action == 2:
        offset_aux = (0, int(size_mask[1] * scale_mask))
        offset = (offset[0], offset[1] + int(size_mask[1] * scale_mask))
    elif action == 3:
        offset_aux = (int(size_mask[0] * scale_mask), 0)
        offset = (offset[0] + int(size_mask[0] * scale_mask), offset[1])
    elif action == 4:
        offset_aux = (int(size_mask[0] * scale_mask), 
                      int(size_mask[1] * scale_mask))
        offset = (offset[0] + int(size_mask[0] * scale_mask),
                  offset[1] + int(size_mask[1] * scale_mask))
    elif action == 5:
        offset_aux = (int(size_mask[0] * scale_mask / 2),
                      int(size_mask[0] * scale_mask / 2))
        offset = (offset[0] + int(size_mask[0] * scale_mask / 2),
                  offset[1] + int(size_mask[0] * scale_mask / 2))
    region_image = region_image[offset_aux[0]:offset_aux[0] + size_mask[0],
                   offset_aux[1]:offset_aux[1] + size_mask[1]]
    region_mask[offset[0]:offset[0] + size_mask[0], offset[1]:offset[1] + size_mask[1]] = 1
    return offset, region_image, size_mask, region_mask
