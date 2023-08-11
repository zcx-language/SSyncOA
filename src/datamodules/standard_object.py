#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : standard_object.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/7/11 17:19

# Import lib here
import numpy as np
import cv2
from PIL import Image
from pathlib import Path


def standard_object(image: np.ndarray, mask: np.ndarray):
    """Standardize the image and mask by moments,
        1. Get the centroid of the object by moments;
        2. Pad the image and mask so as the object centroid is in the center of the image;
        3. Calculate the principle axis of the object and rotate the image and mask;
        4. Get the bounding box of the object;
        5. Pad the image and mask so as a square is fitted to the bounding box;
        6. Crop the center square from the image
        7. Resize the image and mask to 256*256
    Args:
        image:
        mask:

    Returns:
    """

    img = image

    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    moments = cv2.moments(contours[0])

    # Calculate the center of the object
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    # Pad so that the object is in the center of the image for rotation
    right_pad, left_pad, top_pad, bottom_pad = 0, 0, 0, 0
    if cx > img.shape[1] - cx:
        right_pad = (cx - img.shape[1] + cx)
    else:
        left_pad = (img.shape[1] - cx - cx)
    if cy > img.shape[0] - cy:
        bottom_pad = (cy - img.shape[0] + cy)
    else:
        top_pad = (img.shape[0] - cy - cy)

    img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    mask = np.pad(mask, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')

    # Calculate the angle of the object
    theta = np.degrees(np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02']) / 2)

    # Rotate the image and mask
    rotated_img = np.array(Image.fromarray(img).rotate(theta, resample=2, expand=True))
    rotated_mask = np.array(Image.fromarray(mask).rotate(theta, resample=2, expand=True))

    # Check if the object is in the center
    # contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    # moments = cv2.moments(contours[0])
    # cx = int(moments['m10'] / moments['m00'])
    # cy = int(moments['m01'] / moments['m00'])
    # cv2.drawMarker(rotated_img, (cx, cy), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
    # cv2.drawMarker(rotated_mask, (cx, cy), color=0, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
    # cv2.circle(rotated_img, (rotated_img.shape[1]//2, rotated_img.shape[0]//2), 10, 0, 1)
    # cv2.circle(rotated_mask, (rotated_mask.shape[1]//2, rotated_mask.shape[0]//2), 10, 0, 1)
    # plt.subplot(1, 2, 1)
    # plt.imshow(rotated_img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(rotated_mask, cmap='gray')
    # plt.show()
    # plt.close()
    # input()

    # Get the centroid of the object
    contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    moments = cv2.moments(contours[0])
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    # Get the max distance from the centroid to the boundary
    x, y, w, h = cv2.boundingRect(contours[0])
    left, right, top, bottom = x, x + w, y, y + h
    max_dist = max(abs(cx - left), abs(cx - right), abs(cy - top), abs(cy - bottom))

    top_pad, left_pad, bottom_pad, right_pad = 0, 0, 0, 0
    if cy < max_dist:
        top_pad = max_dist - cy
    if cx < max_dist:
        left_pad = max_dist - cx
    if rotated_img.shape[0] - cy < max_dist:
        bottom_pad = max_dist - (rotated_img.shape[0] - cy)
    if rotated_img.shape[1] - cx < max_dist:
        right_pad = max_dist - (rotated_img.shape[1] - cx)
    pad_img = np.pad(rotated_img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    pad_mask = np.pad(rotated_mask, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')

    # plt.subplot(1, 2, 1)
    # plt.imshow(pad_img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(pad_mask, cmap='gray')
    # plt.show()
    # plt.close()

    crop_img = pad_img[cy + top_pad - max_dist:cy + top_pad + max_dist,
               cx + left_pad - max_dist:cx + left_pad + max_dist]
    crop_mask = pad_mask[cy + top_pad - max_dist:cy + top_pad + max_dist,
                cx + left_pad - max_dist:cx + left_pad + max_dist]

    std_img = cv2.resize(crop_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    std_mask = cv2.resize(crop_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
    std_mask = cv2.threshold(std_mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Crop the image and mask
    # cropped_img = rotated_img[y:y+h, x:x+w]
    # cropped_mask = rotated_mask[y:y+h, x:x+w]

    # Rescale the image and mask so that the longer side is 256
    # longer_side = max(cropped_img.shape[:2])
    # scale = 256 / longer_side
    # cropped_img = cv2.resize(cropped_img, (round(cropped_img.shape[1]*scale), round(cropped_img.shape[0]*scale)))
    # cropped_mask = cv2.resize(cropped_mask, (round(cropped_mask.shape[1]*scale), round(cropped_mask.shape[0]*scale)))

    # Pad the image and mask so that it is a square
    # height, width = cropped_img.shape[:2]
    # if height > width and height == 256:
    #     pad = (256 - width) // 2
    #     cropped_img = np.pad(cropped_img, ((0, 0), (pad, 256-width-pad), (0, 0)), mode='constant')
    #     cropped_mask = np.pad(cropped_mask, ((0, 0), (pad, 256-width-pad)), mode='constant')
    # elif width > height and width == 256:
    #     pad = (256 - height) // 2
    #     cropped_img = np.pad(cropped_img, ((pad, 256-height-pad), (0, 0), (0, 0)), mode='constant')
    #     cropped_mask = np.pad(cropped_mask, ((pad, 256-height-pad), (0, 0)), mode='constant')
    # else:
    #     raise ValueError('The image is not square after rescaling')

    # Pad the image and mask so that it is a square
    # height, width = rotated_img.shape[:2]
    # if height > width:
    #     pad = (height - width) // 2
    #     pad_img = np.pad(rotated_img, ((0, 0), (pad, height-width-pad), (0, 0)), mode='constant')
    #     pad_mask = np.pad(rotated_mask, ((0, 0), (pad, height-width-pad)), mode='constant')
    # else:
    #     pad = (width - height) // 2
    #     pad_img = np.pad(rotated_img, ((pad, width-height-pad), (0, 0), (0, 0)), mode='constant')
    #     pad_mask = np.pad(rotated_mask, ((pad, width-height-pad), (0, 0)), mode='constant')

    # Rescale the image and mask to 256
    # std_img = cv2.resize(pad_img, (256, 256))
    # std_mask = cv2.resize(pad_mask, (256, 256))

    # plt.subplot(1, 2, 1)
    # plt.imshow(crop_img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(crop_mask, cmap='gray')
    # plt.show()
    # plt.close()
    # input()
    return std_img, std_mask


def run():
    pass


if __name__ == '__main__':
    run()
