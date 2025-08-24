# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
# Copyright 2025 The CSRef Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for bounding box format conversions in the CSRef project.

This module provides essential coordinate transformation functions for converting between
different bounding box formats used in object detection tasks. These utilities are critical
components of the CSRef data processing pipeline, enabling seamless integration between
COCO format annotations (used in many datasets) and YOLO format (used by the detection model).

The main functions in this module handle:
1. COCO to YOLO format conversion (label2yolobox)
2. YOLO to COCO format conversion (yolobox2label)

These transformations account for image resizing, padding, and augmentation operations,
ensuring accurate coordinate mapping between original image space and normalized model input space.
"""

import numpy as np

"""
Coordinate System Conversions:

This module handles transformations between two major bounding box formats:

1. COCO Format:
   - Representation: [x, y, w, h] or [x1, y1, x2, y2]
   - Coordinate System: Absolute pixel values
   - Reference Point: Top-left corner (x, y) or corners (x1, y1, x2, y2)
   - Value Range: [0, image_width] for x-coordinates, [0, image_height] for y-coordinates
   - Usage: Common in datasets and annotations

2. YOLO Format:
   - Representation: [xc, yc, w, h]
   - Coordinate System: Normalized values [0, 1]
   - Reference Point: Center point (xc, yc)
   - Value Range: [0, 1] for all coordinates relative to image dimensions
   - Usage: Required by YOLO-based object detection models

Transformation Process:
- COCO → YOLO: Convert absolute coordinates to normalized center-based coordinates
- YOLO → COCO: Convert normalized center-based coordinates back to absolute coordinates

The transformations account for:
- Image resizing (original → resized dimensions)
- Padding (to maintain aspect ratio)
- Data augmentation (e.g., horizontal flipping)
- Normalization to ensure values are in valid ranges
"""


def label2yolobox(labels, info_img, maxsize, lrflip=False):
    """
    Transform COCO format bounding box labels to YOLO format normalized coordinates.
    
    This function converts bounding boxes from COCO format [x, y, w, h] (absolute coordinates
    with top-left as origin) to YOLO format [xc, yc, w, h] (normalized coordinates with center
    as origin). The transformation accounts for image resizing, padding, and optional horizontal
    flipping during preprocessing.
    
    Coordinate Systems:
    - COCO format: [x, y, w, h] where (x,y) is top-left corner, values in absolute pixels
    - YOLO format: [xc, yc, w, h] where (xc,yc) is center point, values normalized to [0,1]
    
    Args:
        labels (numpy.ndarray): Label data with shape :math:`(N, 5)` where N is number of boxes.
            Each label consists of [x, y, w, h, score] where:
                x, y (float): Coordinates of the top-left corner of the bounding box in pixels.
                w, h (float): Width and height of the bounding box in pixels.
                ...: Additional values (e.g., class) are preserved unchanged.
        info_img (tuple): Image processing information containing:
            h, w (int): Original height and width of the image.
            nh, nw (int): Height and width of the resized image without padding.
            dx, dy (int): Padding size added to the width and height dimensions.
            (Optionally contains additional values which are ignored)
        maxsize (int): Target image size after preprocessing (both dimensions).
        lrflip (bool, optional): Flag indicating whether horizontal flip was applied.
            If True, the x-coordinates will be flipped. Defaults to False.

    Returns:
        numpy.ndarray: Transformed label data with shape :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where:
                xc, yc (float): Normalized center coordinates of the bounding box.
                    Values range from 0 to 1 relative to maxsize.
                w, h (float): Normalized width and height of the bounding box.
                    Values range from 0 to 1 relative to maxsize.
                ...: Additional values (e.g., class) are preserved unchanged.

    """
    # Extract image dimensions and preprocessing information
    if len(info_img) == 8:
        h, w, nh, nw, dx, dy, _, _ = info_img
    else:
        h, w, nh, nw, dx, dy, _ = info_img
    
    # Convert COCO format [x, y, w, h] to normalized coordinates [0,1]
    # x1, y1: top-left corner normalized by original image dimensions
    x1 = labels[:, 0] / w
    y1 = labels[:, 1] / h
    # x2, y2: bottom-right corner normalized by original image dimensions
    x2 = (labels[:, 0] + labels[:, 2]) / w
    y2 = (labels[:, 1] + labels[:, 3]) / h
    
    # Transform to YOLO format [xc, yc, w, h] normalized to maxsize
    # Calculate center coordinates and apply resizing and padding
    labels[:, 0] = (((x1 + x2) / 2) * nw + dx) / maxsize  # xc: normalized center x
    labels[:, 1] = (((y1 + y2) / 2) * nh + dy) / maxsize  # yc: normalized center y
    
    # Scale width and height according to resize ratio and normalize to maxsize
    labels[:, 2] *= nw / w / maxsize  # w: normalized width
    labels[:, 3] *= nh / h / maxsize  # h: normalized height
    
    # Clip coordinates to ensure they're within valid range [0, 0.99]
    labels[:, :4] = np.clip(labels[:, :4], 0., 0.99)
    
    # Apply horizontal flip if needed (mirror x-coordinate)
    if lrflip:
        labels[:, 0] = 1 - labels[:, 0]
    
    return labels


def yolobox2label(box, info_img):
    """
    Transform YOLO format normalized bounding box labels back to COCO format absolute coordinates.
    
    This function converts bounding boxes from YOLO format [xc, yc, w, h] (normalized coordinates
    with center as origin) back to COCO format [x1, y1, x2, y2] (absolute coordinates with top-left
    and bottom-right corners). This is the inverse operation of label2yolobox, accounting for image
    resizing and padding that was applied during preprocessing.
    
    Coordinate Systems:
    - YOLO format: [xc, yc, w, h] where (xc,yc) is center point, values normalized to [0,1]
    - COCO format: [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right corner
    
    Args:
        box (numpy.ndarray): Box data with shape :math:`(N, 5+)` where N is number of boxes.
            Each box contains [xc, yc, w, h, ...] where:
                xc, yc (float): Normalized center coordinates of the bounding box.
                    Values range from 0 to 1 relative to maxsize.
                w, h (float): Normalized width and height of the bounding box.
                    Values range from 0 to 1 relative to maxsize.
                ...: Additional values (e.g., class) are preserved unchanged.
        info_img (tuple): Image processing information containing:
            h, w (int): Original height and width of the image.
            nh, nw (int): Height and width of the resized image without padding.
            dx, dy (int): Padding size added to the width and height dimensions.
            (Optionally contains additional values which are ignored)

    Returns:
        numpy.ndarray: Transformed box data with shape :math:`(N, 5+)`.
            Each box contains [x1, y1, x2, y2, ...] where:
                x1, y1 (float): Coordinates of the top-left corner in original image pixels.
                x2, y2 (float): Coordinates of the bottom-right corner in original image pixels.
                ...: Additional values from the input box (e.g., class) preserved unchanged.
    """
    # Extract image dimensions and preprocessing information
    if len(info_img) == 8:
        h, w, nh, nw, dx, dy, _, _ = info_img
    else:
        h, w, nh, nw, dx, dy, _ = info_img
    
    # Extract normalized YOLO format coordinates [xc, yc, w, h]
    x1, y1, x2, y2 = box[:4]  # Note: These are actually [xc, yc, w, h] in normalized form
    
    # Convert normalized dimensions back to absolute pixel values in original image
    # Calculate height in original image: normalized height * (original height / resized height)
    box_h = ((y2 - y1) / nh) * h
    # Calculate width in original image: normalized width * (original width / resized width)
    box_w = ((x2 - x1) / nw) * w
    
    # Convert normalized center coordinates back to top-left coordinates in original image
    # First remove padding, then scale to resized dimensions, then scale to original dimensions
    y1 = ((y1 - dy) / nh) * h  # Convert yc to y1 in original image coordinates
    x1 = ((x1 - dx) / nw) * w  # Convert xc to x1 in original image coordinates
    
    # Create COCO format [x1, y1, x2, y2] where x2 = x1 + width, y2 = y1 + height
    label = [x1, y1, x1 + box_w, y1 + box_h]
    
    # Preserve any additional values (e.g., class) from the original box
    return np.concatenate([np.array(label), box[4:]])
