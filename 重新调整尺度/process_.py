import mmcv
import numpy as np
from large_image import merge_results_by_nms
from mmdet.apis import inference_detector
import cv2
from sahi.slicing import slice_image
from tqdm import tqdm

from utils import (rotate_image,
                   merge_tta_results,
                   rotate_results_back,
                   filter_bboxes_within_bounds,
                   compute_slice_boundaries
                   )

def infer_one_image(img, model):
    """Function to handle an image input that can be a path or an image array.
    Perform inference directly on the given image.

    Args:
        img (str or numpy.ndarray): The image path or the loaded image data.
        model (Model): The loaded model object for inference.

    Returns:
        dict: The detection results from the model.
    """
    # Check if img is a string and assume it's a path to an image file
    if isinstance(img, str):
        img = mmcv.imread(img)  # Load the image if a path is provided

    # Perform inference on the image
    result = inference_detector(model, img)

    return result.pred_instances


def infer_one_tta(image_path, model, tta_iou=0.2):
    original_img = mmcv.imread(image_path)
    angles = [0, 90, 180, 270]
    tta_results = []

    for angle in angles:
        # Rotate image
        rotated_img = rotate_image(original_img, angle)

        # Process rotated image
        results = infer_one_image(rotated_img, model)

        # Rotate results back and store
        rotated_back_results = rotate_results_back(
            results, -angle, original_img.shape[1], original_img.shape[0])
        tta_results.append(rotated_back_results)

    # Merge all TTA results
    final_results = merge_tta_results(tta_results, iou_threshold=tta_iou)
    return final_results


def infer_slice_tta(
    image_path,
    model,
    scale_factor=3,
    slice_iou=0.2,
    tta_iou=0.2,
    overlap=0.25
):
    original_img = mmcv.imread(image_path)
    angles = [0, 90, 180, 270]
    tta_results = []

    for angle in angles:
        # Rotate image
        rotated_img = rotate_image(original_img, angle)

        # Process rotated image
        results = infer_slice(rotated_img, model, scale_factor,
                              slice_iou=slice_iou, overlap=overlap)

        # Rotate results back and store
        rotated_back_results = rotate_results_back(
            results, -angle, original_img.shape[1], original_img.shape[0])
        tta_results.append(rotated_back_results)

    # Merge all TTA results
    final_results = merge_tta_results(tta_results, iou_threshold=tta_iou)
    return final_results


def infer_slice(img, model, scale_factor=3, slice_iou=0.2, overlap=0.25):
    """Function to handle large images, scale them up for slicing, inference, and then correctly scale and merge the results."""

    if isinstance(img, str):
        img = mmcv.imread(img)  # Load the image if a path is provided

    # # Enlarge the image according to the scale factor
    # img_enlarged = cv2.resize(img, (int(img.shape[1] * scale_factor), int(
    #     img.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)

    # 计算切片尺寸
    height, width = img.shape[:2]
    slice_height = round(height / scale_factor)
    slice_width = round(width / scale_factor)

    # 切片处理
    sliced_image_object = slice_image(
        img,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap,  # 这里也可以作为参数调整
        overlap_width_ratio=overlap
    )

    slice_results = []
    for img_slice in tqdm(sliced_image_object.images, desc="Processing slices"):
        slice_result = inference_detector(model, img_slice)
        slice_results.append(slice_result)
    
    starting_pixels=sliced_image_object.starting_pixels

    slice_boundaries =compute_slice_boundaries(
        starting_pixels,
        height, 
        width, 
        slice_height,
        slice_width,
        buffer=10)
    
    for (start_pixel, slice_result) in zip(starting_pixels, slice_results):
        # start_pixel 是起始坐标，slice_result 是对应的推理结果
        rect=slice_boundaries[tuple(start_pixel)]
        pred_instances=slice_result.pred_instances
        keep=filter_bboxes_within_bounds(pred_instances.bboxes, rect["start_point"], rect["end_point"])
        pred_instances=pred_instances[keep]
        slice_result.pred_instances=pred_instances

    image_instances = merge_results_by_nms(
        slice_results,
        sliced_image_object.starting_pixels,
        src_image_shape=(img.shape[1], img.shape[0]),  # Original image shape
        nms_cfg={'type': "nms", 'iou_threshold': slice_iou}
    )

    return image_instances  # .pred_instances
