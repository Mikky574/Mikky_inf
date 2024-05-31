import mmcv
import numpy as np
from large_image import merge_results_by_nms
from mmdet.apis import inference_detector
import cv2
from sahi.slicing import slice_image

from utils import (rotate_image,
                     merge_tta_results,
                     rotate_results_back,
                     scale_down_detections)


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


def infer_one_tta(image_path,model,iou_threshold=0.2):
    original_img = mmcv.imread(image_path)
    angles = [0, 90, 180, 270]
    tta_results = []

    for angle in angles:
        # Rotate image
        rotated_img = rotate_image(original_img, angle)

        # Process rotated image
        results = infer_one_image(rotated_img,model)

        # Rotate results back and store
        rotated_back_results = rotate_results_back(
            results, -angle, original_img.shape[1], original_img.shape[0])
        tta_results.append(rotated_back_results)

    # Merge all TTA results
    final_results = merge_tta_results(tta_results, iou_threshold=iou_threshold)
    return final_results


def infer_slice_tta(image_path,model,scale_factor=3,iou_threshold=0.2):
    original_img = mmcv.imread(image_path)
    angles = [0, 90, 180, 270]
    tta_results = []

    for angle in angles:
        # Rotate image
        rotated_img = rotate_image(original_img, angle)

        # Process rotated image
        results = infer_slice(rotated_img,model,scale_factor)

        # Rotate results back and store
        rotated_back_results = rotate_results_back(
            results, -angle, original_img.shape[1], original_img.shape[0])
        tta_results.append(rotated_back_results)

    # Merge all TTA results
    final_results = merge_tta_results(tta_results, iou_threshold=iou_threshold)
    return final_results

def infer_slice(img,model, scale_factor):
    """Function to handle large images, scale them up for slicing, inference, and then correctly scale and merge the results."""

    if isinstance(img, str):
        img = mmcv.imread(img)  # Load the image if a path is provided

    # Enlarge the image according to the scale factor
    img_enlarged = cv2.resize(img, (int(img.shape[1] * scale_factor), int(
        img.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)

    # Slice the enlarged image
    sliced_image_object = slice_image(
        img_enlarged,
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.5,
        overlap_width_ratio=0.5
    )

    # Perform inference on the slices
    slice_results = []
    start = 0
    batch_size = 4
    small_shape = (round(img.shape[0]/scale_factor),
                   round(img.shape[1]/scale_factor), img.shape[2])
    while True:
        end = min(start + batch_size, len(sliced_image_object.images))
        batch_slices = sliced_image_object.images[start:end]

        # Inference
        batch_results = inference_detector(model, batch_slices)

        # Scale down the predictions for each batch result
        for result in batch_results:
            result = scale_down_detections(
                result.pred_instances, scale_factor, small_shape)
            # result = filter_by_score(result, 0.4)  # Filter results with score > 0.4

        slice_results.extend(batch_results)

        if end >= len(sliced_image_object.images):
            break
        start += batch_size

    # Scale down the starting pixels for merging results
    starting_pixels = np.array(sliced_image_object.starting_pixels)
    scaled_starting_pixels = starting_pixels / scale_factor

    # Convert the starting pixels into integer format if necessary
    scaled_starting_pixels = np.round(
        scaled_starting_pixels).astype(int).tolist()

    # Merge results with Non-Maximum Suppression (NMS)
    image_result = merge_results_by_nms(
        slice_results,
        scaled_starting_pixels,
        src_image_shape=(img.shape[1], img.shape[0]),  # Original image shape
        nms_cfg={'type': "nms", 'iou_threshold': 0.2}
    )

    return image_result.pred_instances
