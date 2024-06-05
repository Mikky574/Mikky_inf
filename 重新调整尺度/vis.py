import cv2
import mmcv
import matplotlib.pyplot as plt
import numpy as np
import random

def random_color():
    """生成随机颜色"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def visualize_bbox_results(img_path, pred_instances):
    """
    Visualizes RPN results by drawing bounding boxes and scores on the image.

    Parameters:
        img_path (str or numpy.array): Path to the image file or a numpy array of the image.
        rpn_results (object): An object containing bounding boxes and scores.
                              It must have 'bboxes' and 'scores' attributes.
    """
    # 加载图像
    if isinstance(img_path, str):
        image = mmcv.imread(img_path)
    else:
        image = img_path

    # 转换为RGB以适应matplotlib显示
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取边界框和分数
    bboxes = pred_instances.bboxes
    scores = pred_instances.scores

    # 创建用于显示的图像副本
    display_image = image.copy()

    # 绘制每个边界框和分数
    for bbox, score in zip(bboxes, scores):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        color = random_color()

        # 绘制边界框
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

        # 显示分数
        cv2.putText(display_image, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 使用matplotlib显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(display_image)
    plt.axis('off')
    plt.show()


def visualize_detections(img_path, pred_instances):
    """
    Visualizes detections by drawing bounding boxes, scores, and masks on the image.

    Parameters:
        img_path (str or numpy.array): Path to the image file or a numpy array of the image.
        pred_instances (object): An object containing bounding boxes, scores, and masks.
                                 It must have 'bboxes', 'scores', and 'masks' attributes.
    """
    # 加载图像
    if isinstance(img_path, str):
        image = mmcv.imread(img_path)
    else:
        image = img_path

    # 转换为RGB以适应matplotlib显示
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取边界框、分数和掩码
    bboxes = pred_instances.bboxes
    scores = pred_instances.scores
    masks = pred_instances.masks

    # 创建用于显示的图像副本
    display_image = image.copy()

    # 绘制每个边界框、分数和掩码
    for bbox, score, mask in zip(bboxes, scores, masks):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        color = random_color()

        # 绘制边界框
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

        # 将掩码叠加到图像上
        mask = mask.cpu().numpy()  # 确保掩码是numpy数组
        for i in range(3):  # 叠加到RGB的每个通道
            display_image[:, :, i] = np.where(mask == 1,
                                              display_image[:, :, i] * 0.5 + color[i] * 0.5,
                                              display_image[:, :, i])

        # 显示分数
        cv2.putText(display_image, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 使用matplotlib显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(display_image)
    plt.axis('off')
    plt.show()

