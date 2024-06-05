import os
import json
import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector
from pycocotools.mask import encode

from process_ import (
    infer_one_image,
    infer_one_tta,
    infer_slice,
    infer_slice_tta
)

# # 初始化模型
# config_file = r'C:\Users\Mikky\Desktop\树冠数据集\USIS10K\work_dirs\USIS10KDataset\huge\multiclass_usis_train_huge.py'
# checkpoint_file = r'C:\Users\Mikky\Desktop\树冠数据集\USIS10K\work_dirs\USIS10KDataset\huge\epoch_9.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

config_file = r'C:\Users\Mikky\Desktop\树冠数据集\USIS10K\work_dirs\USIS10KDataset\huge\multiclass_usis_train_huge.py'
checkpoint_file = r'C:\Users\Mikky\Desktop\树冠数据集\USIS10K\work_dirs\USIS10KDataset\huge\best_coco_segm_mAP_epoch_30.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 图片文件夹路径
folder_path = r'C:\Users\Mikky\Desktop\树冠数据集\第二次数据集\Testing_set_phase1\Testing_set\images'

# Define scale factors for different datasets
scale_factors = {
    'Dataset_3': (1,"one-tta"),
    'Dataset_4': (2,"slice-tta"),
    'Dataset_5': (1,"one-tta")  # No scaling
}

# 读取映射文件
mapping_file_path = r'C:\Users\Mikky\Desktop\树冠数据集\第二次数据集\Testing_set_phase1\Testing_set\image_ids\validation_img_id.json'
with open(mapping_file_path, 'r') as file:
    data = json.load(file)

# 创建文件名到ID的映射
file_to_id = {img['file_name']: img['id'] for img in data['images']}

# 读取所有图片文件名
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

results_list = []

# 遍历并处理每张图片
for img_path in tqdm(image_files, desc="Processing Images"):
    img_file_name = os.path.basename(img_path)
    dataset_label = img_file_name.split('_')[0] + '_' + img_file_name.split('_')[1]
    scale_factor,infer_type = scale_factors.get(dataset_label, 1)  # 如果没有匹配到，默认 scale_factor 为 1

    # 根据推理类型选择函数
    if infer_type == "one":
        pred_instances = infer_one_image(img_path, model)
    elif infer_type == "one-tta":
        pred_instances = infer_one_tta(img_path, model,tta_iou=0.4)
    elif infer_type == "slice":
        pred_instances = infer_slice(img_path, model, scale_factor)
    elif infer_type == "slice-tta":
        pred_instances = infer_slice_tta(img_path, model, scale_factor,tta_iou=0.2)

    # else:
    #     continue  # 如果找不到对应的类型，跳过当前图片

    # pred_instances = infer_and_process(img_path, scale_factor)

    scores = pred_instances.scores.cpu().numpy()
    masks = pred_instances.masks # .cpu().numpy()
    bboxes = pred_instances.bboxes.cpu().numpy()
    
    img_file_name = os.path.basename(img_path)
    img_id = file_to_id.get(img_file_name, -1)  # 使用映射查找ID，如果未找到则返回-1

    for score, mask, bbox in zip(scores, masks, bboxes):
        # 获取RLE编码
        # print(mask.shape)
        rle = encode(np.asfortranarray(mask.cpu().numpy().astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')

        # 边界框转换
        bbox_out = [float(bbox[0]), float(bbox[1]),float(bbox[2]), float(bbox[3])] # [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]

        result_dict = {
            "image_id": img_id,
            "bbox": bbox_out,
            "score": float(score),
            "category_id": 1,  # 假设类别ID为1，根据实际情况调整
            "segmentation": {
                "size": [mask.shape[0], mask.shape[1]],
                "counts": rle['counts']
            }
        }
        results_list.append(result_dict)

# 保存为JSON
output_json_path = r'C:\Users\Mikky\Desktop\树冠数据集\results_sample.json'
with open(output_json_path, 'w') as f:
    json.dump(results_list, f, indent=4)

print(f"Completed. Results saved to {output_json_path}")
