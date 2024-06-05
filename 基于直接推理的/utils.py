from tqdm import tqdm
from mmdet.utils.large_image import merge_results_by_nms, shift_predictions
from sahi.slicing import slice_image
import mmcv
import torch
import numpy as np

from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg


def prepare_pipeline(model, imgs):  # imgs和img都可以用
    """
    根据模型配置和图像数据类型创建测试流水线。

    Args:
        model (Model): 一个带有配置的模型实例。
        imgs (list): 图像数据的列表，可以是图像路径列表或者numpy数组的列表。

    Returns:
        Compose: 构建好的测试流水线。
    """

    # 获取模型配置
    cfg = model.cfg.copy()

    # 根据图像数据的类型调整测试流水线配置
    test_pipeline_cfg = get_test_pipeline_cfg(cfg).copy()
    if type(imgs) == list:
        if isinstance(imgs[0], np.ndarray):
            # 如果输入是numpy数组，修改流水线配置以使用LoadImageFromNDArray
            test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    elif type(imgs) == str:
        test_pipeline_cfg[0] = dict(
            type='LoadImageFromFile', backend_args=None, to_float32=True)
    else:
        if isinstance(imgs, np.ndarray):
            # 如果输入是numpy数组，修改流水线配置以使用LoadImageFromNDArray
            test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'

    # 构建测试流水线
    test_pipeline = Compose(test_pipeline_cfg)
    return test_pipeline


def prepare_model_inputs(img, model, test_pipeline=None):
    """
    准备模型输入。

    Args:
        img (str or np.ndarray): 图像路径或者numpy数组。
        test_pipeline (Compose): 测试流水线，用于图像的预处理。
        model (YourModelClass): 使用的模型，必须包含data_preprocessor方法。

    Returns:
        tuple: 包含两个元素(inputs, data_samples)的元组，用于模型预测。
    """
    # 根据图像数据的类型准备数据字典
    if isinstance(img, np.ndarray):
        data_ = dict(img=img, img_id=0)
    else:
        data_ = dict(img_path=img, img_id=0)

    # 应用测试流水线对数据进行预处理
    data_ = test_pipeline(data_)

    # 转换为模型输入格式
    processed_data = model.data_preprocessor({
        "inputs": [data_['inputs']],
        "data_samples": [data_['data_samples']]
    })

    # 分离处理后的数据
    inputs = processed_data["inputs"]
    data_samples = processed_data["data_samples"]

    return inputs, data_samples

# def prepare_model_inputs(imgs, model,test_pipeline=None):
#     """
#     准备模型输入。

#     Args:
#         img (str or np.ndarray): 图像路径或者numpy数组。
#         test_pipeline (Compose): 测试流水线，用于图像的预处理。
#         model (YourModelClass): 使用的模型，必须包含data_preprocessor方法。

#     Returns:
#         tuple: 包含两个元素(inputs, data_samples)的元组，用于模型预测。
#     """

#     if isinstance(imgs, (list, tuple)):
#         is_batch = True
#     else:
#         imgs = [imgs]
#         is_batch = False

#     if test_pipeline is None:
#         test_pipeline=prepare_pipeline(model, imgs)

#     prepare_list = []
#     for i, img in enumerate(imgs):
#         if isinstance(img, np.ndarray):
#             # TODO: remove img_id.
#             data_ = dict(img=img, img_id=0)
#         else:
#             # TODO: remove img_id.
#             data_ = dict(img_path=img, img_id=0)

#         # build the data pipeline
#         data_ = test_pipeline(data_)

#         processed_data = model.data_preprocessor({
#             "inputs": [data_['inputs']],
#             "data_samples": [data_['data_samples']]
#         })

#         inputs = processed_data["inputs"]
#         data_samples = processed_data["data_samples"]
#         prepare_list.append((inputs,data_samples))

#     if not is_batch:
#         return prepare_list[0]
#     else:
#         return prepare_list


def inference_bbox(model, inputs, data_samples):
    """
    执行目标检测模型的推理流程，并删除mask预测。

    Args:
        model (YourModelClass): 模型实例，必须包含extract_feat, rpn_head, roi_head, 和 add_pred_to_datasample方法。
        inputs (list): 处理后的输入数据。
        data_samples (list): 处理后的数据样本。

    Returns:
        list: 更新后的数据样本，其中不包含mask信息。
    """
    with torch.no_grad():
        # 提取特征
        x, image_embeddings, image_positional_embeddings = model.extract_feat(
            inputs)

        # 使用 RPN 生成建议框
        if data_samples[0].get('proposals', None) is None:
            rpn_results_list = model.rpn_head.predict(
                x, data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples]

        # return model.add_pred_to_datasample(data_samples,rpn_results_list)
    
        # 使用处理后的建议框进行 Mask 预测
        results_list = model.roi_head.predict(
            x, rpn_results_list, data_samples, rescale=True,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )

        # 添加预测结果到数据样本中
        updated_data_samples = model.add_pred_to_datasample(
            data_samples, results_list)

        # 删除pred_instances中的masks属性
        for sample in updated_data_samples:
            if hasattr(sample.pred_instances, 'masks'):
                del sample.pred_instances.masks

    return updated_data_samples


def slice_bbox(img_path, model, scale_factor=1, overlap=0.5,nms=False, nms_cfg=None):
    """
    处理大图像，进行切片和推理，最后合并结果。

    Args:
        img_path (str): 图像的路径。
        model (Model): 使用的模型实例。
        scale_factor (float, optional): 图像的缩放因子。默认为1，不进行缩放。
        nms_cfg (dict, optional): NMS的配置。默认配置为{'type': "nms", 'iou_threshold': 0.2}。

    Returns:
        Tensor: 合并后的检测结果。
    """
    # 设置默认的NMS配置
    if nms_cfg is None:
        nms_cfg = {
            'type': "nms",
            'iou_threshold': 0.2
        }

    # 加载图像
    img = mmcv.imread(img_path)

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

    # 准备测试流水线
    test_pipeline = prepare_pipeline(model, sliced_image_object.images[0])

    # 处理每个切片并进行推理
    slice_results = []
    for img_slice in tqdm(sliced_image_object.images, desc="Processing slices"):
        inputs, data_samples = prepare_model_inputs(
            img_slice, model, test_pipeline)
        data_sample = inference_bbox(model, inputs, data_samples)[0]
        slice_results.append(data_sample)

    if nms:
        # 使用NMS合并结果
        image_result = merge_results_by_nms(
            slice_results,
            sliced_image_object.starting_pixels,
            src_image_shape=(height, width),
            nms_cfg=nms_cfg
        )
    else:
        results = shift_predictions(
            slice_results, sliced_image_object.starting_pixels, src_image_shape=(height, width),)
        image_result = slice_results[0].clone()
        image_result.set_metainfo({"ori_shape":(height,width),"scale_factor":(1,1)})
        image_result.pred_instances = results
    return image_result
