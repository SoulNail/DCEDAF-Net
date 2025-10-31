import torch
import time
import numpy as np
import os

from models import metrics
from models import utils
from models import dataloader as dataloader_hub
from models import model_implements
from train import Trainer_seg
from PIL import Image

import numpy as np
import pickle
import os
from datetime import datetime


# class Inferencer:
#     def __init__(self, args):
#         self.start_time = time.time()
#         self.args = args
#
#         use_cuda = self.args.cuda and torch.cuda.is_available()
#         self.device = torch.device('cuda' if use_cuda else 'cpu')
#
#         self.loader_form = self.__init_data_loader(self.args.val_x_path,
#                                                    self.args.val_y_path,
#                                                    batch_size=1,
#                                                    mode='validation')
#
#         self.model = Trainer_seg.init_model(self.args.model_name, self.device, self.args)
#         self.model.load_state_dict(torch.load(args.model_path))
#         self.model.eval()
#
#         self.metric = self._init_metric(self.args.task, self.args.n_classes)
#
#         self.image_mean = self.loader_form.image_loader.image_mean
#         self.image_std = self.loader_form.image_loader.image_std
#         self.fn_list = []
class Inferencer:
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.loader_form = self.__init_data_loader(self.args.val_x_path,
                                                   self.args.val_y_path,
                                                   batch_size=1,
                                                   mode='validation')

        self.model = Trainer_seg.init_model(self.args.model_name, self.device, self.args)

        # 修改的部分：安全加载模型权重
        checkpoint = torch.load(args.model_path, map_location=self.device)

        # 过滤掉 total_ops 和 total_params 相关的键
        if isinstance(checkpoint, dict):
            # 检查是否是包装的checkpoint
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 过滤统计信息键
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if 'total_ops' not in k and 'total_params' not in k
            }
        else:
            filtered_state_dict = checkpoint

        # 加载过滤后的state_dict，使用strict=False以容忍可能的不匹配
        self.model.load_state_dict(filtered_state_dict, strict=False)

        # 打印加载信息（可选）
        print(f"Model loaded from {args.model_path}")
        if isinstance(checkpoint, dict):
            removed_keys = len(state_dict) - len(filtered_state_dict)
            if removed_keys > 0:
                print(f"Filtered out {removed_keys} statistical keys (total_ops/total_params)")

        self.model.eval()

        self.metric = self._init_metric(self.args.task, self.args.n_classes)

        self.image_mean = self.loader_form.image_loader.image_mean
        self.image_std = self.loader_form.image_loader.image_std
        self.fn_list = []



    # def start_inference_segmentation(self):
    # def start_inference_segmentation(self):
    #     f1_list = []
    #     acc_list = []
    #     auc_list = []
    #     sen_list = []
    #     mcc_list = []
    #
    #     # 添加时间统计变量
    #     total_inference_time = 0
    #     total_preprocessing_time = 0
    #     total_postprocessing_time = 0
    #     num_images = 0
    #
    #     for batch_idx, (img, target) in enumerate(self.loader_form.Loader):
    #         with torch.no_grad():
    #             # 1. 预处理时间统计
    #             preprocess_start = time.time()
    #             x_in, img_id = img
    #             target, origin_size = target
    #             x_in = x_in.to(self.device)
    #             x_img = x_in
    #             target = target.long().to(self.device)
    #             preprocess_end = time.time()
    #             total_preprocessing_time += (preprocess_end - preprocess_start)
    #
    #             # 2. 模型推理时间统计（最重要）
    #             inference_start = time.time()
    #             # 如果使用GPU，添加同步以获得准确时间
    #             if self.device.type == 'cuda':
    #                 torch.cuda.synchronize()
    #
    #             output = self.model(x_in)
    #
    #             if self.device.type == 'cuda':
    #                 torch.cuda.synchronize()
    #             inference_end = time.time()
    #             total_inference_time += (inference_end - inference_start)
    #
    #             if isinstance(output, tuple) or isinstance(output, list):
    #                 output = output[0]
    #
    #             # 3. 后处理时间统计
    #             postprocess_start = time.time()
    #             metric_result = self.post_process(output, target, x_img, img_id)
    #             postprocess_end = time.time()
    #             total_postprocessing_time += (postprocess_end - postprocess_start)
    #
    #             f1_list.append(metric_result['f1'])
    #             acc_list.append(metric_result['acc'])
    #             auc_list.append(metric_result['auc'])
    #             sen_list.append(metric_result['sen'])
    #             mcc_list.append(metric_result['mcc'])
    #
    #             num_images += 1
    #
    #             # 4. 单张图片的时间输出（可选）
    #             single_image_time = (inference_end - inference_start) * 1000  # 转换为毫秒
    #             print(f'Image {img_id[0]} - Inference time: {single_image_time:.2f} ms')
    #
    #     # 5. 统计信息输出
    #     metrics = self.metric.get_results()
    #     cIoU = [metrics['Class IoU'][i] for i in range(self.args.n_classes + 1)]
    #     mIoU = sum(cIoU) / (self.args.n_classes + 1)
    #
    #     print('\n========== Performance Metrics ==========')
    #     print('mean mIoU', mIoU)
    #     print('mean F1 score:', sum(f1_list) / len(f1_list))
    #     print('mean Accuracy', sum(acc_list) / len(acc_list))
    #     print('mean AUC', sum(auc_list) / len(auc_list))
    #     print('mean Sensitivity', sum(sen_list) / len(sen_list))
    #     print('mean MCC', sum(mcc_list) / len(mcc_list))
    #
    #     print('\n========== Time Statistics ==========')
    #     print(f'Total images processed: {num_images}')
    #     print(f'Average preprocessing time: {(total_preprocessing_time / num_images) * 1000:.2f} ms/image')
    #     print(f'Average inference time: {(total_inference_time / num_images) * 1000:.2f} ms/image')
    #     print(f'Average postprocessing time: {(total_postprocessing_time / num_images) * 1000:.2f} ms/image')
    #     print(
    #         f'Average total time per image: {((total_preprocessing_time + total_inference_time + total_postprocessing_time) / num_images) * 1000:.2f} ms')
    #     print(f'FPS (inference only): {num_images / total_inference_time:.2f}')
    #     print(
    #         f'FPS (total): {num_images / (total_preprocessing_time + total_inference_time + total_postprocessing_time):.2f}')



    def start_inference_segmentation(self):
        f1_list = []
        acc_list = []
        auc_list = []
        sen_list = []
        mcc_list = []

        # 添加时间统计变量
        total_inference_time = 0
        total_preprocessing_time = 0
        total_postprocessing_time = 0
        num_images = 0

        # ============ 新增：用于保存AUC详细数据 ============
        auc_detailed_data = {}  # 存储每张图的详细AUC数据
        all_predictions = []  # 存储所有预测值（概率）
        all_targets = []  # 存储所有真实标签
        image_wise_data = []  # 存储每张图的数据

        # 创建保存结果的目录
        save_dir = f"auc_analysis/{self.args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n将保存AUC分析数据到: {save_dir}")
        print("=" * 60)

        for batch_idx, (img, target) in enumerate(self.loader_form.Loader):
            with torch.no_grad():
                # 1. 预处理时间统计
                preprocess_start = time.time()
                x_in, img_id = img
                target, origin_size = target
                x_in = x_in.to(self.device)
                x_img = x_in
                target = target.long().to(self.device)
                preprocess_end = time.time()
                total_preprocessing_time += (preprocess_end - preprocess_start)

                # 2. 模型推理时间统计（最重要）
                inference_start = time.time()
                # 如果使用GPU，添加同步以获得准确时间
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                output = self.model(x_in)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                inference_end = time.time()
                total_inference_time += (inference_end - inference_start)

                if isinstance(output, tuple) or isinstance(output, list):
                    output = output[0]

                # ============ 新增：保存原始预测概率 ============
                # 获取预测概率（softmax之后的值）
                if output.shape[1] > 1:  # 多类别分割
                    pred_probs = torch.softmax(output, dim=1)
                    # 假设前景类是第1类（index=1）
                    pred_probs_foreground = pred_probs[:, 1, :, :].cpu().numpy()
                else:  # 二分类
                    pred_probs = torch.sigmoid(output).cpu().numpy()
                    pred_probs_foreground = pred_probs.squeeze()

                # 保存当前图像的预测概率和真实标签
                target_numpy = target.cpu().numpy()

                # 记录单张图像数据
                for i in range(len(img_id)):
                    single_pred = pred_probs_foreground[i] if len(
                        pred_probs_foreground.shape) > 2 else pred_probs_foreground
                    single_target = target_numpy[i] if len(target_numpy.shape) > 2 else target_numpy

                    # 展平用于AUC计算
                    pred_flat = single_pred.flatten()
                    target_flat = single_target.flatten()

                    # 保存到总列表
                    all_predictions.extend(pred_flat)
                    all_targets.extend(target_flat)

                    # 保存单张图像的详细数据
                    image_data = {
                        'image_id': img_id[i],
                        'predictions': single_pred,  # 保存2D预测图
                        'targets': single_target,  # 保存2D真实标签
                        'pred_flat': pred_flat,
                        'target_flat': target_flat,
                    }
                    image_wise_data.append(image_data)

                # 3. 后处理时间统计
                postprocess_start = time.time()
                metric_result = self.post_process(output, target, x_img, img_id)
                postprocess_end = time.time()
                total_postprocessing_time += (postprocess_end - postprocess_start)

                # ============ 新增：记录每张图的AUC值 ============
                current_auc = metric_result['auc']
                auc_detailed_data[img_id[0]] = {
                    'auc': current_auc,
                    'f1': metric_result['f1'],
                    'acc': metric_result['acc'],
                    'sen': metric_result['sen'],
                    'mcc': metric_result['mcc']
                }

                f1_list.append(metric_result['f1'])
                acc_list.append(metric_result['acc'])
                auc_list.append(current_auc)
                sen_list.append(metric_result['sen'])
                mcc_list.append(metric_result['mcc'])

                num_images += 1

                # 4. 单张图片的时间输出（可选）
                single_image_time = (inference_end - inference_start) * 1000  # 转换为毫秒
                print(f'Image {img_id[0]} - AUC: {current_auc:.4f}, F1: {metric_result["f1"]:.4f}, '
                      f'Inference time: {single_image_time:.2f} ms')

        # ============ 新增：计算整体AUC（使用所有像素） ============
        from sklearn.metrics import roc_curve, auc as auc_score

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 计算整体ROC曲线
        fpr_overall, tpr_overall, thresholds_overall = roc_curve(all_targets, all_predictions)
        auc_overall = auc_score(fpr_overall, tpr_overall)

        # 5. 统计信息输出
        metrics = self.metric.get_results()
        cIoU = [metrics['Class IoU'][i] for i in range(self.args.n_classes + 1)]
        mIoU = sum(cIoU) / (self.args.n_classes + 1)

        print('\n========== Performance Metrics ==========')
        print('mean mIoU', mIoU)
        print('mean F1 score:', sum(f1_list) / len(f1_list))
        print('mean Accuracy', sum(acc_list) / len(acc_list))
        print('mean AUC (image-wise average):', sum(auc_list) / len(auc_list))
        print('Overall AUC (all pixels):', auc_overall)  # 新增：整体AUC
        print('mean Sensitivity', sum(sen_list) / len(sen_list))
        print('mean MCC', sum(mcc_list) / len(mcc_list))

        # ============ 新增：保存详细AUC数据 ============
        print('\n========== AUC Analysis ==========')
        print(f'Image-wise AUC range: [{min(auc_list):.4f}, {max(auc_list):.4f}]')
        print(f'Image-wise AUC std: {np.std(auc_list):.4f}')

        # 打印每张图的AUC
        print('\nPer-image AUC values:')
        for img_id, data in auc_detailed_data.items():
            print(f"  {img_id}: AUC={data['auc']:.4f}")

        # 保存数据用于后续分析
        save_data = {
            'model_name': self.args.model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'per_image_metrics': auc_detailed_data,
            'image_wise_data': image_wise_data,  # 包含每张图的预测和标签
            'overall_metrics': {
                'mIoU': mIoU,
                'mean_f1': sum(f1_list) / len(f1_list),
                'mean_acc': sum(acc_list) / len(acc_list),
                'mean_auc_imagewise': sum(auc_list) / len(auc_list),
                'overall_auc': auc_overall,
                'mean_sen': sum(sen_list) / len(sen_list),
                'mean_mcc': sum(mcc_list) / len(mcc_list)
            },
            'roc_curve_data': {
                'fpr': fpr_overall.tolist(),
                'tpr': tpr_overall.tolist(),
                'thresholds': thresholds_overall.tolist(),
                'auc': auc_overall
            },
            'all_predictions': all_predictions,  # 所有像素的预测值
            'all_targets': all_targets,  # 所有像素的真实值
        }

        # 保存为pickle文件
        pickle_path = os.path.join(save_dir, 'auc_analysis_data.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f'\nAUC analysis data saved to: {pickle_path}')

        # 也保存为numpy文件（便于其他工具读取）
        np_save_path = os.path.join(save_dir, 'predictions_and_targets.npz')
        np.savez(np_save_path,
                 predictions=all_predictions,
                 targets=all_targets,
                 fpr=fpr_overall,
                 tpr=tpr_overall,
                 auc=auc_overall)
        print(f'Numpy data saved to: {np_save_path}')

        # 保存简单的文本报告
        report_path = os.path.join(save_dir, 'auc_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n{'=' * 50}\n")
            f.write(f"Overall Metrics:\n")
            f.write(f"  mIoU: {mIoU:.4f}\n")
            f.write(f"  Mean F1: {sum(f1_list) / len(f1_list):.4f}\n")
            f.write(f"  Mean Acc: {sum(acc_list) / len(acc_list):.4f}\n")
            f.write(f"  Mean AUC (image-wise): {sum(auc_list) / len(auc_list):.4f}\n")
            f.write(f"  Overall AUC (all pixels): {auc_overall:.4f}\n")
            f.write(f"  Mean Sensitivity: {sum(sen_list) / len(sen_list):.4f}\n")
            f.write(f"  Mean MCC: {sum(mcc_list) / len(mcc_list):.4f}\n")
            f.write(f"\n{'=' * 50}\n")
            f.write(f"Per-image AUC values:\n")
            for img_id, data in auc_detailed_data.items():
                f.write(f"  {img_id}: AUC={data['auc']:.4f}, F1={data['f1']:.4f}\n")
        print(f'Report saved to: {report_path}')

        return save_dir  # 返回保存目录路径


    def post_process(self, output, target, x_img, img_id):
        # reconstruct original image
        x_img = x_img.squeeze(0).data.cpu().numpy()
        x_img = np.transpose(x_img, (1, 2, 0))
        x_img = x_img * np.array(self.image_std)
        x_img = x_img + np.array(self.image_mean)
        x_img = x_img * 255.0
        x_img = x_img.astype(np.uint8)

        output = utils.remove_center_padding(output)
        target = utils.remove_center_padding(target)

        output_argmax = torch.where(output > 0.5, 1, 0).cpu().detach()
        self.metric.update(target.squeeze(1).cpu().detach().numpy(), output_argmax.numpy())

        path, fn = os.path.split(img_id[0])
        img_id, ext = os.path.splitext(fn)
        dir_path, fn = os.path.split(self.args.model_path)
        fn, ext = os.path.splitext(fn)
        save_dir = dir_path + '/' + fn + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        Image.fromarray(x_img).save(save_dir + img_id + '.png', quality=100)
        Image.fromarray((output_argmax.squeeze().numpy() * 255).astype(np.uint8)).save(save_dir + img_id + f'_argmax.png', quality=100)
        # Image.fromarray(output_heatmap.astype(np.uint8)).save(save_dir + img_id + f'_heatmap_overlay.png', quality=100)

        metric_result = metrics.metrics_np(output_argmax[None, :], target.squeeze(0).detach().cpu().numpy(), b_auc=True)
        print(f'{img_id} \t Done !!')

        return metric_result

    def __init_model(self, model_name):
        if model_name == 'UNet':
            model = model_implements.UNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'UNet2P':
            model = model_implements.UNet2P(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'UNet3P_Deep':
            model = model_implements.UNet3P_Deep(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ResUNet':
            model = model_implements.ResUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ResUNet2P':
            model = model_implements.ResUNet2P(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'SAUNet':
            model = model_implements.SAUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ATTUNet':
            model = model_implements.ATTUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'DCSAU_UNet':
            model = model_implements.DCSAU_UNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'AGNet':
            model = model_implements.AGNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'R2UNet':
            model = model_implements.R2UNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ConvUNeXt':
            model = model_implements.ConvUNeXt(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'FRUNet':
            model = model_implements.FRUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'FSGNet':
            model = model_implements.FSGNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'MyModel_modify':
            model = model_implements.MyModel_modify(n_classes=1, in_channels=self.args.input_channel)
        else:
            raise Exception('No model named', model_name)

        return torch.nn.DataParallel(model)

    def __init_data_loader(self,
                           x_path,
                           y_path,
                           batch_size,
                           mode):

        if self.args.dataloader == 'Image2Image_zero_pad':
            loader = dataloader_hub.Image2ImageDataLoader_zero_pad(x_path=x_path,
                                                                   y_path=y_path,
                                                                   batch_size=batch_size,
                                                                   num_workers=self.args.worker,
                                                                   pin_memory=self.args.pin_memory,
                                                                   mode=mode,
                                                                   args=self.args)
        else:
            raise Exception('No dataloader named', self.args.dataloader)

        return loader

    def _init_metric(self, task_name, num_class):
        if task_name == 'segmentation':
            metric = metrics.StreamSegMetrics_segmentation(num_class + 1)
        else:
            raise Exception('No task named', task_name)

        return metric
