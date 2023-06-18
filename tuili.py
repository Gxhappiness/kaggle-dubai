# -*- coding = utf-8 -*-
# @Time : 2023/6/18 11:49
# @Author : Happiness
# @File : tuili.py
# @Software : PyCharm


####导入工具包

import numpy as np
import matplotlib.pyplot as plt



from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2




####载入配置文件

# 载入 config 配置文件
from mmengine import Config
cfg = Config.fromfile('pspnet-DubaiDataset.py')
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# register all modules in mmseg into the registries
# do not init the default scope here because it will be init in the runner

register_all_modules(init_default_scope=False)
runner = Runner.from_cfg(cfg)


###载入模型

checkpoint_path = './work_dirs/DubaiDataset/iter_3000.pth'
model = init_model(cfg, checkpoint_path, 'cuda:0')




###载入测试集图像，或新图像

#使用绝对路径不怕报错
img = mmcv.imread('D:/0.dive into pytorch/openmmlab/mmsegmentation/data/Dubai-dataset/img_dir/val/3.jpg')




###语义分割预测

result = inference_model(model, img)
['seg_logits', 'pred_sem_seg']
pred_mask = result.pred_sem_seg.data[0].cpu().numpy()



###可视化语义分割预测结果

plt.imshow(pred_mask)
plt.show()


# 可视化预测结果
visualization = show_result_pyplot(model, img, result, opacity=0.7, out_file='pred.jpg')
plt.imshow(mmcv.bgr2rgb(visualization))
plt.show()



#####获取测试集标注

label = mmcv.imread('D:/0.dive into pytorch/openmmlab/mmsegmentation/data/Dubai-dataset/img_dir/val/3.jpg')

label_mask = label[:,:,0]
plt.imshow(label_mask)
plt.show()



####对比测试集标注和语义分割预测结果
# 测试集标注


# 语义分割预测结果

# 真实为前景，预测为前景
TP = (label_mask == 1) & (pred_mask==1)
# 真实为背景，预测为背景
TN = (label_mask == 0) & (pred_mask==0)
# 真实为前景，预测为背景
FN = (label_mask == 1) & (pred_mask==0)
# 真实为背景，预测为前景
FP = (label_mask == 0) & (pred_mask==1)
plt.imshow(TP)
plt.show()

confusion_map = TP * 255 + FP * 150 + FN * 80 + TN * 30
plt.imshow(confusion_map)
plt.show()



####混淆矩阵
from sklearn.metrics import confusion_matrix
confusion_matrix_model = confusion_matrix(label_mask.flatten(), pred_mask.flatten())

import itertools


def cnf_matrix_plotter(cm, classes, cmap=plt.cm.Blues):
    """
    传入混淆矩阵和标签名称列表，绘制混淆矩阵
    """
    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar() # 色条
    tick_marks = np.arange(len(classes))

    plt.title('Confusion Matrix', fontsize=30)
    plt.xlabel('Pred', fontsize=25, c='r')
    plt.ylabel('True', fontsize=25, c='r')
    plt.tick_params(labelsize=16)  # 设置类别文字大小
    plt.xticks(tick_marks, classes, rotation=90)  # 横轴文字旋转
    plt.yticks(tick_marks, classes)

    # 写数字
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 fontsize=12)

    plt.tight_layout()

    plt.savefig('混淆矩阵.pdf', dpi=300)  # 保存图像
    plt.show()


classes = ['Land', 'Road', 'Building', 'Vegetation', 'Water', 'Unlabeled']
cnf_matrix_plotter(confusion_matrix_model, classes, cmap='Blues')