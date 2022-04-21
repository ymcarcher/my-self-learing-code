# 导入tensorboard中的SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

img_path = ""
img = Image.open(img_path)
img = np.array(img)  # 将img转换为numpy.array
print(img.shape)  # 查看图片形状
writer.add_image("pic", img, 1, None, 'HWC')
"""
    add_image(tag(图像标题),
              img_tensor(图像数据类型,必须是Tensor,numpy.array,或者是字符串),
              global_step=None(训练步骤,x轴),
              wall_time=None,
              data_formats='CHW'
                    data_formats是指定数据的输入格式,默认是(3, H, W)C通道,H高度,W宽度
                    也可以自己改,比如'HW','HWC'
"""

for i in range(100):
    writer.add_scalar("y=x", i, i, None)  # 添加数据进去
    """
        add_scalar(tag(图表标题),
                   scalar_value(绘制的图像,y轴),
                   global_step(训练到多少步,x轴),
                   wall_time(不常用,不管,填None)
    """

writer.close()
