import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

"""
# ToTensor的使用
writer = SummaryWriter("logs")

img_path = "D:\\desktop\\xmym\\my_self_learing\\pytorch\\greenhand\\train\\ants_image\\45472593_bfd624f8dc.jpg"
img = Image.open(img_path)  # 读取一张图片,PIL_img
cv_img = cv2.imread(img_path)  # 读取一张图片,numpy.array
tensor_trans = transforms.ToTensor()  # 实例化ToTensor
tensor_img = tensor_trans(img)  # 调用转换,输出结果就会输出一个tensor的数据类型
print(tensor_img)

writer.add_image("img_show", tensor_img, 1, None, 'CHW')  # 使用Tensorboard显示图片
writer.close()
"""

writer = SummaryWriter("logs")
#  常见的Transforms使用
img_path = "D:\\desktop\\xmym\\my_self_learing\\pytorch\\greenhand\\train\\ants_image\\45472593_bfd624f8dc.jpg"
img = Image.open(img_path)  # 读取一张图片,PIL_img
#  ToTensor
trans_tensor = transforms.ToTensor()  # 实例化ToTensor
img_tensor = trans_tensor(img)  # 调用转换,输出结果就会输出一个tensor的数据类型
writer.add_image("ToTensor", img_tensor)
# Normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)
# Resize
trans_res = transforms.Resize((512, 512))  # 输入目标尺寸大小(x, y),只输入一个数的话就是等比缩放
img_res = trans_res(img)  # 输入是PIL_img,输出也是PIL_img
img_res = trans_tensor(img_res)  # 转换为ToTensor
writer.add_image("Resize", img_res)
# Compose
tran_res_2 = transforms.Resize(512)  # 实例化一个等比转换的Resize
trans_com = transforms.Compose([tran_res_2, trans_tensor])
# 实例化一个Compose,输入是一个列表,列表内容必须是transform的实例化,且后一个参数的输入必须是前一个参数的输出
img_com = trans_com(img)  # 转换图片
writer.add_image("Compose", img_com)
# RandomCrop
print(img)
# trans_random = transforms.RandomCrop(40)  # 这个值不能超过图片的最小边
trans_random = transforms.RandomCrop((40, 50))  # 这个范围不能超过图片大小
trans_com_2 = transforms.Compose([trans_random, trans_tensor])
for i in range(10):
    img_crop = trans_com_2(img)
    writer.add_image("RandomCrop", img_crop, i)
writer.close()
