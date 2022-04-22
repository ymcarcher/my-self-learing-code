from torch.utils.data import dataset
from PIL import Image
import os


# 创建一个类,这个类继承于dataset下的Dataset
class Mydata(dataset.Dataset):
    # 初始化类为后面提供全局变量和方法
    def __init__(self, root_dir, label_dir):
        # self. 可以把变量编程后面函数可以使用的全局变量(私有)
        self.root_dir = root_dir  # 数据的根路径
        self.label_dir = label_dir  # 数据标签路径
        self.path = os.path.join(self.root_dir, self.label_dir)  # os.path.join()函数可以把路径相加
        self.img_path = os.listdir(self.path)  # os.listdir()函数可以获得path路径下单文件名,存在列表中

    # 这里以Image库为例
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_path_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_path_path)  # 读取图片
        label = self.label_dir  # 读取标签
        return img, label

    def __len__(self):
        return len(self.img_path)  # 返回数据集长度(数目)


root_dir = "datasheet/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_datasheet = Mydata(root_dir, ants_label_dir)
bees_datasheet = Mydata(root_dir, bees_label_dir)

train_datasheet = ants_datasheet + bees_datasheet  # 将两个数据集相加
