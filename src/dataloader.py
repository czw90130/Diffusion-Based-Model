import cv2
import numpy as np
import os
import random
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class gray_color_data(Dataset):
    def __init__(self,path_color,path_gray):
        super().__init__()
        self.path_color = path_color
        self.path_gray = path_gray
        self.data_color = np.load(path_color)
        self.data_gray = np.load(path_gray)
    def __len__(self):
        return len(self.data_color)
    def __getitem__(self,idx):
        image_gray =  self.data_gray[idx]
        shape = (image_gray.shape[0],image_gray.shape[1],3)
        image_color = np.zeros(shape)
        image_color[:,:,0] = image_gray
        image_color[:,:,1:] = self.data_color[idx]
        image_color = image_color.astype('uint8')
        image_color = cv2.cvtColor(image_color,cv2.COLOR_LAB2RGB)
        return(ToTensor()(image_gray),ToTensor()(image_color))

class gcloth_mask_uv(Dataset):
    def __init__(self, csv_path, edge, low_thres, up_thres):
        super().__init__()
        self.edge = edge

        self.low_thres = low_thres
        self.up_thres = up_thres
        # fix the averange depth to 1555 mm (for better results)
        self.fix_p = (up_thres-low_thres) / 2 + low_thres

        self.csv_path = csv_path
        print("From file: {}".format(self.csv_path))

        with open(filename, "r") as f:
            self.image_lists = f.readlines()
            input_path = image_lists[0].split(',')[0]
            if not os.path.exists(input_path):
                raise ValueError(
                    'input path in csv not exist: {}'.format(input_path))
    def __len__(self):
        return len(self.image_lists)

    def cut_img(self, img, edge):
        if img.shape[1] != edge:
            l = img.shape[1]
            e = (l-edge)//2
            if img.shape[1] - 2*e == edge:
                img = img[:,e:-e]
            else:
                img = img[:,e+1:-e]
        return img

    def __getitem__(self,idx):
        image_names = self.image_lists[idx].strip().split(',')
        # INPUT前深度图
        depth_front = cv2.imread(image_names[0], cv2.IMREAD_UNCHANGED)
        depth_front = self.cut_img(depth_front, self.edge)
        # INPUT后深度图
        depth_back = cv2.imread(image_names[1], cv2.IMREAD_UNCHANGED)
        depth_back = self.cut_img(depth_back, self.edge)
        # 增加随机噪声
        if bool(random.getrandbits(1)):
            mask = (depth_front > 0)
            mean = 0
            var = random.randint(50,800)
            noise = (np.random.normal(mean, var ** 0.5, depth_front.shape) * mask).astype(np.uint16)
            depth_front -= noise
            var = random.randint(50,800)
            noise = (np.random.normal(mean, var ** 0.5, depth_front.shape) * mask).astype(np.uint16)
            depth_back += noise

        # BODY前深度图
        depth_body_front = cv2.imread(image_names[2], cv2.IMREAD_UNCHANGED)
        depth_body_front = self.cut_img(depth_body_front, self.edge)
        # CLOTH前深度图
        depth_cloth_front = cv2.imread(image_names[5], cv2.IMREAD_UNCHANGED)
        depth_cloth_front = self.cut_img(depth_cloth_front, self.edge)
        # 计算前gap图
        gap = 1
        gap_front = depth_body_front.astype(np.float64) - depth_cloth_front.astype(np.float64)
        gap_front /= gap
        gap_front[gap_front<0] = 0.0
        gap_front[gap_front>1] = 1.0
        gap_front = gap_front * 2.0 - 1.0
        gap_front[np.logical_and(depth_cloth_front>1, depth_body_front<1)] = 1.0
        gap_front[np.logical_and(depth_cloth_front<1, depth_body_front>1)] = -1.0

        # BODY后深度图
        depth_body_back = cv2.imread(image_names[6], cv2.IMREAD_UNCHANGED)
        depth_body_back = self.cut_img(depth_body_back, self.edge)
        # CLOTH后深度图
        depth_cloth_back = cv2.imread(image_names[9], cv2.IMREAD_UNCHANGED)
        depth_cloth_back = self.cut_img(depth_cloth_back, self.edge)
        # 计算后gap图
        gap_back = depth_cloth_back.astype(np.float64) - depth_body_back.astype(np.float64)
        gap_back /= gap
        gap_back[gap_back<0] = 0.0
        gap_back[gap_back>1] = 1.0
        gap_back = gap_back * 2.0 - 1.0
        gap_back[np.logical_and(depth_cloth_back>1, depth_body_back<1)] = 1.0
        gap_back[np.logical_and(depth_cloth_back<1, depth_body_back>1)] = -1.0
        #进行 UV to UVW
        umap_body_front = None
        vmap_body_front = None
        umap_body_back = None
        vmap_body_back = None

        # BODY前U坐标图
        umap_body_front = cv2.imread(image_names[3], cv2.IMREAD_UNCHANGED)
        umap_body_front = self.cut_img(umap_body_front, self.edge)
        # BODY前V坐标图
        vmap_body_front = cv2.imread(image_names[4], cv2.IMREAD_UNCHANGED)
        vmap_body_front = self.cut_img(vmap_body_front, self.edge)
        # BODY后U坐标图
        umap_body_back = cv2.imread(image_names[7], cv2.IMREAD_UNCHANGED)
        umap_body_back = self.cut_img(umap_body_back, self.edge)
        # BODY后V坐标图
        vmap_body_back = cv2.imread(image_names[8], cv2.IMREAD_UNCHANGED)
        vmap_body_back = self.cut_img(vmap_body_back, self.edge)
            

        umap_body_front = umap_body_front.astype(np.float64)
        umap_body_front = umap_body_front / 5000.0 - 1.0

        vmap_body_front = vmap_body_front.astype(np.float64)
        vmap_body_front = vmap_body_front / 5000.0 - 1.0


        umap_body_back = umap_body_back.astype(np.float64)
        umap_body_back = umap_body_back / 5000.0 - 1.0

        vmap_body_back = vmap_body_back.astype(np.float64)
        vmap_body_back = vmap_body_back / 5000.0 - 1.0

        ########################################################

        depth_front = torch.tensor(depth_front.astype(np.float32))
        depth_back = torch.tensor(depth_back.astype(np.float32))
        
        umap_body_front = torch.tensor(umap_body_front)
        vmap_body_front = torch.tensor(vmap_body_front)
        depth_body_front = torch.tensor(depth_body_front.astype(np.float32))
        depth_cloth_mask_front = torch.tensor(gap_front.astype(np.float32))

        umap_body_back = torch.tensor(umap_body_back)
        vmap_body_back = torch.tensor(vmap_body_back)
        depth_body_back = torch.tensor(depth_body_back.astype(np.float32))
        depth_cloth_mask_back = torch.tensor(gap_back.astype(np.float32))


        # 处理 input
        o_mask = (depth_front > 0).float()

        tmp1 = self.fix_p - (torch.sum(depth_front) / torch.sum(o_mask))

        depth_front = depth_front + tmp1
        depth_back = depth_back + tmp1

        depth_body_front = depth_body_front + tmp1
        depth_body_back = depth_body_back + tmp1

        # low_thres 500.0; up_thres 3000.0
        o_depth_front = ops.convert_depth_to_m1_1(
            depth_front, low_thres, up_thres).clamp(-1.0, 1.0)
        o_depth_back = ops.convert_depth_to_m1_1(
            depth_back, low_thres, up_thres).clamp(-1.0, 1.0)

        body_depth_front = ops.convert_depth_to_m1_1(
            depth_body_front, low_thres, up_thres).clamp(-1.0, 1.0)
        body_depth_back = ops.convert_depth_to_m1_1(
            depth_body_back, low_thres, up_thres).clamp(-1.0, 1.0)

        x_in = torch.cat((o_depth_front, o_depth_back), dim=1)

        real_y = torch.cat((
                        umap_body_front, vmap_body_front,
                        body_depth_front, depth_cloth_mask_front, 

                        umap_body_back, vmap_body_back,
                        body_depth_back, depth_cloth_mask_back,
                          ), dim=1)

        return(x_in, real_y)