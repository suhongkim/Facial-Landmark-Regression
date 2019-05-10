# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# -----------------------------------------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import cv2
import random
import copy

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class LFWDataset(Dataset):
    def __init__(self, image_dir, anno_path, n_augmented=0, net_size=(225, 225), debug_mode=False):
        self.image_dir = image_dir
        self.anno_path = anno_path
        self.n_augmented = n_augmented
        self.net_size = net_size
        self.debug_mode = debug_mode
        self.lfw_list = self.get_lfwlist()

    def get_lfwlist(self):
        lfw_list = []
        with open(self.anno_path, "r") as f:
            for line in f:
                tokens = line.split()

                if len(tokens) < 19:
                    continue;

                image_path = tokens[0][:-9] + '/' + tokens[0]
                bbox = [int(tokens[1]),  # left top x
                        int(tokens[2]),  # left top y
                        int(tokens[3]),  # right bottom x
                        int(tokens[4])  # right bottom y
                        ]
                landmarks = [
                    [float(tokens[5]), float(tokens[6])],  # canthus_rr
                    [float(tokens[7]), float(tokens[8])],  # canthus_rl
                    [float(tokens[9]), float(tokens[10])],  # canthus_lr
                    [float(tokens[11]), float(tokens[12])],  # canthus_ll
                    [float(tokens[13]), float(tokens[14])],  # mouse_corner_r
                    [float(tokens[15]), float(tokens[16])],  # mouse_corner_l
                    [float(tokens[17]), float(tokens[18])]  # nose
                ]
                lfw_list.append({'image_path': image_path, 'bbox': bbox, 'landmarks': landmarks})
        return lfw_list

    def __len__(self):
        return len(self.lfw_list) * (self.n_augmented if self.n_augmented > 0 else 1)

    def __str__(self):
        return '==> len(lfw_list) : {}\n==> # of augmented data : {} '.format(len(self.lfw_list), self.n_augmented)

    def __getitem__(self, idx):
        # load image and landmarks ------------------------------------------------------
        sample_idx = (lambda i, n: i//n if n is not 0 else i)(idx, self.n_augmented)
        img = Image.open(os.path.join(self.image_dir, self.lfw_list[sample_idx]['image_path']))
        bbox = np.asarray(self.lfw_list[sample_idx]['bbox'], dtype=np.int)
        lmarks = np.asarray(self.lfw_list[sample_idx]['landmarks'], dtype=np.float32)
        sample = {'img': img, 'bbox': bbox, 'lmarks': lmarks}

        # Data Augmentation -------------------------------------------------------------
        if self.n_augmented > 0:
            d_sample = [] # for debugging
            d_sample.append(sample)
            sample = self.crop_image(sample, max_noise=20)
            d_sample.append(copy.deepcopy(sample))
            sample = self.flip_image(sample)
            d_sample.append(copy.deepcopy(sample))
            sample = self.change_brightness(sample, bright_noise=0.5)
            d_sample.append(copy.deepcopy(sample))

            if self.debug_mode:
                t_title = ['original', 'crop', 'flip', 'brightness']

                fig = plt.figure()
                for i in range(0, len(d_sample)):
                    ax = plt.subplot(1, len(d_sample), i + 1)
                    ax.imshow(d_sample[i]['img'])
                    ax.scatter(d_sample[i]['lmarks'][:, 0], d_sample[i]['lmarks'][:, 1], marker='.', c='r')
                    ax.set_title(t_title[i])
                fig.tight_layout()
                plt.suptitle('imate_path({}th): {}'.format(sample_idx, self.lfw_list[sample_idx]['image_path']))
                plt.show()
        else:
            # No data augmentation : bbox crop only
            sample = self.crop_image(sample, max_noise=0)
            if self.debug_mode:
                d_sample = sample
                plt.imshow(d_sample['img'])
                plt.scatter(d_sample['lmarks'][:, 0], d_sample['lmarks'][:, 1], marker='.', c='r')
                plt.tight_layout()
                plt.title('no augmentation')
                plt.show()

        # Rescle image to net input size----------------------------------------------
        sample = self.rescale_image(sample, output_size=self.net_size)

        # Feature Scaling-------------------------------------------------------------
        input_image = sample['img'] / 255. * 2. - 1  # input image: from -1 to 1
        label_lmarks = sample['lmarks'] / self.net_size[0] # label: from 0 to 1 because relu

        # Transfer from images to tensors---------------------------------------------
        image_tensor = torch.from_numpy(input_image.transpose()).type(torch.float32) # H, W, C -> C, H, W (Transpose)
        label_tensor = torch.from_numpy(label_lmarks).view((14)).type(torch.float32) # 7 X 2 -> 14

        if self.debug_mode:
            # check reverse
            image = image_tensor.numpy().astype(np.float32).transpose()  # c , h, w -> h, w, c
            landmarks = label_tensor.numpy().astype(np.float32).reshape((7, 2))
            landmarks = landmarks * 225

            plt.figure()
            plt.imshow((image + 1) / 2.)
            plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='.', c='r')
            plt.title('Tensor Image')
            plt.show()

        return image_tensor, label_tensor


    @staticmethod
    def crop_image(sample, max_noise=0):
        # Random Cropping : Bbox + noise
        s_img = sample['img']
        s_bbox = sample['bbox']
        s_lmrk = sample['lmarks']

        bbox_cropped = [s_bbox[0] - random.randint(0, max_noise),
                        s_bbox[1] - random.randint(0, max_noise),
                        s_bbox[2] + random.randint(0, max_noise),
                        s_bbox[3] + random.randint(0, max_noise)]

        s_img = s_img.crop(bbox_cropped)
        s_lmrk = s_lmrk - [bbox_cropped[0], bbox_cropped[1]]

        return {'img': s_img, 'bbox': s_bbox, 'lmarks': s_lmrk}

    @staticmethod
    def flip_image(sample):
        f_img = sample['img']
        f_lmrk = sample['lmarks']
        flip = random.choice([True, False])

        if flip:
            # transpose image
            f_img = f_img.transpose(Image.FLIP_LEFT_RIGHT)

            # change landmarks location as flipped position
            f_lmrk[:, 0] = f_img.width - f_lmrk[:, 0]

            # re-arrange landmarks as flipped order
            # 0, 1, 2, 3, 4, 5, 6 --> 3, 2, 1, 0, 5, 4, 6
            flipped_order = [3, 2, 1, 0, 5, 4, 6]
            temp = f_lmrk.copy()
            f_lmrk[:, :] = [temp[flipped_order[i], :] for i in range(0, len(flipped_order))]

        return {'img': f_img, 'bbox': sample['bbox'], 'lmarks': f_lmrk}

    @staticmethod
    def change_brightness(sample, bright_noise=0.5):
        # bright_noise: 0 is the brighset, 1 is original
        s_img = sample['img']
        s_lmrk = sample['lmarks']

        s_brightness = random.uniform((1-bright_noise), 1)
        s_img = ImageEnhance.Brightness(s_img).enhance(s_brightness)

        return {'img': s_img, 'bbox': sample['bbox'], 'lmarks': s_lmrk}

    @staticmethod
    def rescale_image(sample, output_size):
        s_img = np.asarray(sample['img'], dtype=np.float32)
        s_lmrk = sample['lmarks']

        # rescale the landmarks based on the resized image
        s_lmrk[:, 0] = s_lmrk[:, 0] * (output_size[0] / sample['img'].width)
        s_lmrk[:, 1] = s_lmrk[:, 1] * (output_size[1] / sample['img'].height)

        # rescale image from bbox to net input size
        s_img = cv2.resize(s_img, dsize=output_size, interpolation=cv2.INTER_CUBIC)

        return {'img': s_img, 'bbox': sample['bbox'], 'lmarks': s_lmrk}


if __name__ == "__main__":
    # print(help(Dataset))
    lfw_lab_dir = "./"
    lfw_image_dir = os.path.join(lfw_lab_dir, "lfw/")
    lfw_train_anno_path = os.path.join(lfw_lab_dir, "LFW_annotation_train.txt")

    #Alexnet
    alexnet_size = (225, 225)
    lfw_train = LFWDataset(lfw_image_dir, lfw_train_anno_path, n_augmented=1, net_size=alexnet_size, debug_mode=True)

    lfw_train.__getitem__(random.randint(0, 8000))
    print(len(lfw_train))