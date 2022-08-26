# Create by Packetsss
# Personal use is allowed
# Commercial use is prohibited
import os
import sys
import cv2
import qimage2ndarray
import pathlib
from copy import deepcopy
from scripts import Images
import numpy as np
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


######## lib ##########

from os.path import join, isfile, isdir
from os import listdir
import argparse
from argparse import ArgumentParser
import numpy as np
import cv2
from skimage import color
import skimage.io

# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
setup_logger()

import torch
from tqdm import tqdm
from PIL import Image
import shutil
import glob
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import (batch_lab2rgb_transpose_mc, folder2vid, mkdir_if_not,
                        save_frames, tensor_lab2rgb, uncenter_l)
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--frame_propagate", default=False, type=bool, help="propagation mode, , please check the paper"
)
parser.add_argument("--image_size", type=int, default=[216 * 2, 384 * 2], help="the image size, eg. [216,384]")
parser.add_argument("--cuda", action="store_false")
parser.add_argument("--gpu_ids", type=str, default="0", help="separate by comma")
parser.add_argument("--clip_path", type=str, default="./sample_videos/frames", help="path of input clips")
parser.add_argument("--ref_path", type=str, default="./sample_videos/ref/taxi", help="path of refernce images")
parser.add_argument("--output_path", type=str, default="./sample_videos/output", help="path of output clips")
opt = parser.parse_args()
opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")]
cudnn.benchmark = True
print("running on GPU", opt.gpu_ids)


def colorize_video(opt, input_path, reference_file, output_path, nonlocal_net, colornet, vggnet):
    # parameters for wls filter
    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4

    # processing folders
    mkdir_if_not(output_path)
    files = glob.glob(output_path + "*")
    print("processing the folder:", input_path)
    path, dirs, filenames = os.walk(input_path).__next__()
    file_count = len(filenames)
    filenames.sort(key=lambda f: int("".join(filter(str.isdigit, f) or -1)))

    # NOTE: resize frames to 216*384
    # transform = transforms.Compose(
    #     [CenterPad(opt.image_size), transform_lib.CenterCrop(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
    # )

    transform = transforms.Compose(
        [transform_lib.Resize(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
    )
    
    # if frame propagation: use the first frame as reference
    # otherwise, use the specified reference image
    ref_name = input_path + filenames[0] if opt.frame_propagate else reference_file
    print("reference name:", ref_name)
    frame_ref = Image.open(ref_name)
    frame_ref = frame_ref.convert("RGB") ####

    total_time = 0
    I_last_lab_predict = None

    IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()

    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
    IB_l = IB_lab[:, 0:1, :, :]
    IB_ab = IB_lab[:, 1:3, :, :]
    with torch.no_grad():
      I_reference_lab = IB_lab
      I_reference_l = I_reference_lab[:, 0:1, :, :]
      I_reference_ab = I_reference_lab[:, 1:3, :, :]
      I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
      features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

    for index, frame_name in enumerate(tqdm(filenames)):
        
        frame1 = Image.open(os.path.join(input_path, frame_name))
        frame1 = frame1.convert("RGB") ####
        
        IA_lab_large = transform(frame1).unsqueeze(0).cuda()

        
        IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")

        IA_l = IA_lab[:, 0:1, :, :]
        IA_ab = IA_lab[:, 1:3, :, :]

        
        if I_last_lab_predict is None:
            if opt.frame_propagate:
                I_last_lab_predict = IB_lab
            else:
                I_last_lab_predict = torch.zeros_like(IA_lab).cuda()

        # start the frame colorization
        with torch.no_grad():
            I_current_lab = IA_lab
            I_current_ab_predict, I_current_nonlocal_lab_predict, features_current_gray = frame_colorization(
                I_current_lab,
                I_reference_lab,
                I_last_lab_predict,
                features_B,
                vggnet,
                nonlocal_net,
                colornet,
                feature_noise=0,
                temperature=1e-10,
            )
            I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

        # upsampling
        curr_bs_l = IA_lab_large[:, 0:1, :, :]
        curr_predict = (
            torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
        )

        # filtering
        if wls_filter_on:
            guide_image = uncenter_l(curr_bs_l) * 255 / 100
            wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
            )
            curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
            curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
            curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
            curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
            curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
        else:
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

        # save the frames
        # save_frames(IA_predict_rgb, output_path, index)
        save_frames(IA_predict_rgb, output_path, image_name = frame_name)

clip_name = opt.clip_path.split("/")[-1]
refs = os.listdir(opt.ref_path)
refs.sort()

nonlocal_net = WarpNet(1)
colornet = ColorVidNet(7)
vggnet = VGG19_pytorch()
vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
for param in vggnet.parameters():
    param.requires_grad = False

nonlocal_test_path = os.path.join("checkpoints/", "video_moredata_l1/nonlocal_net_iter_76000.pth")
color_test_path = os.path.join("checkpoints/", "video_moredata_l1/colornet_iter_76000.pth")
print("succesfully load nonlocal model: ", nonlocal_test_path)
print("succesfully load color model: ", color_test_path)
nonlocal_net.load_state_dict(torch.load(nonlocal_test_path))
colornet.load_state_dict(torch.load(color_test_path))

nonlocal_net.eval()
colornet.eval()
vggnet.eval()
nonlocal_net.cuda()
colornet.cuda()
vggnet.cuda()

#####################

class Filter(QWidget):
    def __init__(self, main):
        super().__init__()
        uic.loadUi(f"{pathlib.Path(__file__).parent.absolute()}/ui/filter_frame.ui", self)
        self.img_class, self.update_img, self.base_frame, self.vbox = \
            main.img_class, main.update_img, main.base_frame, main.vbox

        self.frame = self.findChild(QFrame, "frame")
        self.contrast_btn = self.findChild(QPushButton, "contrast_btn")
        self.sharpen_btn = self.findChild(QPushButton, "sharpen_btn")
        self.cartoon_btn = self.findChild(QPushButton, "cartoon_btn")
        self.cartoon_btn1 = self.findChild(QPushButton, "cartoon_btn2")
        self.invert_btn = self.findChild(QPushButton, "invert_btn")
        self.bypass_btn = self.findChild(QPushButton, "bypass_btn")


        self.y_btn = self.findChild(QPushButton, "y_btn")
        self.y_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/check.png"))
        self.y_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.y_btn.setIconSize(QSize(60, 60))
        self.n_btn = self.findChild(QPushButton, "n_btn")
        self.n_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/cross.png"))
        self.n_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.n_btn.setIconSize(QSize(60, 60))

        self.y_btn.clicked.connect(lambda _: self.click_y())
        self.n_btn.clicked.connect(lambda _: self.click_n())
        self.contrast_btn.clicked.connect(lambda _: self.click_contrast())
        self.sharpen_btn.clicked.connect(lambda _: self.click_sharpen())
        self.cartoon_btn.clicked.connect(lambda _: self.click_cartoon())
        self.cartoon_btn1.clicked.connect(lambda _: self.click_cartoon1())
        self.invert_btn.clicked.connect(lambda _: self.click_invert())
        self.bypass_btn.clicked.connect(lambda _: self.click_bypass())



    def click_contrast(self):
        self.img_class.auto_contrast()
        self.update_img()
        self.contrast_btn.clicked.disconnect()

    def click_sharpen(self):
        self.img_class.auto_sharpen()
        self.update_img()
        self.sharpen_btn.clicked.disconnect()

    def click_cartoon(self):
        self.img_class.auto_cartoon()
        self.update_img()
        self.cartoon_btn.clicked.disconnect()

    def click_cartoon1(self):
        self.img_class.auto_cartoon(1)
        self.update_img()
        self.cartoon_btn1.clicked.disconnect()

    def click_invert(self):
        self.img_class.auto_invert()
        self.update_img()
        self.invert_btn.clicked.disconnect()

    def click_bypass(self):
        self.img_class.bypass_censorship()
        self.update_img()
        self.bypass_btn.clicked.disconnect()

    def click_y(self):
        self.frame.setParent(None)
        self.img_class.img_copy = deepcopy(self.img_class.img)
        self.img_class.grand_img_copy = deepcopy(self.img_class.img)
        self.vbox.addWidget(self.base_frame)

    def click_n(self):
        if not np.array_equal(self.img_class.grand_img_copy, self.img_class.img):
            msg = QMessageBox.question(self, "Cancel edits", "Confirm to discard all the changes?   ",
                                       QMessageBox.Yes | QMessageBox.No)
            if msg != QMessageBox.Yes:
                return False

        self.frame.setParent(None)
        self.img_class.grand_reset()
        self.update_img()
        self.vbox.addWidget(self.base_frame)


class Adjust(QWidget):
    def __init__(self, main):
        super().__init__()
        uic.loadUi(f"{pathlib.Path(__file__).parent.absolute()}/ui/adjust_frame.ui", self)
        self.get_zoom_factor = main.get_zoom_factor

        self.img_class, self.update_img, self.base_frame = main.img_class, main.update_img, main.base_frame
        self.rb, self.vbox, self.flip, self.zoom_factor = main.rb, main.vbox, main.flip, main.zoom_factor
        self.zoom_moment, self.slider, self.gv, self.vbox1 = main.zoom_moment, main.slider, main.gv, main.vbox1
        self.start_detect = False

        self.frame = self.findChild(QFrame, "frame")
        self.crop_btn = self.findChild(QPushButton, "crop_btn")
        self.rotate_btn = self.findChild(QPushButton, "rotate_btn")
        self.brightness_btn = self.findChild(QPushButton, "brightness_btn")
        self.contrast_btn = self.findChild(QPushButton, "contrast_btn")
        self.saturation_btn = self.findChild(QPushButton, "saturation_btn")
        self.mask_btn = self.findChild(QPushButton, "mask_btn")

        self.y_btn = self.findChild(QPushButton, "y_btn")
        self.y_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/check.png"))
        self.y_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.y_btn.setIconSize(QSize(60, 60))
        self.n_btn = self.findChild(QPushButton, "n_btn")
        self.n_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/cross.png"))
        self.n_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.n_btn.setIconSize(QSize(60, 60))

        self.y_btn.clicked.connect(lambda _: self.click_y())
        self.n_btn.clicked.connect(lambda _: self.click_n())
        self.crop_btn.clicked.connect(lambda _: self.click_crop())
        self.rotate_btn.clicked.connect(lambda _: self.click_crop(rotate=True))
        self.brightness_btn.clicked.connect(lambda _: self.click_brightness())
        self.contrast_btn.clicked.connect(lambda _: self.click_brightness(mode=1))
        self.saturation_btn.clicked.connect(lambda _: self.click_brightness(mode=2))
        self.mask_btn.clicked.connect(lambda _: self.click_brightness(mode=3))

    def click_crop(self, rotate=False):
        def click_y1():
            self.rb.update_dim()
            if rotate:
                self.img_class.rotate_img(self.rotate_value, crop=True, flip=self.flip)
                self.img_class.crop_img(int(self.rb.top * 2 / self.zoom_factor),
                                        int(self.rb.bottom * 2 / self.zoom_factor),
                                        int(self.rb.left * 2 / self.zoom_factor),
                                        int(self.rb.right * 2 / self.zoom_factor))
            else:
                self.img_class.reset(self.flip)
                self.img_class.crop_img(int(self.rb.top / self.zoom_factor), int(self.rb.bottom / self.zoom_factor),
                                        int(self.rb.left // self.zoom_factor), int(self.rb.right // self.zoom_factor))

            self.update_img()
            self.zoom_moment = False

            self.img_class.img_copy = deepcopy(self.img_class.img)
            self.slider.setParent(None)
            self.slider.valueChanged.disconnect()
            crop_frame.frame.setParent(None)
            self.vbox.addWidget(self.frame)
            self.rb.close()

        def click_n1():
            if not np.array_equal(img_copy, self.img_class.img):
                msg = QMessageBox.question(self, "Cancel edits", "Confirm to discard all the changes?   ",
                                           QMessageBox.Yes | QMessageBox.No)
                if msg != QMessageBox.Yes:
                    return False

            self.img_class.reset()
            self.update_img()
            self.zoom_moment = False

            self.slider.setParent(None)
            self.slider.valueChanged.disconnect()
            crop_frame.frame.setParent(None)
            self.vbox.addWidget(self.frame)
            self.rb.close()

        def change_slide():
            self.rotate_value = self.slider.value()
            self.slider.setValue(self.rotate_value)

            self.img_class.rotate_img(self.rotate_value)

            self.rb.setGeometry(int(self.img_class.left * self.zoom_factor), int(self.img_class.top * self.zoom_factor),
                                int((self.img_class.right - self.img_class.left) * self.zoom_factor),
                                int((self.img_class.bottom - self.img_class.top) * self.zoom_factor))

            self.rb.update_dim()
            self.update_img(True)

        def add_90():
            if self.rotate_value <= 270:
                self.rotate_value += 90
            else:
                self.rotate_value = 360
            self.slider.setValue(self.rotate_value)
            change_slide()

        def subtract_90():
            if self.rotate_value >= 90:
                self.rotate_value -= 90
            else:
                self.rotate_value = 0
            self.slider.setValue(self.rotate_value)
            change_slide()

        def vertical_flip():
            nonlocal vflip_ct
            self.img_class.img = cv2.flip(self.img_class.img, 0)
            if rotate:
                self.update_img(True)
            else:
                self.update_img()
            vflip_ct += 1
            self.flip[0] = vflip_ct % 2 == 1

        def horizontal_flip():
            nonlocal hflip_ct
            self.img_class.img = cv2.flip(self.img_class.img, 1)
            if rotate:
                self.update_img(True)
            else:
                self.update_img()
            hflip_ct += 1
            self.flip[1] = hflip_ct % 2 == 1

        crop_frame = Crop()
        crop_frame.n_btn.clicked.connect(click_n1)
        crop_frame.y_btn.clicked.connect(click_y1)
        crop_frame.rotate.clicked.connect(add_90)
        crop_frame.rotatect.clicked.connect(subtract_90)
        crop_frame.vflip.clicked.connect(vertical_flip)
        crop_frame.hflip.clicked.connect(horizontal_flip)
        self.flip = [False, False]
        vflip_ct = 2
        hflip_ct = 2

        self.frame.setParent(None)
        
        self.vbox.addWidget(crop_frame.frame)
        self.zoom_factor = self.get_zoom_factor()

        self.rb = ResizableRubberBand(self)
        self.rb.setGeometry(0, 0, self.img_class.img.shape[1] * self.zoom_factor,
                            self.img_class.img.shape[0] * self.zoom_factor)
        self.img_class.change_b_c(beta=-40)
        self.slider.valueChanged.connect(change_slide)


        if not rotate:
            self.update_img()
            crop_frame.rotate.setParent(None)
            crop_frame.rotatect.setParent(None)
        else:
            self.vbox1.insertWidget(1, self.slider)
            self.slider.setRange(0, 360)
            self.slider.setValue(0)
            self.zoom_moment = True
            self.img_class.rotate_img(0)
            self.rb.setGeometry(0, 0, int(self.img_class.img.shape[1] * self.zoom_factor),
                                int(self.img_class.img.shape[0] * self.zoom_factor))
            self.update_img(True)

        img_copy = deepcopy(self.img_class.img)

    def click_brightness(self, mode=0):
        def click_y1():
            self.img_class.img_copy = deepcopy(self.img_class.img)
            if mode != 3:
                self.slider.setParent(None)
                self.slider.valueChanged.disconnect()
            brightness_frame.frame.setParent(None)
            self.vbox.addWidget(self.frame)

        def click_n1():
            if not np.array_equal(self.img_class.img_copy, self.img_class.img):
                msg = QMessageBox.question(self, "Cancel edits", "Confirm to discard all the changes?   ",
                                           QMessageBox.Yes | QMessageBox.No)
                if msg != QMessageBox.Yes:
                    return False
            self.img_class.reset()
            self.update_img()

            if mode != 3:
                self.slider.setParent(None)
                self.slider.valueChanged.disconnect()
            brightness_frame.frame.setParent(None)
            self.vbox.addWidget(self.frame)

        def change_slide():
            self.brightness_value = self.slider.value()
            self.img_class.reset()
            self.img_class.change_b_c(beta=self.brightness_value)
            self.update_img()

        def change_slide_contr():
            self.contrast_value = self.slider.value() / 100
            self.img_class.reset()
            self.img_class.change_b_c(alpha=self.contrast_value)
            self.update_img()

        def change_slide_sat():
            self.saturation_value = self.slider.value() / 250
            self.img_class.reset()
            self.img_class.change_b_c(alpha=self.saturation_value)
            self.update_img()

        def color_dialog():
            color = QColorDialog.getColor()
            self.img_class.remove_color(color.name())
            self.update_img()

        brightness_frame = Brightness()
        brightness_frame.y_btn.clicked.connect(click_y1)
        brightness_frame.n_btn.clicked.connect(click_n1)

        self.frame.setParent(None)
        self.vbox.addWidget(brightness_frame.frame)

        if mode == 1:
            self.vbox1.insertWidget(1, self.slider)
            self.slider.setRange(0, 300)
            self.slider.setValue(100)
            self.slider.valueChanged.connect(change_slide_contr)
        elif mode == 2:
            self.vbox1.insertWidget(1, self.slider)
            self.slider.setRange(0, 1000)
            self.slider.setValue(250)
            self.slider.valueChanged.connect(change_slide_sat)
        elif mode == 3:
            btnn = QPushButton("Select color", brightness_frame)
            btnn.setFont(QFont("Neue Haas Grotesk Text Pro Medi", 14))
            btnn.setStyleSheet("QPushButton{border: 0px solid;}")
            btnn.setMaximumHeight(50)
            btnn.clicked.connect(color_dialog)
            brightness_frame.vbox2.insertWidget(0, btnn)
        else:
            self.vbox1.insertWidget(1, self.slider)
            self.slider.setRange(-120, 160)
            self.slider.setValue(0)
            self.slider.valueChanged.connect(change_slide)

    def click_y(self):
        self.start_detect = False
        self.frame.setParent(None)
        self.img_class.img_copy = deepcopy(self.img_class.img)
        self.img_class.grand_img_copy = deepcopy(self.img_class.img)
        self.vbox.addWidget(self.base_frame)

    def click_n(self):
        if not np.array_equal(self.img_class.grand_img_copy, self.img_class.img):
            msg = QMessageBox.question(self, "Cancel edits", "Confirm to discard all the changes?   ",
                                       QMessageBox.Yes | QMessageBox.No)
            if msg != QMessageBox.Yes:
                return False

        self.start_detect = False
        self.frame.setParent(None)
        self.img_class.grand_reset()
        self.update_img()
        self.vbox.addWidget(self.base_frame)


class Crop(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi(f"{pathlib.Path(__file__).parent.absolute()}/ui/crop_btn.ui", self)

        self.frame = self.findChild(QFrame, "frame")
        self.y_btn = self.findChild(QPushButton, "y_btn")
        self.y_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/check.png"))
        self.y_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.y_btn.setIconSize(QSize(70, 70))
        self.n_btn = self.findChild(QPushButton, "n_btn")
        self.n_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/cross.png"))
        self.n_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.n_btn.setIconSize(QSize(70, 70))

        self.rotate = self.findChild(QPushButton, "rotate")
        self.rotate.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/rotate90.png"))
        self.rotate.setStyleSheet("QPushButton{border: 0px solid;}")
        self.rotate.setIconSize(QSize(50, 50))
        self.rotatect = self.findChild(QPushButton, "rotatect")
        self.rotatect.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/rotatect90.png"))
        self.rotatect.setStyleSheet("QPushButton{border: 0px solid;}")
        self.rotatect.setIconSize(QSize(50, 50))

        self.vflip = self.findChild(QPushButton, "vflip")
        self.vflip.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/vflip.png"))
        self.vflip.setStyleSheet("QPushButton{border: 0px solid;}")
        self.vflip.setIconSize(QSize(50, 50))
        self.hflip = self.findChild(QPushButton, "hflip")
        self.hflip.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/hflip.png"))
        self.hflip.setStyleSheet("QPushButton{border: 0px solid;}")
        self.hflip.setIconSize(QSize(50, 50))


class Brightness(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi(f"{pathlib.Path(__file__).parent.absolute()}/ui/brightness_btn.ui", self)

        self.frame = self.findChild(QFrame, "frame")
        self.vbox2 = self.findChild(QVBoxLayout, "vbox2")
        self.y_btn = self.findChild(QPushButton, "y_btn")
        self.y_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/check.png"))
        self.y_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.y_btn.setIconSize(QSize(70, 70))
        self.n_btn = self.findChild(QPushButton, "n_btn")
        self.n_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/cross.png"))
        self.n_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.n_btn.setIconSize(QSize(70, 70))

        self.pten = self.findChild(QPushButton, "pten")
        self.pten.setStyleSheet("QPushButton{border: 0px solid;}")
        self.mten = self.findChild(QPushButton, "mten")
        self.mten.setStyleSheet("QPushButton{border: 0px solid;}")



class Ai(QWidget):
    def __init__(self, main):
        super().__init__()
        uic.loadUi(f"{pathlib.Path(__file__).parent.absolute()}/ui/ai_frame.ui", self)

        # self.get_zoom_factor = main.get_zoom_factor
        self.img_class, self.update_img, self.base_frame, self.rb, self.vbox, self.zoom_factor = \
            main.img_class, main.update_img, main.base_frame, main.rb, main.vbox, main.zoom_factor

        self.main = main
        self.main.scene = main.scene

        self.frame = self.findChild(QFrame, "frame")
        self.colorze_btn = self.findChild(QPushButton, "colorize_btn")
        self.colorze_btn.clicked.connect(lambda _: self.click_colorize())

        self.get_ref_btn = self.findChild(QPushButton, "get_ref_btn")
        self.get_ref_btn.clicked.connect(lambda _: self.click_get_ref())

        self.face_btn = self.findChild(QPushButton, "face_btn")
        self.face_btn.clicked.connect(lambda _: self.click_face())
        self.face_counter, self.face_cord = 0, None

        self.y_btn = self.findChild(QPushButton, "y_btn")
        self.y_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/check.png"))
        self.y_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.y_btn.setIconSize(QSize(60, 60))
        self.n_btn = self.findChild(QPushButton, "n_btn")
        self.n_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/cross.png"))
        self.n_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.n_btn.setIconSize(QSize(60, 60))

        self.y_btn.clicked.connect(self.click_y)
        self.n_btn.clicked.connect(self.click_n)


    def click_colorize(self):
        for ref_name in refs:
            try:
                colorize_video(
                    opt,
                    opt.clip_path,
                    os.path.join(opt.ref_path, ref_name),
                    os.path.join(opt.output_path, clip_name + "_" + ref_name.split(".")[0]),
                    nonlocal_net,
                    colornet,
                    vggnet,
                )
            except Exception as error:
                print("error when colorizing the video " + ref_name)
                print(error)

        self.pixmap = QPixmap("./sample_videos/output/frames_ref/target_0.png")
        self.converted_img = self.main.scene.addPixmap(self.pixmap)

        
    def click_get_ref(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose Image File", "",
                                                "Image Files (*.jpg *.png *.jpeg *.ico);;All Files (*)")
        if files:
            self.files = files
            self.img_list, self.rb = [], None

            for f in files:
                self.img_list.append(Images(f))
                print(type(Images(f)))
                print(Images(f))

            self.img_class = self.img_list[0]
            self.img = QPixmap(qimage2ndarray.array2qimage(cv2.cvtColor(self.img_class.img, cv2.COLOR_BGR2RGB)))
            self.img.save('./sample_videos/ref/taxi/ref.jpg')
            self.img = self.img.scaled(250,250)
            self.scene_img = self.main.scene.addPixmap(self.img)
        
        # os.system(f"python segment.py --test_img_dir {self.base_frame}") # 시작하자마자 segment 

    def click_face(self):
        face_frame = Face(self)
        self.frame.setParent(None)
        self.vbox.addWidget(face_frame.frame)

    def click_y(self):
        self.frame.setParent(None)
        self.img_class.img_copy = deepcopy(self.img_class.img)
        self.img_class.grand_img_copy = deepcopy(self.img_class.img)
        self.vbox.addWidget(self.base_frame)
        self.rb.close()

    def click_n(self):
        if not np.array_equal(self.img_class.grand_img_copy, self.img_class.img):
            msg = QMessageBox.question(self, "Cancel edits", "Confirm to discard all the changes?   ",
                                       QMessageBox.Yes | QMessageBox.No)
            if msg != QMessageBox.Yes:
                return False

        self.frame.setParent(None)
        self.img_class.grand_reset()
        self.update_img()
        self.vbox.addWidget(self.base_frame)
        self.rb.close()



class Face(QWidget):
    def __init__(self, ai_class):
        super().__init__()
        uic.loadUi(f"{pathlib.Path(__file__).parent.absolute()}/ui/face_btn.ui", self)

        self.img_class, self.update_img, self.base_frame, self.rb, self.vbox = \
            ai_class.img_class, ai_class.update_img, ai_class.base_frame, ai_class.rb, ai_class.vbox
        self.frame, self.ai_frame = self.findChild(QFrame, "frame"), ai_class.frame

        self.next_btn = self.findChild(QPushButton, "next_btn")
        self.next_btn.clicked.connect(lambda _: self.click_next())
        self.face_counter, self.face_cord = 0, None

        self.y_btn = self.findChild(QPushButton, "y_btn")
        self.y_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/check.png"))
        self.y_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.y_btn.setIconSize(QSize(60, 60))
        self.n_btn = self.findChild(QPushButton, "n_btn")
        self.n_btn.setIcon(QIcon(f"{pathlib.Path(__file__).parent.absolute()}/icon/cross.png"))
        self.n_btn.setStyleSheet("QPushButton{border: 0px solid;}")
        self.n_btn.setIconSize(QSize(60, 60))
        self.y_btn.clicked.connect(self.click_y)
        self.n_btn.clicked.connect(self.click_n)

    def click_next(self):
        if self.face_cord is None:
            self.face_cord = np.array(self.img_class.detect_face())

        if not len(self.face_cord):
            return False

        if self.face_counter == len(self.face_cord):
            self.face_counter = 0

        face = self.face_cord[self.face_counter]
        self.rb.setGeometry(face[0], face[1], face[2], face[3])

        self.update_img()
        self.face_counter += 1

    def click_y(self):
        self.frame.setParent(None)
        self.img_class.img_copy = deepcopy(self.img_class.img)
        self.vbox.addWidget(self.ai_frame)

    def click_n(self):
        if not np.array_equal(self.img_class.grand_img_copy, self.img_class.img):
            msg = QMessageBox.question(self, "Cancel edits", "Confirm to discard all the changes?   ",
                                       QMessageBox.Yes | QMessageBox.No)
            if msg != QMessageBox.Yes:
                return False

        self.frame.setParent(None)
        self.img_class.reset()
        self.update_img()
        self.vbox.addWidget(self.ai_frame)

class ResizableRubberBand(QWidget):
    def __init__(self, main):
        super(ResizableRubberBand, self).__init__(main.gv)
        self.get_zoom_factor = main.get_zoom_factor

        self.img_class, self.update, self.zoom_factor = main.img_class, main.update, main.zoom_factor
        self.draggable, self.mousePressPos, self.mouseMovePos = True, None, None
        self.left, self.right, self.top, self.bottom = None, None, None, None
        self.borderRadius = 0

        self.setWindowFlags(Qt.SubWindow)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QSizeGrip(self), 0, Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(QSizeGrip(self), 0, Qt.AlignRight | Qt.AlignBottom)

        self._band = QRubberBand(QRubberBand.Rectangle, self)
        self._band.show()
        self.show()

    def update_dim(self):
        self.left, self.top = self.pos().x(), self.pos().y()
        self.right, self.bottom = self._band.width() + self.left, self._band.height() + self.top

    def resizeEvent(self, event):
        try:
            self.left, self.top = self.pos().x(), self.pos().y()
            self.right, self.bottom = self._band.width() + self.left, self._band.height() + self.top
        except:
            pass
        self._band.resize(self.size())

    def paintEvent(self, event):
        # Get current window size
        window_size = self.size()
        qp = QPainter(self)
        qp.drawRoundedRect(0, 0, window_size.width(), window_size.height(), self.borderRadius, self.borderRadius)

    def mousePressEvent(self, event):
        self.zoom_factor = self.get_zoom_factor()
        if self.draggable and event.button() == Qt.LeftButton:
            self.mousePressPos = event.globalPos()  # global
            self.mouseMovePos = event.globalPos() - self.pos()  # local

    def mouseMoveEvent(self, event):
        if self.draggable and event.buttons() & Qt.LeftButton:
            if self.right <= int(self.img_class.img.shape[1] * self.zoom_factor) and self.bottom <= \
                    int(self.img_class.img.shape[0] * self.zoom_factor) and self.left >= 0 and self.top >= 0:
                globalPos = event.globalPos()
                diff = globalPos - self.mouseMovePos
                self.move(diff)  # move window
                self.mouseMovePos = globalPos - self.pos()

            self.left, self.top = self.pos().x(), self.pos().y()
            self.right, self.bottom = self._band.width() + self.left, self._band.height() + self.top

    def mouseReleaseEvent(self, event):
        if self.mousePressPos is not None:
            if event.button() == Qt.LeftButton:
                self.mousePressPos = None

        if self.left < 0:
            self.left = 0
            self.move(0, self.top)
        if self.right > int(self.img_class.img.shape[1] * self.zoom_factor):
            self.left = int(self.img_class.img.shape[1] * self.zoom_factor) - self._band.width()
            self.move(self.left, self.top)
        if self.bottom > int(self.img_class.img.shape[0] * self.zoom_factor):
            self.top = int(self.img_class.img.shape[0] * self.zoom_factor) - self._band.height()
            self.move(self.left, self.top)
        if self.top < 0:
            self.top = 0
            self.move(self.left, 0)

