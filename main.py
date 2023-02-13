#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/1/30 17:35
# @Author  : Mason TAO
# @Site    : 
# @File    : mainlogic_template.py
# @Software: PyCharm


import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from ui import Ui_Form

import pandas as pd
import os

# 这是一个测试
class MyMainWindow(QWidget, Ui_Form):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setup_ui()  # 初始化 UI 界面
        self.connect_signals()  # 绑定触发事件

        # 自定义一些实例属性，并初始化，后续可以通过槽函数更新
        # Tab：Excel重规整
        self.src_xlsx_path = None
        self.nrows_per_group = int(self.nrows_per_group_LineEd.text())
        # Tab：从图像提取像素面积
        self.src_img_path = None
        self.thresh = int(self.currentThresh_LineEdit.text())
        self.src_img_np = None

    def setup_ui(self):
        '''初始化 UI 界面，可在此处加入自定制（大部分可通过 Qt designer 完成）'''
        self.setupUi(self)  # 此实例方法是继承自 ui.ui 里的类 Ui_MainWindow，用以初始化 Qt designer 里设计的 UI 界面
        # Tab：Excel重规整
        self.reshape_proc_indicator.setChecked(False)
        self.reshape_proc_indicator.setEnabled(False)
        # Tab：从图像提取像素面积
        self.thresh_HSlider.setEnabled(False)
        self.thresh_HSlider.setValue(255)
        self.currentThresh_LineEdit.setEnabled(False)
        self.currentThresh_LineEdit.setText("255")

    def connect_signals(self):
        '''绑定信号-槽函数'''
        # Tab：Excel重规整
        self.load_xlsx_PushBtn.clicked.connect(self.openFileDialog4xlsx)
        self.nrows_per_group_LineEd.textChanged.connect(self.check_Reshape_executable)
        self.exe_reshape_PushBtn.clicked.connect(self.exeReshape)
        # Tab：从图像提取像素面积
        self.load_img_PushBtn.clicked.connect(self.openFileDialog4fig)
        self.thresh_HSlider.valueChanged.connect(self.threshold_from_QSlider_to_QLineEdit)
        self.currentThresh_LineEdit.editingFinished.connect(self.thresholding_from_QLineEdit_to_QSlider)

################################################# Tab：Excel重规整 ######################################################
    def openFileDialog4xlsx(self):
        self.src_xlsx_path, _ = QFileDialog.getOpenFileName(self, caption="选择Excel文件", directory="", filter="Excel文件 (*.xlsx)")
        self.show_file_path_LineEd.setText(self.src_xlsx_path)
        self.check_Reshape_executable()

    def check_Reshape_executable(self):
        if self.show_file_path_LineEd.text():
            if self.nrows_per_group_LineEd.text():
                try:
                    if int(self.nrows_per_group_LineEd.text()) >= 1:
                        # 允许点击“一键转换”
                        self.exe_reshape_PushBtn.setEnabled(True)
                        # 重置“工作状态”
                        self.reshape_proc_indicator.setChecked(False)
                    else:
                        self.exe_reshape_PushBtn.setEnabled(False)
                except:
                    self.exe_reshape_PushBtn.setEnabled(False)
            else:
                self.exe_reshape_PushBtn.setEnabled(False)
        else:
            self.exe_reshape_PushBtn.setEnabled(False)

    def exeReshape(self):
        # 实时更新参数 nrows_per_group
        self.nrows_per_group = int(self.nrows_per_group_LineEd.text())
        # 自动生成 dst_xlsx_path
        dir_path = os.path.dirname(self.src_xlsx_path)
        basename = os.path.basename(self.src_xlsx_path)
        name, ext = basename.rsplit(".", maxsplit=1)
        name = name + "_cvt"
        basename = ".".join([name, ext])
        dst_xlsx_path = os.path.join(dir_path, basename)
        # 读取 src_xlsx 进入内存
        df = pd.read_excel(self.src_xlsx_path, sheet_name=0, header=None)
        # 执行 reshape
        df = df.reset_index()
        df["index"] = df['index'] // self.nrows_per_group
        df_groupby = df.groupby("index")
        res = df_groupby.apply(self._agg_func).T
        # 将结果写入 dst_xlsx
        with pd.ExcelWriter(dst_xlsx_path) as exwt:
            res.to_excel(exwt, sheet_name='Sheet1', header=False, index=False)
            exwt.save()
        # 使用 QCheckBox 显示“完成转换”
        self.reshape_proc_indicator.setChecked(True)

    def _agg_func(self, df: pd.DataFrame):
        df = df.drop(columns="index")
        row_num, col_num = df.shape
        volumn = row_num * col_num
        res_list = []
        for c in range(col_num):
            for r in range(row_num):
                cell_value = df.iat[r, c]
                res_list.append(cell_value)
        se = pd.Series(res_list, index=range(volumn))
        return se

################################################# Tab：从图像提取像素面积 ##################################################
    def openFileDialog4fig(self):
        """读取 src_img_path 并打印出来，然后使用 QPixmap 将图像投射到 QLabel 上，最后使用 cv2.imread() 读取 NP 进入内存"""
        # 使用 QFileDialog 读取图片路径，并打印
        self.src_img_path, _ = QFileDialog.getOpenFileName(self, caption="选择图片", directory="",
                                               filter="Image File (*.jpg *.jpeg *.bmp *.png *.tiff)")
        self.showImgPath_lineEdit.setText(self.src_img_path)
        # 使用 QPixmap 将图片投射出来
        img_qpixmap = QPixmap(self.src_img_path).scaled(self.showImg_Lb.width(), self.showImg_Lb.height(),
                                                        Qt.KeepAspectRatio)
        self.showImg_Lb.setPixmap(img_qpixmap)
        # 使用 cv2.imread() 读取 NP 进入内存，如果是灰度图，读取为 1-D NP；如果是彩图，读取为 RGB NP
        src_img_np: np.ndarray = cv2.imread(self.src_img_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        if src_img_np.ndim == 2:
            self.src_img_np = src_img_np
        elif src_img_np.ndim == 3 and src_img_np.shape[2] == 3:
            self.src_img_np = cv2.cvtColor(src_img_np, cv2.COLOR_BGR2RGB)
        elif src_img_np.ndim == 3 and src_img_np.shape[2] == 4:
            self.src_img_np = cv2.cvtColor(src_img_np, cv2.COLOR_BGRA2RGB)
        # 重置和阈值有关的 HSlider 和 QLineEdit
        self.initialize_thresh()

    def initialize_thresh(self):
        self.thresh_HSlider.setEnabled(True)
        self.currentThresh_LineEdit.setEnabled(True)
        self.thresh_HSlider.setValue(255)
        self.currentThresh_LineEdit.setText("255")
        self.showPixArea_lineEdit.setText("")

    def threshold_from_QSlider_to_QLineEdit(self):
        # 联动 QLineEdit 的阈值显示，同时更新实例属性里的阈值
        self.thresh = int(self.thresh_HSlider.value())
        self.currentThresh_LineEdit.setText(str(self.thresh))
        self.projecting_mask()

    def thresholding_from_QLineEdit_to_QSlider(self):
        # 联动 QSlider 的阈值显示，同时更新实例属性里的阈值
        self.thresh = int(self.currentThresh_LineEdit.text())
        self.thresh_HSlider.setValue(self.thresh)
        self.projecting_mask()

    def projecting_mask(self):
        """判断 src_img_np 是单通道灰度图，还是三通道R图、G图、B图？
            单通道灰度图：
                1. 将 src_img_np 扩成3通道灰图
                2. 值作mask，并调制成黄色
                3. 使用 cv2.bitwise_or 叠加上述两图
            三通道单色彩图：
        """
        mask_np = None
        combined_output_rgb_np = None
        # 单通道灰度图
        if self.src_img_np.ndim == 2:
            # 生成 RGB 灰色图
            src_img_rgb_np = cv2.merge([self.src_img_np, self.src_img_np, self.src_img_np]).astype(np.uint8)
            # 生成 grayscale mask
            mask_np = np.where(self.src_img_np > self.thresh, 1, 0).astype(np.uint8)
            _rgb_npwhere_filter = cv2.merge([mask_np, mask_np, mask_np]).astype(np.uint8)
            thresh_tozero_np = (self.src_img_np * mask_np).astype(np.uint8)
            # 将 grayscale mask 变为黄色
            blue_mask_rgb_np = cv2.merge([np.zeros_like(thresh_tozero_np, dtype=np.uint8),
                                          np.zeros_like(thresh_tozero_np, dtype=np.uint8),
                                          thresh_tozero_np]).astype(np.uint8)
            blue_mask_hsv_np = cv2.cvtColor(blue_mask_rgb_np, cv2.COLOR_RGB2HSV).astype(np.uint8)
            h_np, s_np, v_np = cv2.split(blue_mask_hsv_np)
            yellow_mask_hsv_np = cv2.merge([h_np - 82, s_np, v_np]).astype(np.uint8)
            yellow_mask_rgb_np = cv2.cvtColor(yellow_mask_hsv_np, cv2.COLOR_HSV2RGB).astype(np.uint8)
            # 将 RGB 灰色图与黄色 grayscale mask 一起输出
            combined_output_rgb_np = np.where(_rgb_npwhere_filter, yellow_mask_rgb_np, src_img_rgb_np).astype(np.uint8)
        # 三通道彩图
        elif self.src_img_np.ndim == 3:
            # 判断主色
            r_np, g_np, b_np = cv2.split(self.src_img_np)
            major_color_index = np.argmax([np.sum(r_np), np.sum(g_np), np.sum(b_np)])
            major_color_np = (r_np, g_np, b_np)[major_color_index]
            # 对主色取mask，并掩膜，生成 grayscale mask
            mask_np = np.where(major_color_np > self.thresh, 1, 0).astype(np.uint8)
            _rgb_npwhere_filter = cv2.merge([mask_np, mask_np, mask_np]).astype(np.uint8)
            thresh_tozero_np = (major_color_np * mask_np).astype(np.uint8)
            # 将 grayscale mask 变为黄色
            blue_mask_rgb_np = cv2.merge([np.zeros_like(thresh_tozero_np, dtype=np.uint8),
                                          np.zeros_like(thresh_tozero_np, dtype=np.uint8),
                                          thresh_tozero_np]).astype(np.uint8)
            blue_mask_hsv_np = cv2.cvtColor(blue_mask_rgb_np, cv2.COLOR_RGB2HSV).astype(np.uint8)
            h_np, s_np, v_np = cv2.split(blue_mask_hsv_np)
            yellow_mask_hsv_np = cv2.merge([h_np - 82, s_np, v_np]).astype(np.uint8)
            yellow_mask_rgb_np = cv2.cvtColor(yellow_mask_hsv_np, cv2.COLOR_HSV2RGB).astype(np.uint8)
            # 将原图与黄色 grayscale mask 一起输出
            combined_output_rgb_np = np.where(_rgb_npwhere_filter, yellow_mask_rgb_np, self.src_img_np).astype(np.uint8)
        # 显示叠加图像
        width = combined_output_rgb_np.shape[1]
        height = combined_output_rgb_np.shape[0]
        img_qimage = QImage(combined_output_rgb_np, width, height, width * 3, QImage.Format_RGB888)
        img_qpixmap = QPixmap(img_qimage).scaled(self.showImg_Lb.width(), self.showImg_Lb.height(), Qt.KeepAspectRatio)
        self.showImg_Lb.setPixmap(img_qpixmap)
        # 计算存留的像素面积
        self.showPixArea_lineEdit.setText(str(np.sum(mask_np)))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())