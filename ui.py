# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(840, 733)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.widget = QtWidgets.QWidget(self.tab)
        self.widget.setGeometry(QtCore.QRect(330, 270, 154, 25))
        self.widget.setObjectName("widget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.exe_reshape_PushBtn = QtWidgets.QPushButton(self.widget)
        self.exe_reshape_PushBtn.setEnabled(False)
        self.exe_reshape_PushBtn.setObjectName("exe_reshape_PushBtn")
        self.horizontalLayout_3.addWidget(self.exe_reshape_PushBtn)
        self.reshape_proc_indicator = QtWidgets.QCheckBox(self.widget)
        self.reshape_proc_indicator.setCheckable(True)
        self.reshape_proc_indicator.setChecked(False)
        self.reshape_proc_indicator.setObjectName("reshape_proc_indicator")
        self.horizontalLayout_3.addWidget(self.reshape_proc_indicator)
        self.widget1 = QtWidgets.QWidget(self.tab)
        self.widget1.setGeometry(QtCore.QRect(40, 140, 731, 25))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget1)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.load_xlsx_PushBtn = QtWidgets.QPushButton(self.widget1)
        self.load_xlsx_PushBtn.setObjectName("load_xlsx_PushBtn")
        self.horizontalLayout.addWidget(self.load_xlsx_PushBtn)
        self.show_file_path_LineEd = QtWidgets.QLineEdit(self.widget1)
        self.show_file_path_LineEd.setReadOnly(True)
        self.show_file_path_LineEd.setObjectName("show_file_path_LineEd")
        self.horizontalLayout.addWidget(self.show_file_path_LineEd)
        self.widget2 = QtWidgets.QWidget(self.tab)
        self.widget2.setGeometry(QtCore.QRect(40, 190, 301, 21))
        self.widget2.setObjectName("widget2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.nrows_per_group_LineEd = QtWidgets.QLineEdit(self.widget2)
        self.nrows_per_group_LineEd.setAlignment(QtCore.Qt.AlignCenter)
        self.nrows_per_group_LineEd.setObjectName("nrows_per_group_LineEd")
        self.horizontalLayout_2.addWidget(self.nrows_per_group_LineEd)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.showImg_Lb = QtWidgets.QLabel(self.tab_2)
        self.showImg_Lb.setGeometry(QtCore.QRect(20, 120, 781, 551))
        self.showImg_Lb.setText("")
        self.showImg_Lb.setObjectName("showImg_Lb")
        self.splitter = QtWidgets.QSplitter(self.tab_2)
        self.splitter.setGeometry(QtCore.QRect(30, 60, 751, 39))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.widget3 = QtWidgets.QWidget(self.splitter)
        self.widget3.setObjectName("widget3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget3)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.widget3)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.thresh_HSlider = QtWidgets.QSlider(self.widget3)
        self.thresh_HSlider.setAutoFillBackground(False)
        self.thresh_HSlider.setMaximum(255)
        self.thresh_HSlider.setOrientation(QtCore.Qt.Horizontal)
        self.thresh_HSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.thresh_HSlider.setTickInterval(10)
        self.thresh_HSlider.setObjectName("thresh_HSlider")
        self.verticalLayout_4.addWidget(self.thresh_HSlider)
        self.widget4 = QtWidgets.QWidget(self.splitter)
        self.widget4.setObjectName("widget4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget4)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.widget4)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.currentThresh_LineEdit = QtWidgets.QLineEdit(self.widget4)
        self.currentThresh_LineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.currentThresh_LineEdit.setObjectName("currentThresh_LineEdit")
        self.verticalLayout_2.addWidget(self.currentThresh_LineEdit)
        self.widget5 = QtWidgets.QWidget(self.splitter)
        self.widget5.setObjectName("widget5")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget5)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_5 = QtWidgets.QLabel(self.widget5)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.showPixArea_lineEdit = QtWidgets.QLineEdit(self.widget5)
        self.showPixArea_lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.showPixArea_lineEdit.setReadOnly(True)
        self.showPixArea_lineEdit.setObjectName("showPixArea_lineEdit")
        self.verticalLayout_3.addWidget(self.showPixArea_lineEdit)
        self.widget6 = QtWidgets.QWidget(self.tab_2)
        self.widget6.setGeometry(QtCore.QRect(30, 19, 751, 25))
        self.widget6.setObjectName("widget6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget6)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.load_img_PushBtn = QtWidgets.QPushButton(self.widget6)
        self.load_img_PushBtn.setObjectName("load_img_PushBtn")
        self.horizontalLayout_4.addWidget(self.load_img_PushBtn)
        self.showImgPath_lineEdit = QtWidgets.QLineEdit(self.widget6)
        self.showImgPath_lineEdit.setReadOnly(True)
        self.showImgPath_lineEdit.setObjectName("showImgPath_lineEdit")
        self.horizontalLayout_4.addWidget(self.showImgPath_lineEdit)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tabWidget.addTab(self.tab_3, "")
        self.verticalLayout.addWidget(self.tabWidget)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.exe_reshape_PushBtn.setText(_translate("Form", "一键转换"))
        self.reshape_proc_indicator.setText(_translate("Form", "转换完成"))
        self.label.setText(_translate("Form", "源Excel文件路径"))
        self.load_xlsx_PushBtn.setText(_translate("Form", "加载.xlsx文件"))
        self.label_2.setText(_translate("Form", "源Excel数据几行同属一组（每组容量必须相同）"))
        self.nrows_per_group_LineEd.setText(_translate("Form", "2"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Excel重规整"))
        self.label_3.setText(_translate("Form", "调整阈值"))
        self.label_4.setText(_translate("Form", "当前阈值"))
        self.currentThresh_LineEdit.setText(_translate("Form", "0"))
        self.label_5.setText(_translate("Form", "大于阈值的像素（黄色）面积"))
        self.load_img_PushBtn.setText(_translate("Form", "加载图片"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "从图像提取像素面积"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Form", "待新增..."))