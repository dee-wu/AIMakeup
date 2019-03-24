# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Makeup_fpp1.ui'
#
# Created: Tue Jun 19 16:36:08 2018
#      by: PyQt4 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!

import sys, os
import numpy as np
import cv2
from PyQt4 import QtCore, QtGui
from getLandmarkAndBeautify import Beautify

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        self.window = MainWindow
        self.setupUi(self.window)

        # 控件组
        self.group = [self.pushButton, self.pushButton_2, self.pushButton_3,
                      self.pushButton_6, self.pushButton_7, self.pushButton_8, self.horizontalSlider,
                      self.horizontalSlider_2, self.horizontalSlider_3]

        # 批量设置状态
        self.func_set_statu(self.group, False)

        self.path_img = ''
        self.func_set_connect()

    def func_set_connect(self):
        """
        设置程序逻辑
        """
        self.pushButton_4.clicked.connect(self.func_open_img)

        self.pushButton.clicked.connect(self.func_smoothing)
        self.pushButton_2.clicked.connect(self.func_glitting)
        self.pushButton_3.clicked.connect(self.func_whitening)

        self.pushButton_6.clicked.connect(self.func_reset)
        self.pushButton_7.clicked.connect(self.func_view_compare)
        self.pushButton_8.clicked.connect(self.func_save)

    def func_open_img(self):
        """
        打开图片
        """
        self.path_img = QtGui.QFileDialog.getOpenFileName(self.centralwidget, 'open image file', './',
                                                          'Image Files(*.png *.jpg *.bmp)')
        if self.path_img and os.path.exists(self.path_img):
            self.path_img = unicode(self.path_img)
            self.beauty = Beautify(self.path_img)
            self.img, self.faces = self.beauty.read_and_mark(self.path_img)
            self.im_ori, self.previous_bgr = self.img.copy(), self.img.copy()
            self.func_set_statu(self.group, True)
            self.func_set_img()

    def func_qimg(self, cvImg):
        """
        将opencv的图片转换为QImage
        """
        height, width, channel = cvImg.shape
        byte_per_line = 3 * width
        return QtGui.QImage(cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB).data, width, height, byte_per_line,
                            QtGui.QImage.Format_RGB888)

    def func_set_img(self):
        """
        显示pixmap
        """
        pixmap = QtGui.QPixmap.fromImage(self.func_qimg(self.img))
        self.label.setPixmap(pixmap)

    def func_set_statu(self, group, value):
        """
        批量设置状态
        """
        [item.setEnabled(value) for item in group]

    def func_reset(self):
        """
        重置为原始图片
        """
        self.img[:] = self.im_ori[:]
        self.func_set_img()

    def func_view_compare(self):
        cv2.imshow('Compare', np.concatenate([self.im_ori, self.img], 1))
        cv2.waitKey()

    def func_mapfaces(self, detail, value):
        """
        对每张脸进行迭代操作
        """
        self.previous_bgr[:] = self.img[:]
        for face in self.faces[self.path_img]:
            detail(face, value)
        self.func_set_img()

    def func_whitening(self):
        print(self.horizontalSlider_2.value())
        value = min(1, max(self.horizontalSlider_2.value() / 500.0, 0))
        print(value)

        def detail(face, value):
            face.hsv_operating(value, operating=True)

        self.func_mapfaces(detail, value)

    def func_glitting(self):
        value = min(1, max(self.horizontalSlider_3.value() / 200.0, 0))
        print(value)

        def detail(face, value):
            face.organs['mouth_upper'].hsv_operating(value, operating=False)
            face.organs['mouth_lower'].hsv_operating(value, operating=False)

        self.func_mapfaces(detail, value)

    def func_smoothing(self):
        value = min(1, max(self.horizontalSlider.value() / 100.0, 0))
        print(value)

        def detail(face, value):
            face.smoothing(value)
            face.organs['left_eye'].smoothing(value * 2 / 5)
            face.organs['right_eye'].smoothing(value * 2 / 5)
            face.organs['left_eyebrow'].smoothing(value * 2 / 7)
            face.organs['right_eyebrow'].smoothing(value * 2 / 7)
            face.organs['nose'].smoothing(value * 2 / 5)
            face.organs['forehead'].smoothing(value * 3 / 2)
            face.organs['mouth_upper'].smoothing(value)
            face.organs['mouth_lower'].smoothing(value)

        self.func_mapfaces(detail, value)

    def func_save(self):
        output_path = QtGui.QFileDialog.getSaveFileName(self.centralwidget, '选择保存位置', './',
                                                        'Image Files(*.png *.jpg *.bmp)')
        output_path = unicode(output_path)
        if output_path:
            self.saveImg(output_path, self.img)

    def saveImg(self, output_path, output_im):
        """
        保存图片
        """
        cv2.imencode('.jpg', output_im)[1].tofile(output_path)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(601, 829)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 581, 201))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.gridLayout_3 = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_3.setMargin(0)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.pushButton_4 = QtGui.QPushButton(self.gridLayoutWidget)
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.gridLayout_3.addWidget(self.pushButton_4, 0, 0, 1, 1)
        self.pushButton_8 = QtGui.QPushButton(self.gridLayoutWidget)
        self.pushButton_8.setObjectName(_fromUtf8("pushButton_8"))
        self.gridLayout_3.addWidget(self.pushButton_8, 4, 4, 1, 1)
        self.pushButton_6 = QtGui.QPushButton(self.gridLayoutWidget)
        self.pushButton_6.setObjectName(_fromUtf8("pushButton_6"))
        self.gridLayout_3.addWidget(self.pushButton_6, 4, 2, 1, 1)
        self.pushButton_7 = QtGui.QPushButton(self.gridLayoutWidget)
        self.pushButton_7.setObjectName(_fromUtf8("pushButton_7"))
        self.gridLayout_3.addWidget(self.pushButton_7, 4, 3, 1, 1)
        self.horizontalSlider = QtGui.QSlider(self.gridLayoutWidget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName(_fromUtf8("horizontalSlider"))
        self.gridLayout_3.addWidget(self.horizontalSlider, 1, 1, 1, 3)
        self.horizontalSlider_2 = QtGui.QSlider(self.gridLayoutWidget)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName(_fromUtf8("horizontalSlider_2"))
        self.gridLayout_3.addWidget(self.horizontalSlider_2, 2, 1, 1, 3)
        self.horizontalSlider_3 = QtGui.QSlider(self.gridLayoutWidget)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName(_fromUtf8("horizontalSlider_3"))
        self.gridLayout_3.addWidget(self.horizontalSlider_3, 3, 1, 1, 3)
        self.pushButton = QtGui.QPushButton(self.gridLayoutWidget)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.gridLayout_3.addWidget(self.pushButton, 1, 4, 1, 1)
        self.pushButton_3 = QtGui.QPushButton(self.gridLayoutWidget)
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.gridLayout_3.addWidget(self.pushButton_3, 2, 4, 1, 1)
        self.pushButton_2 = QtGui.QPushButton(self.gridLayoutWidget)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.gridLayout_3.addWidget(self.pushButton_2, 3, 4, 1, 1)
        self.label_2 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_3.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_3.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_3.addWidget(self.label_4, 3, 0, 1, 1)
        self.horizontalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 220, 581, 561))
        self.horizontalLayoutWidget.setObjectName(_fromUtf8("horizontalLayoutWidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(self.horizontalLayoutWidget)
        self.label.setText(_fromUtf8(""))
        self.label.setPixmap(QtGui.QPixmap(_fromUtf8("1.jpg")))
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 601, 23))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.pushButton_4.setText(_translate("MainWindow", "载入图片", None))
        self.pushButton_8.setText(_translate("MainWindow", "保存", None))
        self.pushButton_6.setText(_translate("MainWindow", "重置", None))
        self.pushButton_7.setText(_translate("MainWindow", "查看对比图", None))
        self.pushButton.setText(_translate("MainWindow", "确认", None))
        self.pushButton_3.setText(_translate("MainWindow", "确认", None))
        self.pushButton_2.setText(_translate("MainWindow", "确认", None))
        self.label_2.setText(_translate("MainWindow", "磨皮", None))
        self.label_3.setText(_translate("MainWindow", "美白", None))
        self.label_4.setText(_translate("MainWindow", "红唇", None))

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    ui.window.show()
    sys.exit(app.exec_())