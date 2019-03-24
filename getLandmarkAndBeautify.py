# -*- coding: utf-8 -*-
"""
Created on 2018/6/1
"""

import cv2
import numpy as np
import urllib2
import time
import json


class Landmark(object):
    """
    人脸关键点类
    filepath:UI打开的图片路径
    callApi：调用face++的API获得图片内人脸的关键点json字符串
    getlandmark：处理json获得的numpy数组，方便后续处理
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.call_api()
        self.Pts = self.get_landmark()

    def call_api(self):
        """
        调用face++的api
        """
        http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
        key = "c9XLgB6o3EGAu2swepjZSwQOqOiy0Z6J"
        secret = "Um26X4gbWG6ITK0q3TzzAHGBztOBA4-S"
        boundary = '----------%s' % hex(int(time.time() * 1000))
        data = []
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
        data.append(key)
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
        data.append(secret)
        data.append('--%s' % boundary)
        fr=open(self.filepath,'rb')
        data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
        data.append('Content-Type: %s\r\n' % 'application/octet-stream')
        data.append(fr.read())
        fr.close()
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
        data.append('1')
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
        data.append("gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
        data.append('--%s--\r\n' % boundary)

        http_body='\r\n'.join(data)
        #buld http request
        req=urllib2.Request(http_url)
        #header
        req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
        req.add_data(http_body)
        try:
            req.add_header('Referer','http://remotserver.com/')
            #post data to server
            resp = urllib2.urlopen(req, timeout=5)
            #get response

            qrcont = resp.read()
            print type(qrcont)
            file = ('text.json')

            with open(file, 'w') as f:
                f.write(qrcont)
                f.close()

        #print qrcont
        except urllib2.HTTPError as e:
            print e.read()

    def get_landmark(self):
        """
        暴力转换
        只处理一张人脸
        :return:关键点np数组Pts
        """
        file = open('text.json', 'r')
        jsonData = json.load(file)

        Pts = []
        for points in ['contour_right1', 'contour_right2', 'contour_right3', 'contour_right4',
                       'contour_right5', 'contour_right6', 'contour_right7', 'contour_right8', 'contour_right9',
                       'contour_left9', 'contour_chin', 'contour_left8', 'contour_left7','contour_left6',
                       'contour_left5', 'contour_left4', 'contour_left3', 'contour_left2', 'contour_left1',

                       'mouth_left_corner', 'mouth_upper_lip_left_contour2', 'mouth_upper_lip_left_contour1',
                       'mouth_upper_lip_top', 'mouth_upper_lip_right_contour1', 'mouth_upper_lip_right_contour2',
                       'mouth_right_corner', 'mouth_upper_lip_right_contour3', 'mouth_upper_lip_bottom',
                       'mouth_upper_lip_left_contour3',

                       'mouth_left_corner', 'mouth_lower_lip_left_contour1', 'mouth_lower_lip_top',
                       'mouth_lower_lip_right_contour1', 'mouth_right_corner', 'mouth_lower_lip_right_contour2',
                       'mouth_lower_lip_right_contour3', 'mouth_lower_lip_bottom', 'mouth_lower_lip_left_contour3',
                       'mouth_lower_lip_left_contour2',

                       'nose_contour_left1', 'nose_contour_left2', 'nose_left', 'nose_contour_left3',
                       'nose_contour_lower_middle', 'nose_contour_right3', 'nose_right', 'nose_contour_right2',
                       'nose_contour_right1',

                       'left_eye_left_corner', 'left_eye_upper_left_quarter', 'left_eye_top',
                       'left_eye_upper_right_quarter', 'left_eye_right_corner', 'left_eye_lower_right_quarter',
                       'left_eye_bottom', 'left_eye_lower_left_quarter',

                       'right_eye_left_corner', 'right_eye_upper_left_quarter', 'right_eye_top',
                       'right_eye_upper_right_quarter', 'right_eye_right_corner', 'right_eye_lower_right_quarter',
                       'right_eye_bottom', 'right_eye_lower_left_quarter',

                       'left_eyebrow_left_corner', 'left_eyebrow_upper_left_quarter', 'left_eyebrow_upper_middle',
                       'left_eyebrow_upper_right_quarter', 'left_eyebrow_right_corner',
                       'left_eyebrow_lower_right_quarter', 'left_eyebrow_lower_middle',
                       'left_eyebrow_lower_left_quarter',

                       'right_eyebrow_left_corner', 'right_eyebrow_upper_left_quarter', 'right_eyebrow_upper_middle',
                       'right_eyebrow_upper_right_quarter', 'right_eyebrow_right_corner',
                       'right_eyebrow_lower_right_quarter', 'right_eyebrow_lower_middle',
                       'right_eyebrow_lower_left_quarter', ]:
              Pts += [[jsonData['faces'][0]['landmark'][points]['x'], jsonData['faces'][0]['landmark'][points]['y']]]

        return np.array(Pts)


class Organ(object):
    def __init__(self, img, landmark, name, ksize=None):
        """
        五官部位类
        img:打开的bgr图像
        landmark:传入的关键点numpy数组
        name:五官的名字
        """
        self.img = img
        self.landmark = landmark
        self.name = name
        self.left = np.min(self.landmark[:, 0])
        self.right = np.max(self.landmark[:, 0])
        self.top = np.min(self.landmark[:, 1])
        self.bottom = np.max(self.landmark[:, 1])

        self.shape = (int(self.bottom - self.top), int(self.right - self.left))
        self.size = self.shape[0] * self.shape[1] * 3  # 两个维度的大小乘通道数
        self.move = int(np.sqrt(self.size / 3) / 20)  # 计算切片位置用
        self.rec = self.get_segment(self.img)
        self.ksize = self.get_ksize()

        self.rec_mask = self.get_mask_re()

    def get_ksize(self, rate=1):
        """
        模板大小
        """
        size = max([int(np.sqrt(self.size / 3) / rate), 1])

        if not (size % 2):
            size += 1

        return (size, size)

    def get_segment(self, img):
        """
        截取局部矩形切片（缩小mask计算范围）
        """
        top = np.max([0, self.top - self.move])
        bottom = np.min([img.shape[0], self.bottom + self.move])
        left = np.max([0, self.left - self.move])
        right = np.min([img.shape[1], self.right + self.move])
        return img[top:bottom, left:right]

    def get_mask_re(self, flag=True):
        """
        获得局部相对坐标遮罩
        flag：True为正向mask，False为反向mask
        """
        landmark_re = self.landmark.copy()
        landmark_re[:, 1] -= np.max([self.top - self.move, 0])
        landmark_re[:, 0] -= np.max([self.left - self.move, 0])

        if flag:
            mask = np.zeros(self.rec.shape[:2], dtype=np.float64)
        else:
            mask = np.ones(self.rec.shape[:2], dtype=np.float64)

        cv2.fillConvexPoly(mask, cv2.convexHull(landmark_re), color=(flag + 0))
        mask = np.array([mask, mask, mask])

        return mask.transpose((1, 2, 0))

    def get_mask_abs(self):
        """
        获得全局绝对坐标遮罩
        """
        mask = np.zeros(self.img.shape, dtype=np.float64)
        rec = self.get_segment(mask)
        rec[:] = self.rec_mask[:]
        return mask

    def hsv_operating(self, rate, operating=True):
        """
        美白&红唇
        arguments:
            rate:比例
        """
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        rec_hsv = self.get_segment(img_hsv)

        if operating:
            rec_hsv[:, :, -1] = self.whitening_dev(rec_hsv)
        else:
            rec_hsv[:, :, 1] = self.glitting_dev(rec_hsv)
        self.img[:] = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)[:]

    def whitening_dev(self, rec_hsv, rate=0.05):
        rec_mask = self.get_mask_re()
        return np.minimum(
            rec_hsv[:, :, -1] + rec_hsv[:, :, -1] * rec_mask[:, :, -1] * rate, 255).astype(
            'uint8')

    def glitting_dev(self, rec_hsv, rate=0.3):
        rec_mask = self.get_mask_re()
        rec_new = rec_hsv[:, :, 1] * rec_mask[:, :, 1] * rate
        rec_new = cv2.GaussianBlur(rec_new, (3, 3), 0)
        return np.minimum(rec_hsv[:, :, 1] + rec_new, 255).astype('uint8')

    def smoothing(self, rate=0.8):
        """
        磨皮
        arguments:
            rate:float,0~1,
            img=rate*new+(1-rate)*src
            confirm:wether confirm this option
        """
        index = self.rec_mask > 0
        rec_new = cv2.bilateralFilter(src=self.rec, d=20, sigmaColor=40, sigmaSpace=10)
        self.rec[index] = np.minimum(rec_new[index] * rate + self.rec[index] * (1 - rate),
                                255).astype('uint8')


class Face(Organ):
    """
    脸类
    arguments:
        img:uint8 array, inference of BGR image
        landmarks:list, landmark groups
        index:int, index of face in the image
    """
    def __init__(self, img, landmarks, index):
        self.index = index
        # 五官关键点
        self.organs_np = [('jaw', list(range(0, 18))), ('mouth_upper', list(range(19, 28))),
                            ('mouth_lower', list(range(29, 38))), ('nose', list(range(39, 47))),
                            ('left_eye', list(range(48, 55))), ('right_eye', list(range(56, 63))),
                            ('left_eyebrow', list(range(64, 71))), ('right_eyebrow', list(range(72, 79)))]

        # 实例化脸对象和五官对象
        self.organs = {name: Organ(img, landmarks[points], name) for name, points in self.organs_np}

        # 获得额头坐标，实例化额头
        mask_organs = (self.organs['mouth_upper'].get_mask_abs() + self.organs['mouth_lower'].get_mask_abs()
                       + self.organs['nose'].get_mask_abs() + self.organs['left_eye'].get_mask_abs() + self.organs['right_eye'].get_mask_abs()
                       + self.organs['left_eyebrow'].get_mask_abs() + self.organs['right_eyebrow'].get_mask_abs())
        forehead_landmark = self.get_forehead_landmark(img, landmarks, mask_organs)
        self.organs['forehead'] = Organ(img, forehead_landmark, mask_organs, 'forehead')
        mask_organs += self.organs['forehead'].get_mask_abs()

        # 人脸的完整标记点
        self.FACE_POINTS = np.concatenate([landmarks, forehead_landmark])
        super(Face, self).__init__(img, self.FACE_POINTS, 'face')

        mask_face = self.get_mask_abs() - mask_organs
        self.rec_mask = self.get_segment(mask_face)
        pass

    def get_forehead_landmark(self, img, landmark, mask_organs):
        """
        z计算额头的大致范围（实验结果偏小）
        """
        # 画椭圆
        radius = (np.linalg.norm(landmark[0]-landmark[18])/1.95).astype('int32')
        center_abs = tuple(((landmark[0]+landmark[18])/1.95).astype('int32'))
        
        angle =np.degrees(np.arctan((lambda l : l[1]/l[0])(landmark[18]-landmark[0]))).astype('int32')
        mask = np.zeros(mask_organs.shape[:2], dtype=np.float64)
        cv2.ellipse(mask, center_abs, (radius, radius), angle, 180, 360, 1, -1)
        # 剔除与五官重合部分
        mask[mask_organs[:, :, 0] > 0] = 0
        # 根据五官肤色判断真正的额头面积
        index_bool = []
        for i in range(3):
            mean, std = np.mean(img[:, :, i][mask_organs[:, :, i] > 0]), np.std(img[:, :, i][mask_organs[:, :, i] > 0])
            up, down = mean + 0.5 * std, mean - 0.5 * std
            index_bool.append((img[:, :, i] < down) | (img[:, :, i] > up))
        index_zero = ((mask > 0) & index_bool[0] & index_bool[1] & index_bool[2])
        mask[index_zero] = 0
        index_abs = np.array(np.where(mask > 0)[::-1]).transpose()
        return cv2.convexHull(index_abs).squeeze()


class Beautify(Landmark):
    """
    美颜类
    """
    def __init__(self, filepath):
        super(Beautify, self).__init__(filepath)
        self.photo_path = []
        self.faces = {}

    def read_img(self, fname, scale = 1):
        """
        对应UI，读取图片
        """
        img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
        if type(img) == type(None):
            print(fname)
            raise ValueError('Opencv error reading image "{}" , got None'.format(fname))
        return img

    def read_and_mark(self, fname):
        img = self.read_img(fname)

        faces = {fname: [Face(img, np.array(self.Pts), 0)]}

        print(faces)

        return img, faces


if __name__=='__main__':
    path = '3.jpg'
    mu = Beautify(path)
    img, faces = mu.read_and_mark(path)
    imc = img.copy()
    cv2.imshow('ori', imc)
    for face in faces[path]:
        face.hsv_operating(0.5)
        face.organs['mouth_upper'].hsv_operating(1, False)
        face.organs['mouth_lower'].hsv_operating(1, False)
        face.smoothing()
        face.organs['forehead'].smoothing()
        face.organs['mouth_upper'].smoothing()
        face.organs['mouth_lower'].smoothing()
        face.organs['left_eye'].smoothing()
        face.organs['right_eye'].smoothing()
        face.organs['nose'].smoothing()
    cv2.imshow('new', img.copy())
    cv2.waitKey()
    print('Quiting')



