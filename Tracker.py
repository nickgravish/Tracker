


import numpy as np
import cv2 as cv
from skimage.measure import label, regionprops
import json
import os
import glob
import opencv_helper.opencv_helper as cvhlp

import pyqtgraph as pg
from Tracker.Visualize import VideoStreamView
from pyqtgraph.Qt import QtCore, QtGui


class Tracker:

    def __init__(self, videoname, ROI = None, verbose = False):

        self.verbose = verbose

        self.videoname = videoname

        self.file_path = os.path.dirname(self.videoname)
        self.file_name = os.path.splitext(os.path.basename(self.videoname))[0] + '_contours.txt'
        self.out_file = os.path.join(self.file_path, self.file_name)
        self.file_exists = (glob.glob(self.out_file) != [])

        self.threshold_val = 0.7
        self.bkg_method = 'Divide'
        self.trk_method = 'opencv'
        self.bkg_sep = 100

        self.ROI = ROI

    def load_data(self):
        with open(self.out_file, 'r') as infile:
            tmp = json.load(infile)

        # load in contours
        self.contours = tmp['contours']

        # convert to the opencv format
        # tmp = []
        # for frame in self.contours:
        #     c = []
        #     for contour in frame:
        #         c.append(cvhlp.list_to_opencv_contours(contour))
        #     tmp.append(c)


        self.header = tmp['header']

        self.ROI = self.header['ROI']
        self.ROI_Width = self.ROI[3] - self.ROI[1]
        self.ROI_Height = self.ROI[2] - self.ROI[0]

        self.bkg_method = self.header['bkg_method']
        self.trk_method = self.header['trk_method']
        self.bkg_sep = self.header['bkg_sep']

    def load_video(self):
        if self.verbose:
            print("Loading")

        self.vid = cv.VideoCapture(self.videoname)

        self.NumFrames = self.vid.get(cv.CAP_PROP_FRAME_COUNT)
        self.Height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.Width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)


        # if already tracked, load in the data
        if self.file_exists:  # load in contours to associate
            self.load_data()

        else:
            if self.ROI is None:
                self.ROI = (0, 0, self.Width, self.Height)
                self.ROI_Width = self.Width
                self.ROI_Height = self.Height
            else:
                self.ROI_Width = self.ROI[3] - self.ROI[1]
                self.ROI_Height = self.ROI[2] - self.ROI[0]

                if self.verbose:
                    print(self.ROI_Height, self.ROI_Width)

        self.frames = np.zeros((self.NumFrames, self.ROI_Height, self.ROI_Width), np.uint8)
        self.background = None
        self.frames_normed = None
        self.frames_BW = None

        self.loaded = False

        self.vid.set(cv.CAP_PROP_POS_FRAMES,0)

        for kk in range(int(self.NumFrames)):
            tru, ret = self.vid.read(1)
            self.frames[kk, :, :] = ret[self.ROI[0]:self.ROI[2], self.ROI[1]:self.ROI[3], 0] # assumes loading color

            if ((kk % 100) == 0) and self.verbose:
                print(kk)

        self.loaded = True


    def compute_background(self):
        self.background = np.float64(np.median(self.frames[0:self.bkg_sep:,:,:], axis = 0))
        # add a small number to background to not have divide by zeros for division
        self.background = self.background + np.float64(1E-6)
        if self.verbose:
            print('BKG')

    def remove_background(self):
        if self.bkg_method == 'Divide':
            self.frames_normed = self.frames / self.background     # broadcasting !!
        else:
            self.frames_normed = self.frames - self.background     # broadcasting !!

        if self.verbose:
            print('NORM')

    def threshold(self):

        self.frames_BW = self.frames_normed < self.threshold_val

        if self.verbose:
            print('Threshold')

    def find_objects(self):
        if self.verbose:
            print('Tracking')

        self.contours = []

        if self.trk_method == 'sklearn':
            self.frames_objs = [regionprops(label(frame, connectivity=1)) for frame in self.frames_BW]
            for objects in self.frames_objs:
                data = []

                for object in objects:
                    if(object.area > 5): # can be made more functional
                        x = object.centroid[0]
                        y = object.centroid[1]
                        area = object.area
                        angle = object.orientation
                        ecc = object.eccentricity
                        major_axis = object.major_axis_length
                        minor_axis = object.minor_axis_length
                        bbox = object.bbox

                        data.append({'x': x, 'y': y, 'area': area,
                                     'ecc': ecc, 'angle': angle,
                                     'major_axis': major_axis,
                                     'minor_axis': minor_axis,
                                     'bbox': bbox})

                self.contours.append(data)

        elif self.trk_method == 'opencv':
            self.frames_objs = [cv.findContours(np.uint8(frame), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1] for frame in self.frames_BW]

            for objects in self.frames_objs:
                data = []

                for object in objects:
                        if len(object) > 5: # for ellipse fitting
                            ellipse = cv.fitEllipse(object)
                            (x, y), (a, b), angle = ellipse
                            a /= 2.
                            b /= 2.
                            ecc = np.min((a, b)) / np.max((a, b))
                            area = cv.contourArea(object)
                            bbox = cv.boundingRect(object)

                            data.append({'x': x, 'y': y, 'area': area,
                                         'ecc': ecc, 'angle': angle,
                                         'major_axis': a,
                                         'minor_axis': b,
                                         'bbox': bbox,
                                         'contours': cvhlp.opencv_contour_to_list(object)})

                self.contours.append(data)

    def save_JSON(self):
        with open(self.out_file, 'w+') as output:
            json.dump({'header': {'ROI': self.ROI,
                                  'bkg_method': self.bkg_method,
                                  'trk_method': self.trk_method,
                                  'bkg_sep': self.bkg_sep},
                       'contours': self.contours},
                        output,
                        sort_keys=True,
                        indent=4)

    def visualize(self):
        self.frames_contours = self.frames.copy()

        for frame, contours in zip(self.frames_contours, self.frames_objs):
            cv.drawContours(frame, contours, -1, (255,0,0))

        w = QtGui.QWidget()
        w.resize(1200, 600)
        w.move(QtGui.QApplication.desktop().screen().rect().center() - w.rect().center())

        v = VideoStreamView(self.frames_contours, transpose=True)

        layout = QtGui.QGridLayout()
        w.setLayout(layout)
        layout.addWidget(v)

        w.show()
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt4Agg')

    import matplotlib.pyplot as plt
    from time import time



    start = time()

    vidlist = ['/Users/nickgravish/Dropbox/Harvard/HighThroughputExpt/2016-08-05_12.41.20/1_08-05-16_12-41-31.770_Fri_Aug_05_12-41-20.543_115.mp4']

    # Load in images to memory during construction
    video = Tracker(vidlist[0], ROI= (30, 30, 550, 1150))

    video.compute_background()          # form background image
    video.remove_background()           # remove background
    video.threshold()                   # threshold to segment features
    video.find_objects('opencv')



