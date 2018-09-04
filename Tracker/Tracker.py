


import numpy as np
import cv2 as cv
from skimage.measure import label, regionprops
import json
import os
import glob
import sys

# sys.path.append('/Users/nickgravish/Dropbox/Python/')
import opencv_helper.opencv_helper as cvhlp

import pyqtgraph as pg
from Tracker.Visualize import VideoStreamView
from Tracker.DataAssociation import DataAssociator

from pyqtgraph.Qt import QtCore, QtGui

import copy

class Tracker:

#TODO add an open command to open the test folder

    def __init__(self, videoname, ROI = None, verbose = False, min_object_size = 100, frame_range = None):

        self.verbose = verbose

        self.videoname = videoname

        self.file_path = os.path.dirname(self.videoname)
        # contours file
        self.file_name = os.path.splitext(os.path.basename(self.videoname))[0] + '_contours.txt'
        self.out_file = os.path.join(self.file_path, self.file_name)
        self.file_exists = (glob.glob(self.out_file) != [])

        # association file
        self.file_name_assoc = os.path.splitext(os.path.basename(self.videoname))[0] + '_tracks.txt'
        self.out_file_assoc = os.path.join(self.file_path, self.file_name_assoc)
        self.file_exists_assoc = (glob.glob(self.out_file_assoc) != [])

        #TODO Need to make this have optional constructor inputs
        self.threshold_val = 0.8
        self.bkg_method = 'Divide'
        self.trk_method = 'opencv'
        self.bkg_sep = 100
        self.min_object_size = min_object_size
        self.frame_range = frame_range

        self.ROI = ROI

        self.video_loaded = False

        # initialize things that could be loaded from data files
        #TODO need check
        self.objects = {}
        self.frames_tracks = []
        self.contours = []
        self.raw_contours = []

    def load_data(self):
        """
        Load in any tracking data associated with this tracker object. So far this can be either a
            contours file
            association file

        """

        # first contours files
        if self.file_exists is True:
            with open(self.out_file, 'r') as infile:
                data_file = json.load(infile)

            # load in contours
            self.contours = data_file['contours']

            # convert to the opencv format
            tmp = []
            for frame in self.contours:
                c = []
                if frame != [-1]:   # check for untracked placeholder
                    for contour in frame:
                        c.append(cvhlp.list_to_opencv_contours(contour['contours']))

                tmp.append(c)

            self.raw_contours = tmp

            self.header = data_file['header']

            self.ROI = self.header['ROI']
            self.ROI_Width = self.ROI[3] - self.ROI[1]
            self.ROI_Height = self.ROI[2] - self.ROI[0]

            self.bkg_method = self.header['bkg_method']
            self.trk_method = self.header['trk_method']
            self.bkg_sep = self.header['bkg_sep']

        # next association file
        if self.file_exists_assoc is True:
            with open(self.out_file_assoc, 'r') as infile:
                data_file = json.load(infile)

            # load in contours
            objs = data_file['objects']
            trks = data_file['tracks']

            # conver frame by frame tracks to matrices like in original tracking format
            for frm in trks:
                for obj in frm:
                    obj['measurements'] = np.matrix(obj['measurements'])
                    obj['xy_cov_matrix'] = np.matrix(obj['xy_cov_matrix'])

            # convert dict keys to ints rather than strings. This happens because of JSON export
            objs = {int(k): item for k, item in objs.items()}

            # convert objects into matrices
            for kk, obj in objs.items():
                obj['x'] = np.array(obj['x'])
                obj['y'] = np.array(obj['y'])
                obj['vy'] = np.array(obj['vy'])
                obj['vx'] = np.array(obj['vx'])
                obj['area'] = np.array(obj['area'])
                obj['error'] = np.array(obj['error'])
                obj['angle'] = np.array(obj['angle'])
                obj['frames'] = np.array(obj['frames'])
                obj['measurements'] = np.matrix(obj['measurements'])
                obj['xy_cov_matrix'] = [np.matrix(o) for o in obj['xy_cov_matrix']]

            self.frames_tracks = trks
            self.objects = objs

    def associate_contours(self,
                           max_covariance=30,
                           max_velocity=10,
                           n_covariances_to_reject=3,
                           max_tracked_objects=100,
                           kalman_state_cov=0.2,
                           kalman_init_cov=10,
                           kalman_measurement_cov=50):

        tmp = DataAssociator(self.videoname,
                             max_covariance = max_covariance,
                             max_velocity = max_velocity,
                             n_covariances_to_reject = n_covariances_to_reject,
                             max_tracked_objects = max_tracked_objects,
                             kalman_measurement_cov = kalman_measurement_cov,
                             kalman_init_cov = kalman_init_cov,
                             kalman_state_cov = kalman_state_cov)
        tmp.run()

        self.frames_tracks = tmp.frame_objects
        self.objects = tmp.objects

        del tmp

    def get_number_frames(self):
        self.vid = cv.VideoCapture(self.videoname)
        self.NumFrames = int(self.vid.get(cv.CAP_PROP_FRAME_COUNT))
        return self.NumFrames

    def load_video(self):
        if self.verbose:
            print("Loading")

        self.vid = cv.VideoCapture(self.videoname)

        self.NumFrames = int(self.vid.get(cv.CAP_PROP_FRAME_COUNT))
        if not (self.NumFrames > 0):
            self.exit_error()
            raise IOError('Codec issue: cannot read number of frames.')

        self.Height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.Width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)

        if self.frame_range is None:
            self.frame_range = (0, self.NumFrames)
        else:
            # check doesn't exceed number of frames
            if self.frame_range[0] + self.frame_range[1] > self.NumFrames:
                self.frame_range = (self.frame_range[0], self.NumFrames - self.frame_range[0])



        # if already tracked, load in the data
        if self.file_exists:  # load in contours to associate
            self.load_data()

        else:
            if self.ROI is None:
                self.ROI = (0, 0, self.Height, self.Width)
                self.ROI_Width = self.Width
                self.ROI_Height = self.Height
            else:
                self.ROI_Width = self.ROI[3] - self.ROI[1]
                self.ROI_Height = self.ROI[2] - self.ROI[0]

                if self.verbose:
                    print(self.ROI_Height, self.ROI_Width)

        self.frames = np.zeros((self.frame_range[1], self.ROI_Height, self.ROI_Width), np.uint8)
        self.background = None
        self.frames_normed = None
        self.frames_BW = None

        self.video_loaded = False

        # set first frame to load
        self.set_frame(self.frame_range[0])

        for kk in range(self.frame_range[1]):
            tru, ret = self.vid.read(1)

            # check if video frames are being loaded
            if not tru:
                self.exit_error()
                raise IOError('Codec issue: cannot load frames.')

            self.frames[kk, :, :] = ret[self.ROI[0]:self.ROI[2], self.ROI[1]:self.ROI[3], 0] # assumes loading color

            if ((kk % 100) == 0) and self.verbose:
                print(kk)

        self.video_loaded  = True

        # init contours and other things if has not been loaded from a data file
        if len(self.contours) == 0:
            self.contours = [[-1] for k in range(self.NumFrames)]


    def exit_error(self):
        []
        # self.contours = -1
        # self.save_JSON()

    def set_frame(self, frame):
        self.vid.set(cv.CAP_PROP_POS_FRAMES, 0)
        for kk in range(frame):
            tru, ret = self.vid.read(1)
        # self.vid.set(cv.CAP_PROP_POS_FRAMES, frame)


    def compute_background(self):
        """
        Independent of the frame range loaded, background has to be computed over total video or else can run into
        tracking problems
        """

        # if all frames loaded, do as normal
        if self.frame_range[1] == self.NumFrames:
            self.background = np.float32(np.median(self.frames[0::self.bkg_sep,:,:], axis = 0))
            print('all_loaded!')
        else:
            self.background = []
            for kk in range(0, self.NumFrames, self.bkg_sep):
                self.vid.set(cv.CAP_PROP_POS_FRAMES, kk)
                tru, ret = self.vid.read(1)

                # check if video frames are being loaded
                if not tru:
                    self.exit_error()
                    raise IOError('Codec issue: cannot load frames.')

                self.background.append(ret[self.ROI[0]:self.ROI[2], self.ROI[1]:self.ROI[3],0])  # assumes loading color

            self.background = np.array(self.background, dtype='float32')
            self.background = np.float32(np.median(self.background, axis=0))

        # add a small number to background to not have divide by zeros for division
        self.background = self.background + np.float32(1E-6)
        if self.verbose:
            print('BKG')

    def remove_background(self):
        if self.bkg_method == 'Divide':
            norm_bkg = np.mean(self.background[:]) # normalize for mean intensity of image
            # norm_frm = np.mean(self.frames, axis=(1,2)) # normalize for mean intensity of current frame. For flicker
            self.frames_normed = (self.frames / norm_bkg) / (self.background / norm_bkg)    # broadcasting !!

        else:
            self.frames_normed = self.frames - self.background     # broadcasting !!

        if self.verbose:
            print('NORM')

    def threshold(self):
        self.frames_BW = self.frames_normed < self.threshold_val

        if self.verbose:
            print('Threshold')

    def morpho_closing(self, kernel_size = 5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.frames_BW = [cv.morphologyEx(np.uint8(frm), cv.MORPH_CLOSE, kernel) for frm in self.frames_BW]

        if self.verbose:
            print('Threshold')

    def find_objects(self):
        if self.verbose:
            print('Tracking')

        # self.contours = []

        # if self.trk_method == 'sklearn':
        #     self.raw_contours = [regionprops(label(frame, connectivity=1)) for frame in self.frames_BW]
        #     for objects in self.frames_objs:
        #         data = []
        #
        #         for object in objects:
        #             if(object.area > self.min_object_size):
        #                 x = object.centroid[0]
        #                 y = object.centroid[1]
        #                 area = object.area
        #                 angle = object.orientation
        #                 ecc = object.eccentricity
        #                 major_axis = object.major_axis_length
        #                 minor_axis = object.minor_axis_length
        #                 bbox = object.bbox
        #
        #                 data.append({'x': x, 'y': y, 'area': area,
        #                              'ecc': ecc, 'angle': angle,
        #                              'major_axis': major_axis,
        #                              'minor_axis': minor_axis,
        #                              'bbox': bbox})
        #
        #         self.contours.append(data)

        elif self.trk_method == 'opencv':
            self.raw_contours = [cv.findContours(np.uint8(frame), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1] for frame in self.frames_BW]

            for kk, objects in enumerate(self.raw_contours):
                data = []

                for object in objects:
                        if len(object) > self.min_object_size:
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

                self.contours[kk + self.frame_range[0]] = data

    def clean_tracks(self, x_range = (200,900)):
        """
        Criteria to keep or not keep tracks
        For right now I want to keep tracks that go from x = 200-900 contiguously.
        """

        # first clean objects list
        objs_that_dont_satisfy_criteria = [k for k, obj in self.objects.items() if (np.all(obj['x'] > x_range[0]) or
                                                                                np.all(obj['x'] < x_range[1]))]
        for k in objs_that_dont_satisfy_criteria:
            del self.objects[k]

        # then remove frame references to that unused objects
        self.frames_tracks = [[obj for obj in frame if obj['ID'] not in objs_that_dont_satisfy_criteria]
                         for frame in self.frames_tracks]


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

    def save_association_JSON(self):

        # TODO figure out how to best handle conversion to list


        # need to save two lists, frames_tracks, and objects
        tmp_tracks = copy.deepcopy(self.frames_tracks)
        for frm in tmp_tracks:
            for obj in frm:
                obj['measurements'] = obj['measurements'].tolist()
                obj['xy_cov_matrix'] = obj['xy_cov_matrix'].tolist()

        tmp_objs = copy.deepcopy(self.objects)
        for kk, obj in tmp_objs.items():
            obj['x'] = obj['x'].tolist()
            obj['y'] = obj['y'].tolist()
            obj['vy'] = obj['vy'].tolist()
            obj['vx'] = obj['vx'].tolist()
            obj['area'] = obj['area'].tolist()
            obj['error'] = obj['error'].tolist()
            obj['angle'] = obj['angle'].tolist()
            obj['frames'] = obj['frames'].tolist()
            obj['measurements'] = obj['measurements'].tolist()
            obj['xy_cov_matrix'] = [o.tolist() for o in obj['xy_cov_matrix']]

        # a hack right now
        with open(self.out_file_assoc, 'w+') as output:
            json.dump({'tracks': tmp_tracks,
                       'objects': tmp_objs},
                      output,
                      sort_keys=True,
                      indent=4)


    def draw_contours(self):
        self.frames_contours = self.frames.copy()

        for frame, contours in zip(self.frames_contours, self.raw_contours):
            cv.drawContours(frame, contours, -1, (255, 0, 0))

    def visualize(self):

        # Load video if not already
        if self.video_loaded is not True:
            self.load_video()

        self.draw_contours()

        if self.verbose:
            print('visualizing')


        self.w = QtGui.QWidget()
        self.w.resize(1200, 600)
        # w.move(QtGui.QApplication.desktop().screen().rect().center() - w.rect().center())

        self.v = VideoStreamView(self.frames_contours, transpose=True)

        self.layout = QtGui.QGridLayout()
        self.w.setLayout(self.layout)
        self.layout.addWidget(self.v)

        self.w.show()

        #TODO Need to handle memory clean up of visualization much better

    def on_close(self):
        del self.v
        del self.w
        del self.layout

        print('closed')

    def __str__(self):
        """
        #TODO

        """


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt4Agg')

    import matplotlib.pyplot as plt

    file = '/Users/nickgravish/Dropbox/Harvard/HighThroughputExpt/' \
           'Bee_experiments_2016/2016-08-15_13.05.57/' \
           '1_08-15-16_13-06-05.015_Mon_Aug_15_13-05-57.148_2.mp4'

    # Load in images to memory during construction
    video = Tracker(file, ROI=(30, 30, 550, 1174), verbose='True')


    video.load_video()
    video.compute_background()  # form background image
    video.remove_background()  # remove background
    video.threshold()  # threshold to segment features
    video.find_objects()

    video.visualize()


