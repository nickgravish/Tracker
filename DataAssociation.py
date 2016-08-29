
import Tracker
import numpy as np

import sys
import os
import glob

sys.path.append('/Users/nickgravish/Dropbox/Python/')
import Kalman as Kalman
import json

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from Visualize import VideoStreamView

class KalmanParameters():
    def __init__(self):
        self.P0 = None
        self.phi = None
        self.gamma = None
        self.H = None
        self.Q = None
        self.R = None
        self.gammaW = None
        self.association_matrix = None

class Header():
    """
    For ROS compliance if want to use for real time estimation
    """
    def __init__(self, stamp, frame):
        self.stamp = stamp
        self.frame_id = frame

class ContourList():
    """
    For ROS compliance if want to use for real time estimation
    """
    def __init__(self, stamp, frame, contours):
        self.contours = contours
        self.header = Header(stamp, frame)

class Contours():
    """
    for ROS compliance ...
    """
    def __init__(self, contour):
        self.x = contour['x']
        self.y =  contour['y']
        self.area = contour['area']
        self.angle = contour['angle']
        self.z = 0




class DataAssociator():
    """
    This class is originally from and has been modified
    https://github.com/florisvb/multi_tracker/blob/master/multi_tracker_analysis/Kalman.py

    Steps to data association:
    1. From list of contours, loop through all maintained objects and find min dist between each contour and all objs
    2. If distance between contour and object is below the predicted error from kalman filter, assign contour to obj
    3. If distance greater than any objs then either instantiate new obj if this is first frame
    4. Clean up



    """
    def __init__(self, filename):

        # init empty kalman parameters
        self.kalman_parameters = KalmanParameters()
        self.kalman_parameters.P0 = None
        self.kalman_parameters.phi = None
        self.kalman_parameters.gamma = None
        self.kalman_parameters.H = None
        self.kalman_parameters.Q = None
        self.kalman_parameters.R = None
        self.kalman_parameters.gammaW = None
        self.kalman_parameters.association_matrix = None

        self.init_kalman_parameters()

        self.videoname = filename

        self.file_path = os.path.dirname(self.videoname)
        self.file_name = os.path.splitext(os.path.basename(self.videoname))[0] + '_contours.txt'
        self.out_file = os.path.join(self.file_path, self.file_name)
        self.file_exists = (glob.glob(self.out_file) != [])

        with open(self.out_file, 'r') as infile:
            tmp = json.load(infile)
            self.frame_contours = tmp['contours']
            self.header = tmp['header']

        self.frame_objects = []

        self.sampling_frequency = 100

        # the dictionary to hold the tracked objects data in
        self.tracked_objects = {} # currently heald objects
        self.objects = {} # persistent list of all tracked objects
        self.current_objid = 0

        self.max_tracked_objects = 20
        self.max_covariance = 30
        self.max_velocity = 10
        self.n_covariances_to_reject_data = 3


    def init_kalman_parameters(self, filename=None):
        """
        To be modified later. Right now this will hold default update parameters for the kalman filter
        """

        if filename is None:

            # state matrix
            self.kalman_parameters.phi = \
                np.matrix([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # vx
                           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # y
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # vy
                           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # z
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # vz
                           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # area
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # delta area
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # angle
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])  # delta angle

            # observation matrix
            self.kalman_parameters.H = \
                np.matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # y
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # z
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # area
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])  # angle

            # covariance matrix
            self.kalman_parameters.P0 = 10 * np.eye(10)

            # process noise covariance
            self.kalman_parameters.Q = .2 * np.matrix(np.eye(10))

            # measurement noise
            self.kalman_parameters.R = 50 * np.matrix(np.eye(5))

            self.kalman_parameters.gamma = None
            self.kalman_parameters.gammaW = None

            self.kalman_parameters.max_covariance = 30
            self.kalman_parameters.max_velocity = 10

            self.kalman_parameters.association_matrix = np.matrix([[1, 1, 0, 0, 0]], dtype=float).T
            self.kalman_parameters.association_matrix /= np.linalg.norm(self.kalman_parameters.association_matrix)

        else:
            # TODO Account for custom kalman files
            return

    def measurement_covariance(self, obj):
        return (obj['kalmanfilter'].H * obj['kalmanfilter'].P).T * obj['association_matrix']

    def norm_of_covariance(self, obj):
        """
        Helper to compute errors
        """
        return np.linalg.norm(self.measurement_covariance(obj))

    def return_frame_based_tracked(self, tracked_obj):

        # obj = {'x': None, 'y': None, 'vx': None, 'vy': None,
        #        'angle': None, 'area': None, 'error': None,
        #         'measurements': None, 'frames': None, 'xy_cov_matrix': None}

        tracks = []
        for obj_id, track in tracked_obj.items():
            obj = {}

            obj['x'] = np.float64(track['state'][0])
            obj['y'] = np.float64(track['state'][2])
            obj['vx'] = np.float64(track['state'][1])
            obj['vy'] = np.float64(track['state'][3])
            obj['angle'] = np.float64(track['measurement'][4])
            obj['area'] = np.float64(track['state'][6])
            obj['error'] = np.float64(self.norm_of_covariance(track))
            obj['measurements'] = track['measurement']
            obj['frames'] = track['frames'][-1]
            obj['xy_cov_matrix'] = track['kalmanfilter'].P[0:2, 0:2]
            obj['ID'] = obj_id

            tracks.append(obj)

        return tracks


    def update_tracked_memory(self, tracked_obj):
        """
        Helper function to abstract the way we pull state information from the tracked objects to be stored
        """
        if tracked_obj['objid'] in self.objects:
            self.objects[tracked_obj['objid']]['x'] = np.hstack((self.objects[tracked_obj['objid']]['x'],
                                                                 np.float64(tracked_obj['state'][0])))
            self.objects[tracked_obj['objid']]['y'] = np.hstack((self.objects[tracked_obj['objid']]['y'],
                                                                 np.float64(tracked_obj['state'][2])))
            self.objects[tracked_obj['objid']]['vx']= np.hstack((self.objects[tracked_obj['objid']]['vx'],
                                                                 np.float64(tracked_obj['state'][1])))
            self.objects[tracked_obj['objid']]['vy'] = np.hstack((self.objects[tracked_obj['objid']]['vy'],
                                                                     np.float64(tracked_obj['state'][3])))
            self.objects[tracked_obj['objid']]['angle'] = np.hstack((self.objects[tracked_obj['objid']]['angle'],
                                                                  np.float64(tracked_obj['measurement'][4])))
            self.objects[tracked_obj['objid']]['area'] = np.hstack((self.objects[tracked_obj['objid']]['area'],
                                                                  np.float64(tracked_obj['state'][6])))

            self.objects[tracked_obj['objid']]['error'] = np.hstack((self.objects[tracked_obj['objid']]['error'],
                                                                     np.float64(self.norm_of_covariance(tracked_obj))))
            self.objects[tracked_obj['objid']]['measurements'] = np.hstack((self.objects[tracked_obj['objid']]['measurements'],
                                                                            tracked_obj['measurement']))
            self.objects[tracked_obj['objid']]['frames'] = np.hstack((self.objects[tracked_obj['objid']]['frames'],
                                                                      tracked_obj['frames'][-1]))
            self.objects[tracked_obj['objid']]['xy_cov_matrix'].append(tracked_obj['kalmanfilter'].P[0:2,0:2])


        else:
            self.objects.setdefault(tracked_obj['objid'], {'x': np.float64(tracked_obj['state'][0]),
                                                           'y': np.float64(tracked_obj['state'][2]),
                                                           'vx': np.float64(tracked_obj['state'][1]),
                                                           'vy': np.float64(tracked_obj['state'][3]),
                                                           'angle': np.float64(tracked_obj['measurement'][4]),
                                                           'area': np.float64(tracked_obj['measurement'][3]),
                                                           'error': np.float64(self.norm_of_covariance(tracked_obj)),
                                                           'measurements': tracked_obj['measurement'],
                                                           'frames': np.float64(tracked_obj['frames'][-1]),
                                                           'xy_cov_matrix': []})

    def contour_identifier(self, contourlist):


        def update_tracked_object(tracked_object, measurement, frame):
            """
            Takes in a tracked_obj and updates the observed measurement
            """
            if measurement is None:  # updates object if no association was found this frame
                measurement = np.matrix([np.nan for i in range(tracked_object['measurement'].shape[0])]).T
                xhat, P, K = tracked_object['kalmanfilter'].update(None)  # run kalman filter
            else:
                # tracked_object['measurement'] = np.hstack(
                #     (tracked_object['measurement'], measurement))  # add object's data to the tracked object
                xhat, P, K = tracked_object['kalmanfilter'].update(measurement)  # run kalman filter

            tracked_object['measurement'] = measurement
            tracked_object['frames'].append(frame)
            tracked_object['nframes'] += 1
            tracked_object['state'] = xhat

            self.update_tracked_memory(tracked_object)


        # keep track of which new objects have been "taken"
        contours_accounted_for = []

        # iterate through objects first
        # get order of persistence
        objid_in_order_of_persistance = []
        if len(self.tracked_objects.keys()) > 0:
            persistance = []
            objids = []
            for objid, tracked_object in self.tracked_objects.items():
                persistance.append(tracked_object['nframes'])
                objids.append(objid)
            order = np.argsort(persistance)[::-1]
            objid_in_order_of_persistance = [objids[o] for o in order]

        # loop through contours and find errors to all tracked objects (if less than allowed error)
        # then loop through the errors in order of increasing error and assigned contours to objects
        contour_to_object_error = []
        tracked_object_state_estimates = None
        tracked_object_covariances = None
        tracked_object_ids = []
        for objid, tracked_object in self.tracked_objects.items():
            tose = np.array([[tracked_object['kalmanfilter'].xhat_apriori[0, 0],
                              tracked_object['kalmanfilter'].xhat_apriori[2, 0]]])
            cov = np.array([self.norm_of_covariance(tracked_object)])
            tracked_object_ids.append(objid)
            if tracked_object_state_estimates is None:
                tracked_object_state_estimates = tose
                tracked_object_covariances = cov
            else:
                tracked_object_state_estimates = np.vstack((tracked_object_state_estimates, tose))
                tracked_object_covariances = np.vstack((tracked_object_covariances, cov))

        if tracked_object_state_estimates is not None:
            for c, contour in enumerate(contourlist.contours):
                m = np.array([[contour.x, contour.y]]) # state measurement of contour
                error = np.array([np.linalg.norm(m - e) for e in tracked_object_state_estimates]) # mutual error between current contour and all objects
                ncov = self.n_covariances_to_reject_data * np.sqrt(tracked_object_covariances)

                # this selects only the (contour, object) pair where the state error between prediction and measure is less than kalman errror
                indices = np.where((error < ncov))[0]
                if len(indices) > 0:
                    new_contour_object_errors = [[error[i], tracked_object_ids[i], c] for i in indices]
                    contour_to_object_error.extend(new_contour_object_errors)

        # Association and Propagation
        # o = []
        if len(contour_to_object_error) > 0:
            contour_to_object_error = np.array(contour_to_object_error)
            sorted_indices = np.argsort(contour_to_object_error[:, 0])
            contour_to_object_error = contour_to_object_error[sorted_indices, :]
            contours_accounted_for = []
            objects_accounted_for = []
            for data in contour_to_object_error:
                c = int(data[2])
                objid = int(data[1])
                if objid not in objects_accounted_for:
                    if c not in contours_accounted_for:
                        contour = contourlist.contours[c]
                        measurement = np.matrix([contour.x, contour.y, 0, contour.area, contour.angle]).T
                        tracked_object = self.tracked_objects[objid]
                        update_tracked_object(tracked_object, measurement, contourlist.header.frame_id)
                        contours_accounted_for.append(c)
                        objects_accounted_for.append(objid)
                        # e = [   tracked_object['kalmanfilter'].xhat_apriori[0] - contour.x,
                        #        tracked_object['kalmanfilter'].xhat_apriori[2] - contour.y]
                        # tracked_object_covariance = np.linalg.norm( (tracked_object['kalmanfilter'].H*tracked_object['kalmanfilter'].P).T*self.association_matrix )
                        # o.append([objid, e, tracked_object_covariance])

        # any unnaccounted contours should spawn new objects
        for c, contour in enumerate(contourlist.contours):
            if c not in contours_accounted_for:
                obj_state = np.matrix([contour.x, 0, contour.y, 0, 0, 0, contour.area, 0, contour.angle,
                                       0]).T  # pretending 3-d tracking (z and zvel) for now
                obj_measurement = np.matrix([contour.x, contour.y, 0, contour.area, contour.angle]).T
                # If not associated with previous object, spawn a new object
                new_obj = {'objid': self.current_objid,
                           'statenames': {'position': [0, 2, 4],
                                          'velocity': [1, 3, 5],
                                          'size': 6,
                                          'd_size': 7,
                                          'angle': 8,
                                          'd_angle': 9,
                                          },
                           'state': obj_state,
                           'measurement': obj_measurement,
                           'frames': [int(contourlist.header.frame_id)],
                           'kalmanfilter': Kalman.DiscreteKalmanFilter(x0=obj_state,
                                                                       P0=self.kalman_parameters.P0,
                                                                       phi=self.kalman_parameters.phi,
                                                                       gamma=self.kalman_parameters.gamma,
                                                                       H=self.kalman_parameters.H,
                                                                       Q=self.kalman_parameters.Q,
                                                                       R=self.kalman_parameters.R,
                                                                       gammaW=self.kalman_parameters.gammaW,
                                                                       ),
                           'association_matrix': self.kalman_parameters.association_matrix,
                           'nframes': 0,
                           }
                # add a new tracked object
                self.tracked_objects.setdefault(new_obj['objid'], new_obj)
                self.update_tracked_memory(new_obj)

                self.current_objid += 1

        # propagate unmatched objects
        for objid, tracked_object in self.tracked_objects.items():
            if tracked_object['frames'][-1] != int(contourlist.header.frame_id):
                update_tracked_object(tracked_object, None, contourlist.header.frame_id)

        # make sure we don't get too many objects - delete the oldest ones, and ones with high covariances
        objects_to_destroy = []
        if len(objid_in_order_of_persistance) > self.max_tracked_objects:
            for objid in objid_in_order_of_persistance[self.max_tracked_objects:]:
                objects_to_destroy.append(objid)

        # check covariance, and velocity
        for objid, tracked_object in self.tracked_objects.items():
            tracked_object_covariance = self.norm_of_covariance(tracked_object)

            if tracked_object_covariance > self.max_covariance:
                if objid not in objects_to_destroy:
                    objects_to_destroy.append(objid)
            v = np.linalg.norm(
                np.array(tracked_object['state'][tracked_object['statenames']['velocity'], -1]).flatten().tolist())
            if v > self.max_velocity:
                if objid not in objects_to_destroy:
                    objects_to_destroy.append(objid)
        for objid in objects_to_destroy:
            del (self.tracked_objects[objid])
            # print('destroying ', objid)

        # recalculate persistance (not necessary, but convenient)
        objid_in_order_of_persistance = []
        if len(self.tracked_objects.keys()) > 0:
            persistance = []
            for objid, tracked_object in self.tracked_objects.items():
                persistance.append(len(tracked_object['frames']))
                objid_in_order_of_persistance.append(objid)
            order = np.argsort(persistance)[::-1]
            objid_in_order_of_persistance = [objid_in_order_of_persistance[o] for o in order]
        if len(objid_in_order_of_persistance) > 0:
            most_persistant_objid = objid_in_order_of_persistance[0]


    def run(self):
        """
        This function serves contours to the tracker. Can be augmented to work on realtime video streams later.
        """
        for frame, contours_in_frame in enumerate(self.frame_contours):
            contour_as_class = [Contours(c) for c in contours_in_frame]  # convert to class format
            contour_list = ContourList(frame / self.sampling_frequency, frame, contour_as_class)
            self.contour_identifier(contour_list)
            self.frame_objects.append(self.return_frame_based_tracked(self.tracked_objects))
            # print(frame)

    def visualize(self):

        self.tracker = Tracker.Tracker(self.videoname, ROI=(30, 30, 550, 1174), verbose='True')
        self.tracker.load_video()
        self.tracker.draw_contours()

        app = QtGui.QApplication([])

        w = pg.GraphicsView()
        w.show()
        w.resize(1200, 600)
        w.move(QtGui.QApplication.desktop().screen().rect().center() - w.rect().center())

        view = pg.ViewBox()
        w.setCentralItem(view)

        ## lock the aspect ratio
        view.setAspectLocked(True)

        ## Create image item
        img = VideoStreamView(self.tracker.frames_contours, transpose=True)
        view.addItem(img)

        plt = pg.plot([1, 5, 2, 4, 3, 2], pen='r')

        # add an image, scaled
        # img.scale(0.2, 0.1)
        # img.setZValue(-100)
        plt.addItem(img)




        # w = QtGui.QWidget()
        # w.resize(1200, 600)
        # w.move(QtGui.QApplication.desktop().screen().rect().center() - w.rect().center())
        #
        # v = VideoStreamView(self.frames_contours, transpose=True)
        #
        # layout = QtGui.QGridLayout()
        # w.setLayout(layout)
        # layout.addWidget(v)
        #
        # w.show()
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt4Agg')

    import matplotlib.pyplot as plt

    file = '/Users/nickgravish/Dropbox/Harvard/HighThroughputExpt/' \
           'Bee_experiments_2016/2016-08-15_13.05.57/' \
           '1_08-15-16_13-06-05.015_Mon_Aug_15_13-05-57.148_2.mp4'

    data = DataAssociator(filename=file)
    data.run()
    data.visualize()

