
from .Tracker import Tracker
from Tracker.Visualize import *
import pyqtgraph as pg
import cv2 as cv

from pyqtgraph.Qt import QtCore, QtGui

class MultiViewSystem:

    def __init__(self, files):

        self.video1 = Tracker(files[0], verbose='True', frame_range= (1000, 1000), min_object_size=20)
        self.video2 = Tracker(files[1], verbose='True', frame_range= (1000, 1000), min_object_size=20)

        self.video1.load_video()
        self.video2.load_video()

        self.calibration_criteria = cv.CALIB_FIX_PRINCIPAL_POINT

        self.files = [os.path.basename(file) for file in files]
        self.data = dict()
        self.data[self.files[0]] = {}
        self.data[self.files[1]] = {}

        self.camera_intrinsics = dict()
        self.camera_intrinsics[self.files[0]] = {}
        self.camera_intrinsics[self.files[1]] = {}

        self.camera_extrinsics = dict()
        self.camera_extrinsics[self.files[0]] = {}
        self.camera_extrinsics[self.files[1]] = {}

        self.camera_info = dict()
        self.camera_info[self.files[0]] = {'Resolution': (self.video1.Width, self.video1.Height)}
        self.camera_info[self.files[1]] = {'Resolution': (self.video2.Width, self.video2.Height)}

        self.pattern_size = (4,4)
        self.pattern_length = 11.5

    def get_hand_trackdata(self):
        # self.data[self.files[0]]
        # self.data[self.files[1]]

        data1 = self.vv1.video_stream.hand_tracked_points.return_json()
        data2 = self.vv2.video_stream.hand_tracked_points.return_json()

        data1_pts = dict()
        for ptname, pt in data1.items():
            if ptname.isdigit():
                pname = int(ptname)

            for frame, x, y in zip(pt['frames'], pt['x'], pt['y']):

                if frame in data1_pts:                    # else:
                    data1_pts[frame][pname] = [x, y]
                else:
                    data1_pts[frame] = {pname: [x, y]}

        data2_pts = dict()
        for ptname, pt in data2.items():
            if ptname.isdigit():
                pname = int(ptname)

            for frame, x, y in zip(pt['frames'], pt['x'], pt['y']):

                if frame in data2_pts:                    # else:
                    data2_pts[frame][pname] = [x, y]
                else:
                    data2_pts[frame] = {pname: [x, y]}


        self.data[self.files[0]] = data1_pts
        self.data[self.files[1]] = data2_pts


    def visualize(self):
        app = pg.mkQApp()

        self.vv1 = MultiDataView(self.video1.frames,
                                 self.video1.frames_contours,
                                 contours_data=[],
                                 fname=self.video1.videoname)
        self.vv2 = MultiDataView(self.video2.frames,
                                 self.video2.frames_contours,
                                 contours_data=[],
                                 fname=self.video2.videoname)

        # self.vv1_viz = self.video1.visualize()
        # self.vv2_viz = self.video2.visualize()
        #

        # self.vv1.video_stream.imageItem.mouseClickEvent = self.vv1.video_stream.ms_click2
        # self.vv2.video_stream.imageItem.mouseClickEvent = self.vv2.video_stream.ms_click2

        self.vv1.video_stream.sigIndexChanged.connect(self.vv2.video_stream.setCurrentIndex)
        self.vv2.video_stream.sigIndexChanged.connect(self.vv1.video_stream.setCurrentIndex)

        self.vv1.video_stream.sigHandtrackPointNameChanged.connect(self.vv2.video_stream.update_handtrack)
        self.vv2.video_stream.sigHandtrackPointNameChanged.connect(self.vv1.video_stream.update_handtrack)

        self.vv1.video_stream.sigHandtrackPointValChanged.connect(self.vv2.video_stream.update_handtrack_point)
        self.vv2.video_stream.sigHandtrackPointValChanged.connect(self.vv1.video_stream.update_handtrack_point)

        self.vv1.video_stream.sigHandTrackLoaded.connect(self.vv2.video_stream.other_file_loaded)
        self.vv2.video_stream.sigHandTrackLoaded.connect(self.vv1.video_stream.other_file_loaded)

        self.make_multi_control_menu()

    def make_multi_control_menu(self):
        app = pg.mkQApp()
        self.widget = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()

        self.Calibrate = QtGui.QPushButton('Calibrate system')
        self.Calibrate.clicked.connect(self.calibrate_system)
        self.layout.addWidget(self.Calibrate, 1,1)

        self.tree = pg.DataTreeWidget(data=self.data)
        self.layout.addWidget(self.tree, 2, 1)

        self.widget.setLayout(self.layout)
        self.widget.resize(400, 600)
        self.widget.show()


    def calibrate_system(self):

        print('Calibrating!')

        self.get_hand_trackdata()
        self.calibrate_intrinsics()
        self.calibrate_extrinsics()
        self.tree.setData([self.camera_extrinsics, self.camera_intrinsics,self.data])

    # write out calibration to JSon file
    def write_intrinsics_to_json(self, P, D, height, width, name):
        tmpjson = {'P': P.tolist(), 'D': D.tolist(), 'width': width, 'height': height}

        with open(name, 'w') as out:
            json.dump(tmpjson, out, indent=4)

    def compute_rmse_fromcv(self, oP, iP, P, D, translation, R):
        tot_error = 0
        for i in range(len(oP)):
            imgpoints2, _ = cv.projectPoints(oP[i], R[i], translation[i], P, D)

            # watch out here because pytohn will silently broadcast this subtraction and give wrong error results.
            # Because opencv returns annoying list of list

            error = np.linalg.norm(iP[i] - imgpoints2[:, 0, :]) / len(imgpoints2)
            tot_error += error

        print("total error: ", tot_error / len(oP))
        return tot_error / len(oP)

    def compute_rmse_stereo(self, iP1, iP2, cam_system1, cam_system2):
        '''
            Compute the reprojection error from a stereo projection. 
            Returned units are in pixels
        '''

        pterr = []

        tot_error1 = 0
        tot_error2 = 0
        for i in range(len(iP1)):
            xyz = cv.triangulatePoints(np.array(cam_system1['projection_matrix']),
                                        np.array(cam_system2['projection_matrix']), iP1[i], iP2[i])

            iP1_repro, _ = cv.projectPoints(np.array([np.squeeze(xyz[:-1:1] / xyz[3])]), np.array(cam_system1['Q']),
                                             np.array(cam_system1['translation']), np.array(cam_system1['K']),
                                             np.array(cam_system1['D']))

            iP2_repro, _ = cv.projectPoints(np.array([np.squeeze(xyz[:-1:1] / xyz[3])]), np.array(cam_system2['Q']),
                                             np.array(cam_system2['translation']), np.array(cam_system2['K']),
                                             np.array(cam_system2['D']))

            # watch out here because pytohn will silently broadcast this subtraction and give wrong error results.
            # Because opencv returns annoying list of list
            iP1_err = np.sqrt(np.sum((iP1_repro - iP1[i]) ** 2))
            iP2_err = np.sqrt(np.sum((iP2_repro - iP2[i]) ** 2))

            tot_error1 += iP1_err
            tot_error2 += iP2_err

            pterr.append(iP1_err)

        print("total error %s: %f" % (self.files[0], tot_error1 / len(iP1)))
        print("total error %s: %f" % (self.files[1], tot_error2 / len(iP1)))

        return pterr

    def calibrate_intrinsics(self):

        # isolate the object points
        checkerBoardPoints = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3),  dtype='float32')
        for kk in range(self.pattern_size[1]):
            for jj in range(self.pattern_size[0]):
                checkerBoardPoints[kk * self.pattern_size[0] + jj] = \
                    [jj * self.pattern_length, kk * self.pattern_length, 0] #x, y, z

        for camname, camdata in self.data.items():
            object_points = []
            camera_points = []

            for frame, points in camdata.items():

                campts = np.array([np.array(val, dtype='float32') for _,val in points.items()], dtype='float32')

                # only use views where all calibs can be seen
                if campts.shape[0] == self.pattern_size[0]*self.pattern_size[1]:
                    objp = [checkerBoardPoints[ptname] for ptname,_ in points.items()]

                    object_points.append(objp)
                    camera_points.append(campts)

            object_points = np.array(object_points)
            object_points = object_points.astype('float32')
            camera_points = np.array(camera_points)
            camera_points = camera_points.astype('float32')

            if object_points.shape[0] > 0:

                ret, P, D, R, T = cv.calibrateCamera(object_points, camera_points,
                                                               tuple(self.camera_info[camname]['Resolution']),
                                                                None, None, None, None,
                                                                self.calibration_criteria)

                print('Cam %s' % camname)
                rmse = self.compute_rmse_fromcv(object_points, camera_points, P, D, T, R)

                self.camera_intrinsics[camname]['Projection'] = P
                self.camera_intrinsics[camname]['Distortion'] = D
                self.camera_intrinsics[camname]['Rotation'] = R
                self.camera_intrinsics[camname]['Translation'] = T


    def calibrate_extrinsics(self):

        camera_matrix_1 = self.camera_intrinsics[self.files[0]]['Projection']
        camera_matrix_2 = self.camera_intrinsics[self.files[1]]['Projection']

        dist_coeff_1 = self.camera_intrinsics[self.files[0]]['Distortion']
        dist_coeff_2 = self.camera_intrinsics[self.files[1]]['Distortion']

        # isolate the object points
        checkerBoardPoints = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), dtype='float32')
        for kk in range(self.pattern_size[1]):
            for jj in range(self.pattern_size[0]):
                checkerBoardPoints[kk * self.pattern_size[0] + jj] = \
                    [jj * self.pattern_length, kk * self.pattern_length, 0]  # x, y, z

        cp = []
        for camname, camdata in self.data.items():
            object_points = []
            camera_points = []

            for frame, points in camdata.items():

                campts = np.array([np.array(val, dtype='float32') for _, val in points.items()], dtype='float32')

                # only use views where all calibs can be seen
                if campts.shape[0] == self.pattern_size[0] * self.pattern_size[1]:
                    objp = [checkerBoardPoints[ptname] for ptname, _ in points.items()]

                    object_points.append(objp)
                    camera_points.append(campts)

                else:
                    object_points.append([])
                    camera_points.append([])

            cp.append(camera_points)

        camera_points1 = []
        camera_points2 = []
        object_points_out = []
        for cp1, cp2, op in zip(cp[0], cp[1], object_points):
            tmp1 = []
            tmp2 = []
            tmp3 = []
            for pt1, pt2, obj in zip(cp1, cp2, op):
                if (len(pt1) > 0) & (len(pt2) > 0):
                    tmp1.append(pt1)
                    tmp2.append(pt2)
                    tmp3.append(obj)

            if len(tmp1) > 0:
                camera_points1.append(tmp1)
                camera_points2.append(tmp2)
                object_points_out.append(tmp3)

        camera_points1 = np.array(camera_points1, dtype='float32')
        camera_points2 = np.array(camera_points2, dtype='float32')
        object_points_out = np.array(object_points_out, dtype='float32')

        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
            cv.stereoCalibrate(object_points_out, camera_points1, camera_points2,
                               camera_matrix_1, dist_coeff_1, camera_matrix_2,
                               dist_coeff_2, self.camera_info[self.files[0]]['Resolution'],
                               self.calibration_criteria)

        projection_matrix1 = np.array(
            np.matrix(cameraMatrix1) * np.matrix(np.append(np.eye(3), np.zeros((3, 1)), axis=1)))
        projection_matrix2 = np.array(np.matrix(cameraMatrix2) * np.matrix(np.append(R, T, axis=1)))

        cam_system1 = {'P': np.append(cameraMatrix1, [[0], [0], [0]], axis=1)}
        cam_system1['projection_matrix'] = projection_matrix1
        cam_system1['Q'] = np.eye(3)
        cam_system1['translation'] = np.zeros((3, 1))
        cam_system1['K'] = cameraMatrix1
        cam_system1['D'] = distCoeffs1

        cam_system2 = {'P': np.append(cameraMatrix2, [[0], [0], [0]], axis=1)}
        cam_system2['projection_matrix'] = projection_matrix2
        cam_system2['Q'] = R
        cam_system2['translation'] = T
        cam_system2['K'] = cameraMatrix2
        cam_system2['D'] = distCoeffs2

        self.camera_extrinsics = [{'projection_matrix': projection_matrix1.tolist(),
                                'P': np.append(cameraMatrix1, [[0], [0], [0]], axis=1).tolist(),
                                'K': camera_matrix_1.tolist(),
                                'D': dist_coeff_1.tolist(),
                                'R': np.eye(3).tolist(),
                                'Q': np.eye(3).tolist(),
                                'translation': np.squeeze(np.zeros((3, 1))).tolist(),
                                'name': 'cam1'}]
        self.camera_extrinsics.append({'projection_matrix': projection_matrix2.tolist(),
                                    'P': np.append(cameraMatrix2, [[0], [0], [0]], axis=1).tolist(),
                                    'K': cameraMatrix2.tolist(),
                                    'D': distCoeffs2.tolist(),
                                    'R': np.eye(3).tolist(),  # no rectification matrix
                                    'Q': R.tolist(),
                                    'translation': np.squeeze(T).tolist(),
                                    'name': 'cam' + str(jj + 1)})

        print('\r\nAll the points in images\r\n')
        pterr = self.compute_rmse_stereo(np.reshape(camera_points1, (16 * camera_points1.shape[0], 2)),
                                      np.reshape(camera_points2, (16 * camera_points2.shape[0], 2)),
                                      cam_system1, cam_system2)

        return camera_points1, camera_points2, object_points_out

class MultiDataView():
    def __init__(self, video,
                 video_contours,
                 transpose=False,
                 contours_data=None,
                 associated_data=None,
                 view=None,
                 fname=None):

        app = pg.mkQApp()

        self.video_stream = MultiVideoStreamView(video,
                                 video_contours,
                                 transpose=True,
                                 contours_data=contours_data,
                                 fname=fname)

        self.layout = pg.LayoutWidget()

        self.w1 = self.layout.addWidget(self.video_stream, row=0, col=0)
        self.layout.resize(1200, 600)
        self.layout.show()

    def update_data(self, index):
        self.tree.setData(self.data[index])


class MultiVideoStreamView(VideoStreamView):

    sigHandTrackLoaded = QtCore.Signal(object)

    def __init__(self, video, video_contours, transpose = False, contours_data = None,
                 associated_data = None, view = None, fname = None):
        super(MultiVideoStreamView, self).__init__(video, video_contours, transpose, contours_data,
                 associated_data, view, fname)

        self.imageItem.mouseClickEvent = self.ms_click2


    def ms_click2(self, event):
        """
        Handles the mouse click events.
            - Shift + click records a point to the selected handtracked row
        """

        for p in self.lastClicked:
            p.resetPen()

        print("clc")
        event.accept()

        if event.button() == QtCore.Qt.LeftButton:
            pos = event.pos()
            x = pos.x()
            y = pos.y()

            print(int(x), int(y))

            modifiers = QtGui.QApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.ShiftModifier:
                print('Shift+Click')

                element = self.tree.selectionModel().currentIndex()

                if element.row() != -1:
                    var_name = self.tree.item(element.row(), 0).value

                    self.hand_tracked_points.add_xy_point(key = var_name, x = x, y = y, frame= self.currentIndex)

                    self.updateImage()
                    self.tree.selected_row = element.row()
                    self.tree.setData(self.hand_tracked_points.return_items(self.currentIndex))

                    self.sigHandtrackPointValChanged.emit(x,y)

            elif modifiers == QtCore.Qt.ControlModifier:
                print('Control+Click')

                corners = np.array([[x, y]]).astype('float32')
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # for subpixel detector
                # dont centroid vertical post
                cv.cornerSubPix(self.video[self.currentIndex], corners, (11, 11), (-1, -1), criteria)
                x, y = corners[0][0], corners[0][1]

                element = self.tree.selectionModel().currentIndex()

                if element.row() != -1:
                    var_name = self.tree.item(element.row(), 0).value

                    self.hand_tracked_points.add_xy_point(key = var_name, x = x, y = y, frame= self.currentIndex)

                    self.updateImage()
                    self.tree.selected_row = element.row()
                    self.tree.setData(self.hand_tracked_points.return_items(self.currentIndex))


                    self.sigHandtrackPointValChanged.emit(x,y)
                    event.accept()

    def load_handtrack(self):
        print('Overloaded')

        name = os.path.join(self.file_path, self.hand_track_file_name)

        if os.path.exists(name) is True:
            name = QtGui.QFileDialog.getOpenFileName(self, 'Save File', self.file_path)
            name = name[0]

        print(name)
        if name is not '':
            with open(name, 'r') as input:
                data = json.load(input)
                self.hand_tracked_points.set_data(data)
                self.hand_tracked_points.add_keyed_point('')
        else:
            event.ignore()
            return

        self.sigHandTrackLoaded.emit(data)

    def other_file_loaded(self, data):
        if self.hand_tracked_points.data_entered() == False:

            if self.data != []:
                for name, vars in data.items():
                    self.hand_tracked_points.add_keyed_point(name)

                    for pts in vars['x']:
                        self.hand_tracked_points.add_xy_point(key=name,
                                                              x=-2, y=-2,
                                                              frame=vars['frames'])