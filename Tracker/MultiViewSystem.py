
from .Tracker import Tracker
from Tracker.Visualize import *
import pyqtgraph as pg
import cv2 as cv

from pyqtgraph.Qt import QtCore, QtGui

class MultiViewSystem:

    def __init__(self, files):

        self.video1 = Tracker(files[0], verbose='True', frame_range= (1000, 400), min_object_size=20)
        self.video2 = Tracker(files[1], verbose='True', frame_range= (1000, 400), min_object_size=20)

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
        self.camera_info[self.files[0]] = {'Resolution': [self.video1.Width, self.video1.Height]}
        self.camera_info[self.files[1]] = {'Resolution': [self.video2.Width, self.video2.Height]}


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
        self.widget.resize(200, 400)
        self.widget.show()


    def calibrate_system(self):

        print('Calibrating!')

        self.get_hand_trackdata()
        self.tree.setData(self.data)
        self.calibrate_intrinsics()

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

    def calibrate_intrinsics(self):
        pattern_size = (4,4)
        pattern_length = 11.5 #mm

        # isolate the object points
        checkerBoardPoints = np.zeros((pattern_size[0] * pattern_size[1], 3),  dtype='float32')
        for kk in range(pattern_size[1]):
            for jj in range(pattern_size[0]):
                checkerBoardPoints[kk * pattern_size[0] + jj] = [kk * pattern_length, jj * pattern_length, 0] #x, y, z

        for camname, camdata in data.items():
            object_points = []
            camera_points = []

            for frame, points in camdata.items():
                objp = [checkerBoardPoints[ptname] for ptname,_ in points.items()]
                object_points.append(objp)

                campts = [val for _,val in points.items()]
                camera_points.append(campts)

            # object_points = np.array(object_points)
            # object_points = object_points.astype('float32')
            # camera_points = np.array(camera_points)
            # camera_points = camera_points.astype('float32')

            if object_points:

                ret, P, D, R, translation = cv.calibrateCamera([object_points], [camera_points], camera_info[camname]['Resolution'],
                                                                None, None, None, None,
                                                                calibration_criteria)

                rmse = self.compute_rmse_fromcv(oP, iP, P, D, translation, R)
                print(rmse)



            # self.write_intrinsics_to_json(P, D, resolution[cam][0], resolution[cam][1], \
            #                          os.path.join(basedir, calib_video[cam] + 'CALIB.txt'))







print("test")


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