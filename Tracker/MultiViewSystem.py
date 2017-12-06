
from .Tracker import Tracker
from Tracker.Visualize import *
import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtGui

class MultiViewSystem:

    def __init__(self, files):

        self.video1 = Tracker(files[0], verbose='True', frame_range= (1000, 400), min_object_size=20)
        self.video2 = Tracker(files[1], verbose='True', frame_range= (1000, 400), min_object_size=20)

        self.video1.load_video()
        self.video2.load_video()


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

        for name, vars in data.items():
            self.hand_tracked_points.add_keyed_point(name)

            for pts in vars['x']:
                self.hand_tracked_points.add_xy_point(key=name,
                                                      x=-2, y=-2,
                                                      frame=vars['frames'])