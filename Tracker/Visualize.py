


import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

from pyqtgraph import ptime as ptime

import cv2 as cv
import os
import json
import os.path
import copy

class VideoDataView():
    """
    Small class to merge a data stream view and a video view
    """

    def __init__(self, video,
                 video_contours,
                 transpose=False,
                 contours_data=None,
                 associated_data=None,
                 view=None,
                 fname=None):
        app = pg.mkQApp()

        self.video_stream = VideoStreamView(video,
                                 video_contours,
                                 transpose=True,
                                 contours_data=contours_data,
                                 fname=fname)

        # self.data = contours_data
        # self.tree = pg.DataTreeWidget(data=self.data[0])

        self.layout = pg.LayoutWidget()

        self.w1 = self.layout.addWidget(self.video_stream, row=0, col=0)
        # self.w2 = self.layout.addWidget(self.tree, row=0, col=1)
        #
        # qGraphicsGridLayout = self.layout.layout
        # qGraphicsGridLayout.setColumnStretch(0, 2.5)
        # qGraphicsGridLayout.setColumnStretch(1, 1)

        self.layout.resize(1200, 600)
        self.layout.show()

    def update_data(self, index):
        self.tree.setData(self.data[index])


class DataTree(pg.DataTreeWidget):

    def __init__(self, data = None):

        super().__init__(data = data)

        self.setEditTriggers(self.NoEditTriggers)

        # to be able to decide on your own whether a particular item
        # can be edited, connect e.g. to itemDoubleClicked
        self.itemDoubleClicked.connect(self.checkEdit)


    def collapseTree(self, item):
        item.setExpanded(False)
        for i in range(item.childCount()):
            self.collapseTree(item.child(i))

    def listAllItems(self, item=None):
        items = []
        if item != None:
            items.append(item)
        else:
            item = self.invisibleRootItem()

        for cindex in range(item.childCount()):
            foundItems = self.listAllItems(item=item.child(cindex))
            for f in foundItems:
                items.append(f)
        return items

    def checkEdit(self, item, column):
        # e.g. to allow editing only of column 1:
        print('edit')

        self.editItem(item, column)


class HandTrackPoints():

    def __init__(self, num_points = None):

        self.num_points = num_points

        if num_points is not None:
            self.data = {'': {'x': np.ones(num_points)*(-1), 'y': np.ones(num_points)*(-1)},
                         }
        else:
            self.data = {'': {'x': [], 'y': []}}


    def add_xy_point(self, key = None, x = None, y = None, frame = None):
        if key is not None:
            self.data[key]['x'][frame] = x
            self.data[key]['y'][frame] = y

    def remove_xy_point(self, key = None, frame = None):
        if key is not None:
            self.data[key]['x'][frame] = -1
            self.data[key]['y'][frame] = -1


    def add_keyed_point(self, key):
        self.data[key] = {'x': np.ones(self.num_points)*(-1),
                          'y': np.ones(self.num_points)*(-1)}

    def update_key(self, oldkey, newkey):
        self.data[newkey] = self.data[oldkey]
        del self.data[oldkey]

        if oldkey == '':
            self.add_keyed_point('')

    def return_items(self, frame):
        return_data = []

        for key, value in self.data.items():

            x = value['x'][frame]
            y = value['y'][frame]

            if x == -1:
                x = ''
                y = ''

            return_data.append({'Variable name': key,
                                'x': x,
                                'y': y})
        return return_data

    def set_data(self, data):
        self.data = data

    def return_json(self):

        # remove the empty key
        tmp = dict(self.data)
        try:
            del tmp['']
        except:
            []

        return tmp

        # return_data = []
        #
        # for key, value in self.data.items():
        #
        #     x = value['x'].tolist()
        #     y = value['y'].tolist()
        #
        #     if x == -1:
        #         x = ''
        #         y = ''
        #
        #     return_data.append({'Variable name': key,
        #                         'x': x,
        #                         'y': y})
        #
        # return return_data

    def return_next_keyframe(self, frame, direction=1):

        if direction == 1:
            for k in range(frame, self.num_points, direction):
                for key, value in self.data.items():
                    # just need to check if x is empty
                    x = value['x'][k]

                    if x != -1:
                        return k

            return self.num_points
        else:
            for k in range(frame, 0, direction):
                for key, value in self.data.items():
                    # just need to check if x is empty
                    x = value['x'][k]

                    if x != -1:
                        return k
            return 0


class HandTrackTable(pg.TableWidget):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.last_key = []
        self.selected_row = 0

    def appendData(self, data):
        self.blockSignals(True)
        super().appendData(data)
        self.setEditable()
        self.blockSignals(False)

    def setEditable(self, editable=True):
        self.editable = editable
        for item in self.items:
            if item.column() == 0:
                item.setEditable(editable)
            else:
                item.setEditable(False)

    def setData(self, data):
        super().setData(data)
        self.setCurrentCell(self.selected_row, 0)

    def handleItemChanged(self, item):
        self.last_key = item.value
        item.itemChanged()
        self.selected_row = item.row()

class VideoStreamView(pg.ImageView):
    """
    This will take in a video container that will handle the loading. This way
    VideoStreamView class is agnostic to how videos are handled/loaded

    Changed functionality a bit, to be able to load in streams to memory, or to stream from disk, no
    longer treating "image" as 3D object, but instead just as the image. Thus when referencing timing of
    video, ened to use video.shape[0]


    """

    sigIndexChanged = QtCore.Signal(object)
    sigHandtrackPointChanged = QtCore.Signal(object, object, object)

    def __init__(self, video, video_contours, transpose = False, contours_data = None,
                 associated_data = None, view = None, fname = None):

        super().__init__(view = view)
        pg.setConfigOptions(antialias=True)

        self.last_key = None
        self.last_item = None

        self.video = video
        self.video_contours = video_contours
        self.transpose = transpose

        self.videoname = fname
        self.file_path = os.path.dirname(self.videoname)
        self.hand_track_file_name = os.path.splitext(os.path.basename(self.videoname))[0] + '_handtrack.txt'

        if type(video) == np.ndarray:
            self.video = video
            self.NumFrames, self.Height, self.Width = self.video.shape
            self.is_array = True

        else:
            self.NumFrames = self.video.getNumFrames()
            self.Height = self.video.getHeight()
            self.Width = self.video.getWidth()
            self.is_array = False

        self.contours = contours_data
        self.associated_data = associated_data

        self.contour_plot_items = []
        self.hand_track_plot_items = []
        self.association_plot_items = []

        self.contour_plots = pg.PlotDataItem([], [], symbol='o', symbolBrush=None, symbolPen={'color': 'k', 'width':2}
                                    , pen = None, symbolSize = 10)
        self.contour_plots.setParentItem(self.imageItem)
        self.lastClicked = []
        self.contour_plots.sigPointsClicked.connect(self.clicked)

        self.hand_track_plots = pg.PlotDataItem([], [], symbol='o', symbolBrush=None, symbolPen={'color': 'r', 'width':3}
                                    , pen = None, symbolSize = 14)
        self.hand_track_plots.setParentItem(self.imageItem)
        self.hand_track_plots.sigPointsClicked.connect(self.clicked)


        self.hand_tracked_points = HandTrackPoints(num_points=self.NumFrames)

        # add the tree for visualizing points
        # self.tree = pg.TableWidget(editable=True)
        self.tree = HandTrackTable(editable = True)
        self.tree.setData(self.hand_tracked_points.return_items(self.currentIndex))
        self.tree.itemChanged.connect(self.update_handtrack)

        self.data_tree = DataTree(data=self.contours)
        self.data_tree .setData(self.contours[0], hideRoot=True)
        self.data_tree.collapseTree(self.data_tree.invisibleRootItem())

        self.splitter = QtGui.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.ui.gridLayout.addWidget(self.splitter, 0, 3)
        self.ui.gridLayout.setColumnStretch(3, 0.4)
        self.splitter.addWidget(self.data_tree)
        self.splitter.setSizes([100, 35])

        self.splitter2 = QtGui.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.addWidget(self.splitter2)
        self.splitter2.addWidget(self.tree)


        self.handtrack_button = QtGui.QCheckBox()
        self.handtrack_button.clicked.connect(self.updateImage)

        self.contour_button = QtGui.QCheckBox()
        self.contour_button.clicked.connect(self.updateImage)

        self.saveBtn = QtGui.QPushButton()
        self.saveBtn.clicked.connect(self.save_handtrack)

        self.loadBtn = QtGui.QPushButton()
        self.loadBtn.clicked.connect(self.load_handtrack)

        self.handtrack_button.setText("Handtrack")
        self.contour_button.setText("Contours")
        self.saveBtn.setText("Save")
        self.loadBtn.setText("Load")

        self.buttons = QtGui.QSplitter()
        # self.buttons = QtGui.QGridLayout()
        # self.buttons.setStretchFactor(0, 0)

        self.splitter.addWidget(self.buttons)
        # self.ui.gridLayout.addWidget(self.buttons, 1, 3)

        self.buttons.addWidget(self.handtrack_button)
        self.buttons.addWidget(self.contour_button)
        self.buttons.addWidget(self.saveBtn)
        self.buttons.addWidget(self.loadBtn)

        self.image = None
        self.loadFrame(1)
        self.setImage(self.image)
        self.currentIndex = 0

        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                             QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown, QtCore.Qt.Key_W, QtCore.Qt.Key_Q,
                             QtCore.Qt.Key_E, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D, QtCore.Qt.Key_R]

        # self.splitter.setColumnStretch(4, 0.5)
        # self.ui.gridLayout.setColumnMinimumWidth(4, 200)

# override the wheel event zoom functionality so that can be used for timeline changnig
        self.ui.roiPlot.wheelEvent = self.wheelROIEvent

        self.imageItem.mouseClickEvent = self.ms_click
        self.imageItem.installEventFilter(self)

    def save_handtrack(self):

        name = os.path.join(self.file_path, self.hand_track_file_name)

        if os.path.exists(name) is True:
            name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')

        print(name)
        with open(name[0], 'w+') as output:
            json.dump(self.hand_tracked_points.return_json(),
                      output,
                      sort_keys=True,
                      indent=4)

    def load_handtrack(self):

        name = QtGui.QFileDialog.getOpenFileName(self, 'Save File')

        print(name)
        with open(name[0], 'r') as input:
            data = json.load(input)
            self.hand_tracked_points.set_data(data)
            self.hand_tracked_points.add_keyed_point('')

    def update_handtrack(self, item, value = None, last_key = None):
        """
        Has functionality to g 
        """
        # for tying together multiple tables

        if last_key is None:
            self.last_key = self.tree.last_key
        else:
            self.last_key = last_key

        try:
            print("item passed")
            print(item.row())
            print(item.column())
            print(item.value)
            self.last_item = item.value
            self.hand_tracked_points.update_key(self.last_key, self.last_item)
            self.tree.setData(self.hand_tracked_points.return_items(self.currentIndex))

            # only emit if organically called
            self.sigHandtrackPointChanged.emit(item, self.last_item, self.last_key)
        except:
            print("item triggered", value)
            print(self.last_key)
            self.last_item = value
            self.hand_tracked_points.update_key(self.last_key, self.last_item)
            self.tree.setData(self.hand_tracked_points.return_items(self.currentIndex))

    def eventFilter(self, obj, event):
        """
        Handle only necessary mouse clicks 
        """
        if event.type() in (QtCore.QEvent.MouseButtonPress,
                            QtCore.QEvent.MouseButtonDblClick):
            if event.button() == QtCore.Qt.RightButton:
                print("right")
                return True
        return super().eventFilter(obj, event)



    def wheelROIEvent(self, ev):
        sc = ev.angleDelta().y()/8
        self.jumpFrames(sc)
        ev.accept()
        # self.timeLine.getBounds()

    def loadFrame(self, index):

        if self.is_array and self.contour_button.isChecked():
            img = self.video_contours[int(index), :,:]
        elif self.is_array:
            img = self.video[int(index), :, :]
        else:
            img = self.video.getFrame(index)

        if self.transpose:
            img = img.T

        self.image = img


        #
        # if self.videomode == True :
        #     self.vid.set(cv.CAP_PROP_POS_FRAMES, index)
        #     tru, img = self.vid.read(1)
        #     img = img[:, :, 0]
        #
        # else:
        #     img = cv.imread(os.path.join(self.videostream, self.file_list[index]), cv.IMREAD_GRAYSCALE)
        #     self.image = np.array(img)
        #
        #     if self.image is not None:
        #         tru = True
        #
        # if tru:
        #     if self.transpose:
        #         img = img.T
        #
        #     self.image = img


    def jumpFrames(self, n):
        """Move video frame ahead n frames (may be negative)"""
        self.setCurrentIndex(self.currentIndex + n)

    # override so that will draw times
    def setImage(self, img, autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None,
                 transform=None, autoHistogramRange=True):
        """
        Set the image to be displayed in the widget.

        ================== =======================================================================
        **Arguments:**
        img                (numpy array) the image to be displayed.
        xvals              (numpy array) 1D array of z-axis values corresponding to the third axis
                           in a 3D image. For video, this array should contain the time of each frame.
        autoRange          (bool) whether to scale/pan the view to fit the image.
        autoLevels         (bool) whether to update the white/black levels to fit the image.
        levels             (min, max); the white and black level values to use.
        axes               Dictionary indicating the interpretation for each axis.
                           This is only needed to override the default guess. Format is::

                               {'t':0, 'x':1, 'y':2, 'c':3};

        pos                Change the position of the displayed image
        scale              Change the scale of the displayed image
        transform          Set the transform of the displayed image. This option overrides *pos*
                           and *scale*.
        autoHistogramRange If True, the histogram y-range is automatically scaled to fit the
                           image data.
        ================== =======================================================================
        """

        if hasattr(img, 'implements') and img.implements('MetaArray'):
            img = img.asarray()

        if not isinstance(img, np.ndarray):
            required = ['dtype', 'max', 'min', 'ndim', 'shape', 'size']
            if not all([hasattr(img, attr) for attr in required]):
                raise TypeError("Image must be NumPy array or any object "
                                "that provides compatible attributes/methods:\n"
                                "  %s" % str(required))

        self.image = img
        self.imageDisp = None

        self.tVals = np.arange(self.NumFrames)


        self.axes = {'t': 0, 'x': 1, 'y': 2, 'c': None}

        for x in ['t', 'x', 'y', 'c']:
            self.axes[x] = self.axes.get(x, None)


        self.currentIndex = 0
        self.updateImage(autoHistogramRange=autoHistogramRange)
        if levels is None and autoLevels:
            self.autoLevels()
        if levels is not None:  ## this does nothing since getProcessedImage sets these values again.
            self.setLevels(*levels)

        if self.ui.roiBtn.isChecked():
            self.roiChanged()


        if self.axes['t'] is not None:
            # self.ui.roiPlot.show()
            self.ui.roiPlot.setXRange(self.tVals.min(), self.tVals.max())
            self.timeLine.setValue(0)
            # self.ui.roiPlot.setMouseEnabled(False, False)
            if len(self.tVals) > 1:
                start = self.tVals.min()
                stop = self.tVals.max() + abs(self.tVals[-1] - self.tVals[0]) * 0.02
            elif len(self.tVals) == 1:
                start = self.tVals[0] - 0.5
                stop = self.tVals[0] + 0.5
            else:
                start = 0
                stop = 1
            for s in [self.timeLine, self.normRgn]:
                s.setBounds([start, stop])
                # else:
                # self.ui.roiPlot.hide()

        self.imageItem.resetTransform()
        if scale is not None:
            self.imageItem.scale(*scale)
        if pos is not None:
            self.imageItem.setPos(*pos)
        if transform is not None:
            self.imageItem.setTransform(transform)


        if autoRange:
            self.autoRange()
        self.roiClicked()

    def updateImage(self, autoHistogramRange=True):
        ## Redraw image on screen
        if self.image is None:
            return

        image = self.getProcessedImage()

        if autoHistogramRange:
            self.ui.histogram.setHistogramRange(self.levelMin, self.levelMax)

        self.imageItem.updateImage(self.image)
        self.updatePoints()
        self.ui.roiPlot.show()


    def setCurrentIndex(self, ind):
        """Set the currently displayed frame index."""

        # break out of inf loops from connected timelines signal
        if self.currentIndex == ind:
            return

        self.currentIndex = int(np.clip(ind, 0, self.NumFrames - 1))

        self.loadFrame(self.currentIndex)

        self.updateImage()
        self.ignoreTimeLine = True
        self.timeLine.setValue(self.tVals[self.currentIndex])
        self.ignoreTimeLine = False

        self.tree.setData(self.hand_tracked_points.return_items(self.currentIndex))

        # self.tree.setData(self.contours[self.currentIndex], hideRoot=True)
        # self.tree.collapseTree(self.tree.invisibleRootItem())

        self.sigIndexChanged.emit(self.currentIndex)

    def keyPressEvent(self, ev):
        """
        Handles initial key press decisions. If one was registered calls evalKeyState
        """
        # print ev.key()
        if ev.key() == QtCore.Qt.Key_Space:
            if self.playRate == 0:
                fps = 30
                self.play(fps)
                # print fps
            else:
                self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_Home:
            self.setCurrentIndex(0)
            self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_End:
            self.setCurrentIndex(self.NumFrames - 1)
            self.play(0)
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
        else:
            QtGui.QWidget.keyPressEvent(self, ev)

    def evalKeyState(self):
        """
        Handle keypresses:
        - Up/Down or w/s jump ahead 10 frames
        - Left/Right or a/d jump ahead 1 frame
        - Space plays movie at 30 fps
        - End/Home jump to beginning or end
        - q/e jump to next keyframe
        - r erases the selected point
        """

        if len(self.keysPressed) == 1:
            key = list(self.keysPressed.keys())[0]
            if key in [QtCore.Qt.Key_Right, QtCore.Qt.Key_D]:
                self.play(20)
                self.jumpFrames(1)
                self.lastPlayTime = ptime.time() + 0.2  ## 2ms wait before start
                ## This happens *after* jumpFrames, since it might take longer than 2ms
            elif key in [QtCore.Qt.Key_Left, QtCore.Qt.Key_A]:
                print("left")
                self.play(-20)
                self.jumpFrames(-1)
                self.lastPlayTime = ptime.time() + 0.2
            elif key in [QtCore.Qt.Key_Up, QtCore.Qt.Key_W]:
                self.play(-100)
            elif key in [QtCore.Qt.Key_Down, QtCore.Qt.Key_S]:
                self.play(100)
            elif key == QtCore.Qt.Key_PageUp:
                self.play(-1000)
            elif key == QtCore.Qt.Key_PageDown:
                self.play(1000)
            elif key == QtCore.Qt.Key_Q:
                # jump to next keyframe forward
                print("Q")
                frames = self.hand_tracked_points.return_next_keyframe(self.currentIndex-1, -1)
                self.play(20)
                self.setCurrentIndex(frames)
                self.lastPlayTime = ptime.time() + 0.2  ## 2ms wait before start
                ## This happens *after* jumpFrames, since it might take longer than 2ms

            elif key == QtCore.Qt.Key_E:
                # jump to next keyframe reverse
                print("E")
                frames = self.hand_tracked_points.return_next_keyframe(self.currentIndex+1, 1)
                self.play(20)
                self.setCurrentIndex(frames)
                self.lastPlayTime = ptime.time() + 0.2  ## 2ms wait before start

            elif key == QtCore.Qt.Key_R:
                # jump to next keyframe forward
                print("R")
                element = self.tree.selectionModel().currentIndex()
                var_name = self.tree.item(element.row(), 0).value
                self.hand_tracked_points.remove_xy_point(key=var_name, frame=self.currentIndex)


        else:
            self.play(0)


    def timeLineChanged(self):
        # (ind, time) = self.timeIndex(self.ui.timeSlider)
        if self.ignoreTimeLine:
            return
        self.play(0)
        (ind, time) = self.timeIndex(self.timeLine)
        if ind != self.currentIndex:
            self.setCurrentIndex(ind)
            self.updateImage()
        # self.timeLine.setPos(time)
        # self.emit(QtCore.SIGNAL('timeChanged'), ind, time)
        self.sigTimeChanged.emit(ind, time)


    def timeout(self):
        now = ptime.time()
        dt = now - self.lastPlayTime
        if dt < 0:
            return
        n = int(self.playRate * dt)

        if n != 0:
            self.lastPlayTime += (float(n)/self.playRate)
            if self.currentIndex+n > self.NumFrames:
                self.play(0)
            self.jumpFrames(n)


    def updatePoints(self):
        self.contour_plots.clear()
        self.hand_track_plots.clear()

        if self.contours and self.contour_button.isChecked():

            curr_x = []
            curr_y = []

            for c in self.contours[int(self.currentIndex)]:

                curr_x.append(c['x'])
                curr_y.append(c['y'])

            self.contour_plot_items = {'x': curr_x, 'y': curr_y}
            self.contour_plots.setData(self.contour_plot_items['x'], self.contour_plot_items['y'])

        tmp = self.hand_tracked_points.return_items(self.currentIndex)
        var_names = []
        x = []
        y = []

        for element in tmp:
            x_tmp = element['x']
            y_tmp = element['y']

            if x_tmp != '':
                var_names.append(element['Variable name'])
                x.append(x_tmp)
                y.append(y_tmp)

        if x and self.handtrack_button.isChecked():
            self.hand_track_plots.setData(x, y, name=var_names)

            # self.l = pg.LegendItem()
            # self.l.addItem(self.hand_track_plots)


    def clicked(self, plot, points):

        for p in self.lastClicked:
            p.resetPen()

        for p in points:
            p.setPen('b', width=2)
            print("clicked points", p.pos())

        self.lastClicked = points


    def ms_click(self, event):
        """
        Handles the mouse click events.
            - Shift + click records a point to the selected handtracked row

        """

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


            elif modifiers == QtCore.Qt.ControlModifier:
                print('Control+Click')
            elif modifiers == (QtCore.Qt.ControlModifier |
                                   QtCore.Qt.ShiftModifier):
                print('Control+Shift+Click')
            else:
                print('Click')

            return


        else:
            event.ignore()
            return




## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph as pg
    import numpy as np
    import cv2 as cv
    import sys

    app = QtGui.QApplication([])
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()



    file = '/Users/nickgravish/source_code/Tracker/test_data/1_08-15-16_13-06-05.015_Mon_Aug_15_13-05-57.148_2.mp4'


    vid = cv.VideoCapture(file)

    NumFrames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    Height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    Width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))

    frames = np.zeros((10, Height, Width), np.uint8)

    for kk in range(10):
        tru, ret = vid.read(1)

        # check if video frames are being loaded
        if not tru:
            print('Codec issue: cannot load frames.')
            exit()

        frames[kk, :, :] = ret[:, :, 0]  # assumes loading color

    print('Loaded!')

    video = VideoStreamView(frames)

    # p = video.imageItem.addPlot()

    # pg.plot([Width/2], [Height/2], symbolBrush=(255,0,0), symbolPen='w')

    # video.imageItem.

    w = QtGui.QWidget()
    w.resize(1200, 600)
    layout = QtGui.QGridLayout()


    # video.imageItem.mouseClickEvent = click
    #
    # plt = pg.PlotDataItem([Height / 2], [Width / 2], symbol='o', symbolBrush=(255, 0, 0), symbolPen='w')
    # plt.setParentItem(video.imageItem)

    w.setLayout(layout)
    layout.addWidget(video)
    w.show()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



