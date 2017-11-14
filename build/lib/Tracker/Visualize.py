


import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

from pyqtgraph import ptime as ptime

import cv2 as cv
import os


class VideoStreamView(pg.ImageView):
    """
    This will take in a video container that will handle the loading. This way
    VideoStreamView class is agnostic to how videos are handled/loaded

    Changed functionality a bit, to be able to load in streams to memory, or to stream from disk, no
    longer treating "image" as 3D object, but instead just as the image. Thus when referencing timing of
    video, ened to use video.shape[0]


    """
    sigIndexChanged = QtCore.Signal(object)

    def __init__(self, video, transpose = False, contours_data = None, associated_data = None, view = None):
        super().__init__(view = view)
        pg.setConfigOptions(antialias=True)

        self.video = video
        self.transpose = transpose

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

        self.plot_items = []


        self.image = None
        self.loadFrame(1)
        self.setImage(self.image)
        self.currentIndex = 0

        # override the wheel event zoom functionality so that can be used for timeline changnig
        self.ui.roiPlot.wheelEvent = self.wheelEvent

        self.imageItem.mouseClickEvent = self.ms_click


    def updatePoints(self):

        if not self.contours:

            curr_x = self.contours[self.currentIndex]['x']
            curr_y = self.contours[sel  f.currentIndex]['y']

            self.plot_points(curr_x, curr_y)

            print(curr_x)

    def wheelEvent(self, ev):
        sc = ev.angleDelta().y()/8
        self.jumpFrames(sc)
        self.timeLine.getBounds()

    def loadFrame(self, index):

        if self.is_array:
            img = self.video[index, :,:]
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

        self.updatePoints()

    def updateImage(self, autoHistogramRange=True):
        ## Redraw image on screen
        if self.image is None:
            return

        image = self.getProcessedImage()

        if autoHistogramRange:
            self.ui.histogram.setHistogramRange(self.levelMin, self.levelMax)

        self.imageItem.updateImage(self.image)
        self.ui.roiPlot.show()


    def setCurrentIndex(self, ind):
        """Set the currently displayed frame index."""
        self.currentIndex = np.clip(ind, 0, self.NumFrames - 1)

        self.loadFrame(self.currentIndex)

        self.updateImage()
        self.ignoreTimeLine = True
        self.timeLine.setValue(self.tVals[self.currentIndex])
        self.ignoreTimeLine = False
        self.sigIndexChanged.emit(self.currentIndex)

    def keyPressEvent(self, ev):
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

    def plot_points(self, x = [], y = []):

        if not x:
            x = [self.Height/ 2]
            y = [self.Width/ 2]
            print('xx')

        plt = pg.PlotDataItem(x, y, symbol='o', symbolBrush=(255, 0, 0), symbolPen='w')
        plt.setParentItem(self.imageItem)

        self.plot_items.append(plt)
        print(len(self.plot_items))


    def ms_click(self, event):
        event.accept()
        pos = event.pos()

        x = pos.x()
        y = pos.y()

        print(int(x), int(y))

        # self.plot_points([x], [y])
        #
        #
        # for plt in self.plot_items:
        #     plt.setSymbolBrush(np.random.randint(0,254,1), np.random.randint(0,254,1), np.random.randint(0,254,1))
        #
        # self.updateImage()


# def click(event):
#     event.accept()
#     pos = event.pos()
#     print(int(pos.x()), int(pos.y()))


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



