"""
Demonstrate the use of layouts to control placement of multiple plots / views /
labels


"""

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

import Visualize as visualization
import cv2 as cv

app = QtGui.QApplication([])
view = pg.GraphicsView()
l = pg.GraphicsLayout(border=(100,100,100))
view.setCentralItem(l)
view.show()
view.setWindowTitle('pyqtgraph example: GraphicsLayout')
view.resize(800,600)

## Title at top
text = """
This example demonstrates the use of GraphicsLayout to arrange items in a grid.<br>
The items added to the layout must be subclasses of QGraphicsWidget (this includes <br>
PlotItem, ViewBox, LabelItem, and GrphicsLayout itself).
"""
l.addLabel(text, col=1, colspan=4)

l.nextRow()

## Put vertical label on left side
l.addLabel('Long Vertical Label', angle=-90, rowspan=3)

## Add 3 plots into the first row (automatic position)
p1 = l.addPlot(title="Plot 1")
p2 = l.addPlot(title="Plot 2")

file = '/Users/nickgravish/Dropbox/Harvard/HighThroughputExpt/' \
       'Bee_experiments_2016/2016-08-15_13.05.57/' \
       '1_08-15-16_13-06-05.015_Mon_Aug_15_13-05-57.148_2.mp4'

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

video = visualization.VideoStreamView(frames)

l.addWidget(video)



## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
