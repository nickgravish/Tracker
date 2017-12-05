
from Tracker.MultiViewSystem import MultiViewSystem
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import cv2 as cv
import sys


file = '/Users/nickgravish/Dropbox/Harvard/HighThroughputExpt/Bee_Foraging_Experiments_Summer_2015/Photron/08_31_2015_moving_post/C001H001S0017.avi.mp4'
file2 = '/Users/nickgravish/Dropbox/Harvard/HighThroughputExpt/Bee_Foraging_Experiments_Summer_2015/Photron/08_31_2015_moving_post/C002H001S0017.avi.mp4'

# Load in images to memory during construction


m = MultiViewSystem([file, file2])

m.visualize()

print("Back out now")

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()