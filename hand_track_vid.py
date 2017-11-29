
from Tracker.Tracker import Tracker
from pyqtgraph.Qt import QtCore, QtGui
import sys

app = QtGui.QApplication([])

file = sys.argv[1]
print(file)

# Load in images to memory during construction
video = Tracker(file, verbose='True', min_object_size=20)

# video.threshold_val = 0.5

video.load_video()
video.visualize()

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()

