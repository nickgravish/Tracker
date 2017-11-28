
from Tracker.Tracker import Tracker
from pyqtgraph.Qt import QtCore, QtGui
import sys

app = QtGui.QApplication([])

file = '/Users/nickgravish/source_code/Tracker/test_data/C002H001S0031.avi.mp4'

# Load in images to memory during construction
video = Tracker(file, verbose='True', frame_range=(1000,100), min_object_size=20)

# video.threshold_val = 0.5

video.load_video()
# video.compute_background()  # form background image
# video.remove_background()  # remove background
# video.threshold()  # threshold to segment features
# video.morpho_closing()
# video.find_objects()

video.visualize()

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()

