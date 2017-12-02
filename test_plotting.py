
from Tracker.Tracker import Tracker
from Tracker.Visualize import HandTrackPoints
from pyqtgraph.Qt import QtCore, QtGui
import sys

app = QtGui.QApplication([])

file = '/Users/nickgravish/source_code/Tracker/test_data/C001H001S0031.avi.mp4'
file2 = '/Users/nickgravish/source_code/Tracker/test_data/C002H001S0031.avi.mp4'

# Load in images to memory during construction
video = Tracker(file, verbose='True', frame_range=(1000,100), min_object_size=20)
video2 = Tracker(file2, verbose='True', frame_range=(1000,100), min_object_size=20)

# video.threshold_val = 0.5

video.load_video()
video2.load_video()
# video.compute_background()  # form background image
# video.remove_background()  # remove background
# video.threshold()  # threshold to segment features
# video.morpho_closing()
# video.find_objects()

vv1 = video.visualize()
vv2 = video2.visualize()

# @QtCore.pyqtSlot(int)
# def link_timelines(index):
#     print("Change")
#     vv1.video_stream.setCurrentIndex(index)
#     vv2.video_stream.setCurrentIndex(index)

vv1.video_stream.sigIndexChanged.connect(vv2.video_stream.setCurrentIndex)
vv2.video_stream.sigIndexChanged.connect(vv1.video_stream.setCurrentIndex)

vv1.video_stream.sigHandtrackPointNameChanged.connect(vv2.video_stream.update_handtrack)
vv2.video_stream.sigHandtrackPointNameChanged.connect(vv1.video_stream.update_handtrack)

vv1.video_stream.sigHandtrackPointValChanged.connect(vv2.video_stream.update_handtrack_point)
vv2.video_stream.sigHandtrackPointValChanged.connect(vv1.video_stream.update_handtrack_point)


if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()

