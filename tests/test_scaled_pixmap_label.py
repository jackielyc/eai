import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest

from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QApplication
from show_camera_topics import ScaledPixmapLabel


class ScaledPixmapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_stable_frame_keeps_complete_image_and_no_resize_feedback(self):
        label = ScaledPixmapLabel(fill_frame=True)
        label.resize(400, 240)
        source = QPixmap(320, 200)
        source.fill(QColor('red'))
        before = label.sizeHint()
        label.set_source_pixmap(source, source_wh=(640, 400))
        displayed = label.pixmap()
        self.assertEqual(label.sizeHint(), before)
        self.assertLessEqual(displayed.width(), label.width())
        self.assertLessEqual(displayed.height(), label.height())
        self.assertAlmostEqual(displayed.width()/displayed.height(), 320/200, places=2)
        label.resize(500, 240)
        self.app.processEvents()
        label._refresh_scaled()
        displayed = label.pixmap()
        self.assertLessEqual(displayed.width(), 500)
        self.assertLessEqual(displayed.height(), 240)
        self.assertAlmostEqual(displayed.width()/displayed.height(), 320/200, places=2)
        label.deleteLater()

    def test_normal_mode_keeps_entire_image(self):
        label = ScaledPixmapLabel(fill_frame=False)
        label.resize(400, 240)
        source = QPixmap(320, 200)
        label.set_source_pixmap(source)
        displayed = label.pixmap()
        self.assertLessEqual(displayed.width(), label.width())
        self.assertLessEqual(displayed.height(), label.height())
        label.deleteLater()


if __name__ == '__main__':
    unittest.main()
