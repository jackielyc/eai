import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QScrollArea, QTabWidget, QVBoxLayout, QWidget
from show_camera_topics import CameraTopicWindow, _tab_page_matches, _wrap_in_vscroll


class TabScrollTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def test_all_plain_and_nested_pages_get_one_scroll_layer(self):
        window=QMainWindow();root=QWidget();window.setCentralWidget(root);layout=QVBoxLayout(root)
        outer=QTabWidget();window.control_tabs=outer;layout.addWidget(outer)
        page=QWidget();page_layout=QVBoxLayout(page);large=QLabel('large');large.setMinimumHeight(1200)
        page_layout.addWidget(large)
        nested=QTabWidget();nested_page=QWidget();nested_layout=QVBoxLayout(nested_page)
        nested_layout.addWidget(QLabel('nested'));nested.addTab(nested_page,'nested')
        page_layout.addWidget(nested);outer.addTab(page,'plain')
        existing_content=QWidget();existing_content.setLayout(QVBoxLayout())
        existing=_wrap_in_vscroll(existing_content);outer.addTab(existing,'existing')
        outer.setCurrentIndex(1)

        count=CameraTopicWindow._install_scroll_on_all_tabs(window)
        self.assertEqual(count,2)
        self.assertEqual(outer.currentIndex(),1)
        outer_wrapper=outer.widget(0)
        self.assertIsInstance(outer_wrapper,QScrollArea)
        self.assertTrue(_tab_page_matches(outer_wrapper,page))
        self.assertIs(outer.widget(1),existing)
        self.assertIsInstance(nested.widget(0),QScrollArea)
        self.assertTrue(_tab_page_matches(nested.widget(0),nested_page))
        self.assertIs(large.parent(),page)
        self.assertEqual(outer_wrapper.verticalScrollBarPolicy(),Qt.ScrollBarAsNeeded)
        self.assertEqual(outer_wrapper.horizontalScrollBarPolicy(),Qt.ScrollBarAsNeeded)
        self.assertEqual(CameraTopicWindow._install_scroll_on_all_tabs(window),0)
        window.deleteLater()


if __name__=='__main__':unittest.main()
