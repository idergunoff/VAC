"""Explicit GUI assembly; imports do not create a QApplication."""
import sys


def create_window(state=None):
    from PyQt5 import QtWidgets
    from VAC_dialog import Ui_MainWindow
    from .controller import Controller
    window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    controller = Controller(window, ui, state)
    window.controller = controller  # keep signal proxy and dependencies alive
    return window, controller


def main(argv=None):
    import logging
    import os
    if os.environ.get('VAC_DIAGNOSTICS') == '1':
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    from PyQt5 import QtCore, QtWidgets
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv if argv is None else argv)
    window, controller = create_window()
    window.show()
    window.showMaximized()
    return app.exec_()
