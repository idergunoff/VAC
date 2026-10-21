"""Exercise compact workspace geometry and access to every equipment page."""
import os
import sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PyQt5 import QtWidgets, QtCore
from VAC_dialog import Ui_MainWindow

app = QtWidgets.QApplication([])
window = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(window)
window.show()
for width, height in [(800, 600), (1366, 768), (1920, 1080), (3200, 2000)]:
    window.resize(width, height)
    app.processEvents()
    assert window.size().width() == width and window.size().height() == height
    assert ui.responsiveSplitter.orientation() == QtCore.Qt.Horizontal
    assert ui.graphicsView.width() > width * .45
    assert ui.graphicsView.height() > height * .55
    assert ui.horizontalLayoutWidget.isVisible()
    assert ui.pushButton_int.isVisible()
    assert ui.pushButton_save_cement.isVisible()
    for section in range(ui.workspace.pages.count()):
        ui.workspace.select(section)
        app.processEvents()
        scroll = ui.workspace.pages.currentWidget()
        assert scroll.isVisible()
        assert scroll.horizontalScrollBar().maximum() == 0, (width, section)
    ui.workspace.select(0)
    for index in range(ui.equipmentTabs.count()):
        ui.equipmentTabs.setCurrentIndex(index)
        app.processEvents()
        assert ui.equipmentTabs.currentWidget().isVisible()
    plot_width = ui.graphicsView.width()
    ui.sidebarToggle.setChecked(False)
    app.processEvents()
    assert ui.graphicsView.width() > plot_width
    ui.sidebarToggle.setChecked(True)
    app.processEvents()
    print(width, height, 'plot', ui.graphicsView.width(), ui.graphicsView.height(), flush=True)
window.close()
