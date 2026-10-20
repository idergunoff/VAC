import os,sys,importlib
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PyQt5 import QtWidgets,QtCore
app=QtWidgets.QApplication([])
for module in ['VAC_dialog','VAC_dialog_mini']:
 w=QtWidgets.QMainWindow();ui=importlib.import_module(module).Ui_MainWindow();ui.setupUi(w);w.show()
 for width,height in [(800,600),(1366,768),(1920,1080),(3200,2000)]:
  w.resize(width,height);app.processEvents()
  assert w.width()==width and w.height()==height,(w.size(),width,height)
  assert ui.graphicsView.width()>width-40,(ui.graphicsView.size(),width)
  for scroll in w.findChildren(QtWidgets.QScrollArea):
   assert scroll.widget().width()>=scroll.widget().minimumSizeHint().width()
  print(module,width,height,'plot',ui.graphicsView.width(),ui.graphicsView.height(),flush=True)
  for tabs in w.findChildren(QtWidgets.QTabWidget):
   for index in range(tabs.count()):
    tabs.setCurrentIndex(index);app.processEvents()
    page = tabs.currentWidget()
    assert isinstance(page, QtWidgets.QTabWidget) or page.layout() is not None or not page.findChildren(QtWidgets.QWidget)
  assert ui.responsiveSplitter.count() == 2
 w.close()
