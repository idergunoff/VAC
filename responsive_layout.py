"""Layout adapters shared by the full and compact Qt Designer forms.

Keep widget identities and signal connections intact; replace absolute positioning
with layouts whose minimum sizes follow fonts and the active Qt style.
"""
from PyQt5 import QtCore, QtWidgets


def _children(widget):
    return widget.findChildren(QtWidgets.QWidget, options=QtCore.Qt.FindDirectChildrenOnly)


def _rows(parent, widgets):
    """Preserve the form's reading order without retaining pixel-sized rows."""
    positions = [(w, w.geometry()) for w in widgets]
    layout = QtWidgets.QVBoxLayout(parent)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(6)
    rows = []
    for widget, rect in sorted(positions, key=lambda item: (item[1].y(), item[1].x())):
        if not rows or rect.y() - rows[-1][0] > 14:
            rows.append((rect.y(), []))
        rows[-1][1].append((widget, rect))
    for _, members in rows:
        row = QtWidgets.QHBoxLayout()
        for widget, _ in sorted(members, key=lambda item: item[1].x()):
            row.addWidget(widget)
        layout.addLayout(row)
    layout.addStretch()
    return layout


def _adapt(widget):
    if isinstance(widget, QtWidgets.QTabWidget):
        for i in range(widget.count()):
            page = widget.widget(i)
            _adapt(page)
        widget.setMinimumWidth(0)
        return
    if widget.layout() is not None:
        return
    children = _children(widget)
    if not children:
        return
    for child in children:
        if type(child) is QtWidgets.QWidget or isinstance(child, QtWidgets.QTabWidget):
            _adapt(child)
    _rows(widget, children)


def _scroll(content):
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    return scroll


def apply_responsive_layout(ui, window):
    central = ui.centralwidget
    children = _children(central)
    # Capture Designer positions before reparenting any controls.
    positions = {w: w.geometry() for w in children}
    for widget in children:
        if widget is not ui.graphicsView:
            _adapt(widget)
    root = QtWidgets.QVBoxLayout(central)
    root.setContentsMargins(8, 8, 8, 8)
    header = QtWidgets.QWidget()
    header_items = [w for w in children if positions[w].y() < positions[ui.graphicsView].y()]
    _rows(header, header_items)
    root.addWidget(_scroll(header))
    root.setStretch(0, 0)
    # The header scrolls horizontally if the available width is very small.
    root.itemAt(0).widget().setMaximumHeight(header.minimumSizeHint().height() + 24)
    splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    splitter.setChildrenCollapsible(False)
    ui.graphicsView.setMinimumSize(0, 120)
    splitter.addWidget(ui.graphicsView)
    lower = QtWidgets.QWidget()
    lower_layout = QtWidgets.QVBoxLayout(lower)
    lower_layout.setContentsMargins(0, 0, 0, 0)
    lower_layout.addWidget(ui.progressBar)
    info = QtWidgets.QHBoxLayout()
    info.addWidget(ui.label_info, 1)
    info.addWidget(ui.horizontalLayoutWidget_5)
    lower_layout.addLayout(info)
    panels = QtWidgets.QTabWidget()
    excluded = set(header_items + [ui.graphicsView, ui.progressBar, ui.label_info,
                                  ui.horizontalLayoutWidget_5, ui.tabWidget])
    remaining = [w for w in children if w not in excluded]
    if hasattr(ui, 'tabWidget_2'):
        panels.addTab(ui.tabWidget_2, 'Оборудование')
    else:
        for left, right, title in [(0, 680, 'Новое оборудование'),
                                   (680, 1410, 'Старое оборудование')]:
            group = QtWidgets.QGroupBox(title)
            _rows(group, [w for w in remaining if left <= positions[w].x() < right])
            panels.addTab(group, title)
    panels.addTab(ui.tabWidget, 'Настройки')
    lower_layout.addWidget(panels)
    splitter.addWidget(_scroll(lower))
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 2)
    root.addWidget(splitter, 1)
    ui.responsiveSplitter = splitter
    # Start within the usable desktop, including on smaller laptop displays.
    screen = window.screen() or QtWidgets.QApplication.primaryScreen()
    if screen:
        available = screen.availableGeometry()
        window.resize(min(1600, available.width()), min(1000, available.height()))
