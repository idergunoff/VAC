"""Direct curve manipulation; previews never enter the saved interpretation."""
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ..domain.models import ManualCorrection
from ..processing.corrections import local_shift, local_shift_candidates, fix_corr


class CurveEditor(QtCore.QObject):
    def __init__(self, controller):
        super().__init__(controller.window)
        self.c = controller
        self.ui = controller.ui
        self.plot = self.ui.graphicsView
        self.viewport = self.plot.viewport()
        self.vb = self.plot.getPlotItem().vb
        self.curve = None
        self.region = self.marker = self.original = self.before = None
        self.bounds = None
        self.anchor = None
        self.gesture = None
        self.viewport.installEventFilter(self)
        controller.window.installEventFilter(self)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.stop)
        self.ui.raiseCurveButton.clicked.connect(lambda: self.nudge(1))
        self.ui.lowerCurveButton.clicked.connect(lambda: self.nudge(-1))
        self.ui.showOriginalCurve.toggled.connect(self.update_status)
        self.ui.selectInfluenceButton.toggled.connect(self.update_status)
        self.escape = QtWidgets.QShortcut(QtGui.QKeySequence('Escape'), controller.window)
        self.escape.activated.connect(self.cancel)
        self.update_status()

    @property
    def active(self):
        return (self.curve is not None and self.ui.editButton.isChecked()
                and self.table is not None and not self.table.empty and 'corr_sig' in self.table)

    @property
    def table(self):
        return self.c.state.interpretation.table

    def reset(self):
        self.detach()
        self.bounds = self.anchor = None
        self.ui.selectInfluenceButton.setChecked(False)

    def detach(self):
        self.cancel()
        for item in (self.region, self.marker, self.original, self.before):
            if item is not None:
                self.plot.removeItem(item)
        self.region = self.marker = self.original = self.before = None
        self.curve = None
        self.update_status()

    def attach(self, curve):
        self.curve = curve
        if not self.active or self.table is None or self.table.empty:
            self.curve = None
            self.update_status()
            return
        depth = self.table['depth'].to_numpy(dtype=float)
        finite = depth[np.isfinite(depth)]
        if len(finite) < 2 or finite.min() == finite.max():
            self.curve = None
            self.update_status()
            return
        self.limits = float(finite.min()), float(finite.max())
        low, high = self.limits
        left, right = self.ui.doubleSpinBox_x1_line.value(), self.ui.doubleSpinBox_x2_line.value()
        # Spin boxes round depths for display; retain exact data endpoints on redraw.
        if left == round(low, self.ui.doubleSpinBox_x1_line.decimals()):
            left = low
        if right == round(high, self.ui.doubleSpinBox_x2_line.decimals()):
            right = high
        if self.bounds is None and not low <= left < right <= high:
            left, right = low + (high - low) * .25, low + (high - low) * .75
        self.set_bounds(left, right)
        self.region = pg.LinearRegionItem(self.bounds, movable=False,
            brush=pg.mkBrush(65, 160, 210, 45), pen=pg.mkPen('#54B9E6', width=1.5))
        self.region.setZValue(-5)
        self.plot.addItem(self.region, ignoreBounds=True)
        self.marker = pg.ScatterPlotItem(size=11, brush='#54B9E6', pen=pg.mkPen('#FFFFFF', width=1.5))
        self.marker.setZValue(10)
        self.plot.addItem(self.marker, ignoreBounds=True)
        original = self.c.state.interpretation.automatic_table
        if original is not None and {'depth', 'corr_sig'} <= set(original.columns):
            self.original = pg.PlotDataItem(original['depth'], original['corr_sig'],
                                            pen=pg.mkPen('#D285AE', width=1.5, dash=[5, 4]))
            self.original.setZValue(-1)
            self.plot.addItem(self.original, ignoreBounds=True)
        self.before = pg.PlotDataItem(pen=pg.mkPen('#99AABB', width=1, dash=[2, 3]))
        self.plot.addItem(self.before, ignoreBounds=True)
        self.choose_anchor()
        self.update_status()

    def candidates(self):
        if self.table is None or self.bounds is None:
            return np.array([], dtype=int)
        return local_shift_candidates(self.table, *self.bounds)

    def choose_anchor(self):
        candidates = self.candidates()
        if self.anchor not in candidates:
            self.anchor = (int(candidates[np.argmin(np.abs(
                self.table['depth'].to_numpy()[candidates] - np.mean(self.bounds)))])
                if len(candidates) else None)

    def set_bounds(self, left, right):
        self.bounds = tuple(sorted(float(np.clip(x, *self.limits)) for x in (left, right)))
        if self.region is not None:
            self.region.setRegion(self.bounds)
        # Block legacy redraw callbacks: moving a boundary must not rebuild the plot.
        first, last = self.ui.doubleSpinBox_x1_line, self.ui.doubleSpinBox_x2_line
        blockers = [QtCore.QSignalBlocker(first), QtCore.QSignalBlocker(last)]
        for field in (first, last):
            field.setDecimals(4)
            field.setRange(*self.limits)
        first.setValue(self.bounds[0])
        last.setValue(self.bounds[1])
        first.setMaximum(last.value())
        last.setMinimum(first.value())
        del blockers
        self.choose_anchor()
        self.update_status()

    def update_status(self, *_):
        has_anchor = self.active and self.anchor is not None and self.anchor < len(self.table)
        enabled = has_anchor and self.gesture is None
        self.ui.raiseCurveButton.setEnabled(enabled)
        self.ui.lowerCurveButton.setEnabled(enabled)
        self.ui.selectInfluenceButton.setEnabled(self.active)
        self.escape.setEnabled(self.active)
        if self.original is not None:
            self.original.setVisible(self.ui.showOriginalCurve.isChecked())
        if self.marker is not None:
            if not has_anchor:
                self.marker.setData([], [])
            else:
                table = self.gesture.get('preview', self.table) if self.gesture else self.table
                self.marker.setData([table['depth'].iloc[self.anchor]], [table['corr_sig'].iloc[self.anchor]])
        if not self.active:
            text = 'Откройте цементограмму и включите редактирование.'
        elif self.ui.selectInfluenceButton.isChecked():
            text = 'Протяните мышью по графику, чтобы выделить область влияния.'
        elif not has_anchor:
            text = 'Расширьте область: нужна точка внутри неё или на достигнутом краю графика.'
        else:
            y = self.table['corr_sig'].iloc[self.anchor]
            value = f'{y:.4g}'
            if self.gesture and 'preview' in self.gesture:
                value += f" → {self.gesture['preview']['corr_sig'].iloc[self.anchor]:.4g}"
            text = (f"Точка: {self.table['depth'].iloc[self.anchor]:.4f} м · {value}\n"
                    'Потяните кривую или используйте кнопки. Escape — отмена жеста.')
        self.ui.curveEditHint.setText(text)
        if not self.active:
            self.plot.viewport().unsetCursor()
        elif self.ui.selectInfluenceButton.isChecked():
            self.plot.viewport().setCursor(QtCore.Qt.CrossCursor)

    def point_at(self, position):
        return self.vb.mapSceneToView(self.plot.mapToScene(position))

    def pick(self, point):
        candidates = self.candidates()
        if not len(candidates):
            return None
        px, py = self.vb.viewPixelSize()
        x = self.table['depth'].to_numpy()[candidates]
        y = self.table['corr_sig'].to_numpy()[candidates]
        distance = ((x - point.x()) / px) ** 2 + ((y - point.y()) / py) ** 2
        nearest = int(np.argmin(distance))
        return int(candidates[nearest]) if distance[nearest] <= 14 ** 2 else None

    def boundary_at(self, point):
        distance = np.abs(np.array(self.bounds) - point.x()) / abs(self.vb.viewPixelSize()[0])
        nearest = int(np.argmin(distance))
        return nearest if distance[nearest] <= 7 else None

    def region_position(self, x):
        """Snap pointer selection to a nearby data endpoint in screen pixels."""
        for endpoint in self.limits:
            if abs(x - endpoint) <= 7 * abs(self.vb.viewPixelSize()[0]):
                return endpoint
        return float(np.clip(x, *self.limits))

    def eventFilter(self, obj, event):
        # Python may already have cleared this wrapper during cyclic GC while
        # Qt is still dispatching destruction events from its parent window.
        if not hasattr(self, 'c'):
            return False
        kind = event.type()
        if obj is self.c.window:
            if kind == QtCore.QEvent.Close:
                self.stop()
                return False
            # Hide also arrives during Qt teardown. Touch controls only for a
            # live gesture, while the window and plot still belong to the user.
            if kind in (QtCore.QEvent.WindowDeactivate, QtCore.QEvent.Hide) and self.gesture is not None:
                self.cancel()
            return False
        if kind not in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseMove, QtCore.QEvent.MouseButtonRelease):
            return False
        if obj is not self.viewport or not self.active:
            return False
        point = self.point_at(event.pos())
        if kind == QtCore.QEvent.MouseButtonPress:
            if event.button() != QtCore.Qt.LeftButton or not self.vb.sceneBoundingRect().contains(self.plot.mapToScene(event.pos())):
                return False
            index = None if self.ui.selectInfluenceButton.isChecked() else self.pick(point)
            edge = None if self.ui.selectInfluenceButton.isChecked() else self.boundary_at(point)
            if index is not None:
                self.anchor = index
                self.gesture = {'kind': 'curve', 'start_y': point.y(), 'source': self.table.copy(deep=True),
                                'press_pos': event.pos(), 'moved': False,
                                'auto_range': list(self.vb.state['autoRange'])}
                self.vb.disableAutoRange()
                self.before.setData(self.table['depth'], self.table['corr_sig'])
            else:
                self.gesture = {'kind': 'region', 'bounds': self.bounds, 'anchor': self.anchor,
                                'press_pos': event.pos(), 'moved': False,
                                'edge': edge, 'start_x': self.region_position(point.x())}
            self.update_status()
            return True
        if self.gesture is not None:
            if kind == QtCore.QEvent.MouseMove:
                if not self.gesture['moved']:
                    offset = event.pos() - self.gesture['press_pos']
                    distance = offset.y() if self.gesture['kind'] == 'curve' else offset.x()
                    if abs(distance) < 3:
                        return True
                    self.gesture['moved'] = True
                self.move(point)
                return True
            if kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                if self.gesture['moved']:
                    self.move(point)
                self.finish()
                return True
        if kind == QtCore.QEvent.MouseMove:
            cursor = (QtCore.Qt.CrossCursor if self.ui.selectInfluenceButton.isChecked() else
                      QtCore.Qt.SizeVerCursor if self.pick(point) is not None else
                      QtCore.Qt.SizeHorCursor if self.boundary_at(point) is not None else QtCore.Qt.CrossCursor)
            self.plot.viewport().setCursor(cursor)
        return False

    def stop(self):
        """Disconnect before Qt destroys the viewport or other UI controls."""
        self.cancel()
        self.viewport.removeEventFilter(self)
        self.c.window.removeEventFilter(self)

    def move(self, point):
        gesture = self.gesture
        if gesture['kind'] == 'region':
            edge = gesture['edge']
            if edge is None:
                self.set_bounds(gesture['start_x'], self.region_position(point.x()))
            else:
                bounds = list(gesture['bounds'])
                bounds[edge] = self.region_position(point.x())
                self.set_bounds(*bounds)
            return
        delta = point.y() - gesture['start_y']
        try:
            preview = local_shift(gesture['source'], *self.bounds,
                                  float(gesture['source']['depth'].iloc[self.anchor]), delta)
        except ValueError as exc:
            self.ui.curveEditHint.setText(str(exc))
            return
        gesture['preview'] = preview
        gesture['delta'] = delta
        self.curve.setData(preview['depth'], preview['corr_sig'])
        self.update_status()

    def restore_range(self, gesture):
        if 'auto_range' in gesture:
            for axis, enabled in enumerate(gesture['auto_range']):
                self.vb.enableAutoRange(axis=axis, enable=enabled)

    def cancel(self):
        gesture = self.gesture
        self.gesture = None
        if gesture is not None:
            if gesture['kind'] == 'curve':
                source = gesture['source']
                self.curve.setData(source['depth'], source['corr_sig'])
                self.before.setData([], [])
                self.restore_range(gesture)
            else:
                self.anchor = gesture['anchor']
                self.set_bounds(*gesture['bounds'])
        selecting = self.ui.selectInfluenceButton.isChecked()
        self.ui.selectInfluenceButton.setChecked(False)
        self.update_status()
        return gesture is not None or selecting

    def finish(self):
        gesture = self.gesture
        self.gesture = None
        self.ui.selectInfluenceButton.setChecked(False)
        if gesture['kind'] == 'curve':
            self.before.setData([], [])
            if 'preview' in gesture:
                self.commit(gesture['preview'], gesture['delta'])
            self.restore_range(gesture)
        elif gesture['edge'] is None and gesture['moved']:
            self.anchor = None
            self.choose_anchor()
        self.update_status()

    def nudge(self, direction):
        if not self.active or self.anchor is None or self.gesture is not None:
            return
        delta = direction * self.ui.curveStep.value()
        try:
            preview = local_shift(self.table, *self.bounds, float(self.table['depth'].iloc[self.anchor]), delta)
        except ValueError as exc:
            self.ui.curveEditHint.setText(str(exc))
            return
        self.commit(preview, delta)

    def commit(self, table, delta):
        if table['corr_sig'].equals(self.table['corr_sig']):
            return
        self.c.workspace_actions.checkpoint()
        interpretation = self.c.state.interpretation
        # Make this the baseline for subsequent legacy operations as well.
        interpretation.table = fix_corr(table)
        if interpretation.signal is not None:
            parameters = (('x1', self.bounds[0]), ('x2', self.bounds[1]),
                          ('anchor', float(table['depth'].iloc[self.anchor])), ('delta', float(delta)))
            interpretation.corrections.append(ManualCorrection(interpretation.signal, 'local_shift', parameters))
        self.curve.setData(table['depth'], table['corr_sig'])
        self.c.workspace_actions.refresh()
        self.update_status()
