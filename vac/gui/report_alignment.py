"""Report scale preview. Draft values never modify the edited signal."""
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets


def centered_limits(curve, defect1, defect2):
    values = np.asarray(curve, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or not np.isfinite([defect1, defect2]).all():
        raise ValueError('Для выравнивания нужны конечные значения кривой и уровней дефектов.')
    if defect1 >= defect2:
        raise ValueError('Уровень дефекта 1 должен быть ниже уровня дефекта 2.')
    center = (defect1 + defect2) / 2
    radius = max(abs(values.min() - center), abs(values.max() - center),
                 (defect2 - defect1) / 2, 0.0001) * 1.1
    # Round outwards at the precision used by the main scale controls.
    return np.floor((center - radius) * 10000) / 10000, np.ceil((center + radius) * 10000) / 10000


class ReportAlignmentDialog(QtWidgets.QDialog):
    def __init__(self, controller):
        super().__init__(controller.window)
        self.c = controller
        self.save_requested = False
        self.setWindowTitle('Выравнивание перед отчётом')
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            'Подберите масштаб и уровни дефектов. Автовыравнивание размещает уровни '
            'симметрично относительно центра и вмещает всю кривую с запасом 10%.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.preview = pg.GraphicsLayoutWidget()
        self.preview.setBackground('w')
        self.preview.setMinimumSize(360, 260)
        layout.addWidget(self.preview, 1)
        self.plots = []
        table = controller.state.interpretation.table
        for index in range(4):
            plot = self.preview.addPlot(row=0, col=index, title='КЦ · ЧЦ · ОЦ')
            plot.getViewBox().invertY(True)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.hideAxis('bottom')
            plot.setLabel('left', 'Глубина', units='м')
            plot.showGrid(y=True, alpha=.25)
            plot.plot(table['corr_sig'].to_numpy(), table['depth'].to_numpy(),
                      pen=pg.mkPen('#222222', width=1.5))
            lines = [pg.InfiniteLine(angle=90, pen=pg.mkPen(color, width=1.2, dash=[5, 4]))
                     for color in ('#c08020', '#dd3030')]
            for line in lines:
                plot.addItem(line, ignoreBounds=True)
            plot.setYRange(index * 100, (index + 1) * 100, padding=0)
            self.plots.append((plot, lines))
        form = QtWidgets.QGridLayout()
        layout.addLayout(form)
        self.fields = {}
        for index, (name, label) in enumerate([
                ('min_cement', 'Минимум'), ('max_cement', 'Максимум'),
                ('defect1', 'Дефект 1'), ('defect2', 'Дефект 2')]):
            source = getattr(controller.ui, 'doubleSpinBox_' + name)
            field = QtWidgets.QDoubleSpinBox()
            field.setDecimals(source.decimals())
            field.setRange(-10000, 10000)
            field.setSingleStep(source.singleStep())
            field.setValue(source.value())
            field.setKeyboardTracking(False)
            self.fields[name] = field
            form.addWidget(QtWidgets.QLabel(label), 0, index)
            form.addWidget(field, 1, index)
            field.valueChanged.connect(self.redraw)
        self.auto_button = QtWidgets.QPushButton('Автовыравнивание')
        self.auto_button.clicked.connect(self.auto_align)
        layout.addWidget(self.auto_button)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Cancel)
        self.apply_button = self.buttons.button(QtWidgets.QDialogButtonBox.Apply)
        self.buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText('Отмена')
        self.apply_button.setText('Применить')
        self.apply_button.clicked.connect(self.apply)
        self.save_button = self.buttons.addButton('Применить и сохранить отчёт…', QtWidgets.QDialogButtonBox.ActionRole)
        self.save_button.clicked.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        screen = self.screen().availableGeometry()
        self.resize(min(980, screen.width()), min(640, screen.height()))
        self.redraw()

    def values(self):
        return {name: field.value() for name, field in self.fields.items()}

    def redraw(self, *_):
        values = self.values()
        low, high = values['min_cement'], values['max_cement']
        first, second = values['defect1'], values['defect2']
        valid = low < first < second < high
        self.apply_button.setEnabled(valid)
        self.save_button.setEnabled(valid)
        if not valid:
            self.message.setText('Нужно: минимум < дефект 1 < дефект 2 < максимум.')
            return
        curve = self.c.state.interpretation.table['corr_sig'].to_numpy(dtype=float)
        clipped = np.any(np.isfinite(curve) & ((curve < low) | (curve > high)))
        self.message.setText('Часть кривой выходит за выбранный масштаб.' if clipped else
                             'Масштаб будет применён к изображению в отчёте и сохранён в .vac.')
        for plot, lines in self.plots:
            plot.setXRange(low, high, padding=0)
            lines[0].setValue(first)
            lines[1].setValue(second)

    def auto_align(self):
        try:
            low, high = centered_limits(self.c.state.interpretation.table['corr_sig'],
                self.fields['defect1'].value(), self.fields['defect2'].value())
            if low < -10000 or high > 10000:
                raise ValueError('Выбранный масштаб выходит за диапазон от −10000 до 10000.')
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        with QtCore.QSignalBlocker(self.fields['min_cement']), QtCore.QSignalBlocker(self.fields['max_cement']):
            self.fields['min_cement'].setValue(low)
            self.fields['max_cement'].setValue(high)
        self.redraw()

    def apply(self):
        if not self.apply_button.isEnabled():
            return
        fields = [getattr(self.c.ui, 'doubleSpinBox_' + name) for name in self.fields]
        blockers = [QtCore.QSignalBlocker(field) for field in fields]
        for name, value in self.values().items():
            field = getattr(self.c.ui, 'doubleSpinBox_' + name)
            field.setRange(-10000, 10000)
            field.setValue(value)
        self.c.ui.doubleSpinBox_defect1.setMaximum(self.c.ui.doubleSpinBox_defect2.value())
        self.c.ui.doubleSpinBox_defect2.setMinimum(self.c.ui.doubleSpinBox_defect1.value())
        del blockers
        self.c.capture_cement_options()
        self.c.draw_cement()
        self.c.ui.graphicsView.setYRange(self.fields['min_cement'].value(), self.fields['max_cement'].value(), padding=0)
        self.accept()

    def save(self):
        self.save_requested = True
        self.apply()
