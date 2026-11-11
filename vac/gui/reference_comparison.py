"""Reference comparison in the standard plot, using the report's averages."""
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from vac import reference


class ReferenceComparison:
    def __init__(self, controller):
        self.c = controller
        ui = controller.ui
        self.mode = ui.plotMode
        self.device = ui.referenceDevice
        self.selection = ui.referenceSelection
        self.frequency = ui.referenceFrequency
        self.legend = None
        self.update_signals()
        self.mode.currentIndexChanged.connect(self.changed)
        for combo in (self.device, self.selection, self.frequency):
            combo.currentIndexChanged.connect(self.redraw)
        self.changed()

    @property
    def active(self):
        return self.mode.currentIndex() == 1

    def changed(self):
        self.c.ui.referenceControls.setVisible(self.active)
        if self.active:
            self.redraw()
        else:
            self.leave()
            self.c.ui.name_cement.setText('Сигналы')
            self.c.choice_signal()

    def leave(self):
        blocker = QtCore.QSignalBlocker(self.mode)
        self.mode.setCurrentIndex(0)
        del blocker
        self.c.ui.referenceControls.hide()
        if self.legend is not None:
            self.legend.scene().removeItem(self.legend)
            self.c.ui.graphicsView.getPlotItem().legend = None
            self.legend = None
        self.c.ui.graphicsView.setLabel('left', '')
        self.c.ui.graphicsView.setTitle('')

    def open(self, equipment):
        self.device.setCurrentIndex(0 if equipment == 'new' else 1)
        self.mode.setCurrentIndex(1)
        self.redraw()

    def update_signals(self):
        """Keep individual channels and computed sums in sync with the data."""
        equipment = 'new' if self.device.currentIndex() == 0 else 'old'
        table = getattr(self.c.state, equipment).signals
        group = getattr(self.c.ui, equipment + '_device')
        labels = {}
        for check in group.findChildren(QtWidgets.QCheckBox):
            first = check.text().split()[0]
            if first.isdigit():
                labels[int(first)] = check.text()
        channels = []
        for column in table.columns:
            if equipment == 'new' and isinstance(column, str) and column.endswith('_envelope'):
                number = column.removesuffix('_envelope')
                if not number.isdigit():
                    continue
                channel = int(number)
            elif equipment == 'old' and isinstance(column, int):
                channel = column
            else:
                continue
            if table[column].notna().any():
                channels.append(channel)
        items = [('Обе частоты', 'both'), ('5 кГц', 'freq5'), ('2 кГц', 'freq2')]
        items += [(labels.get(n, 'Сигнал ' + str(n)), n) for n in sorted(channels)]
        current = self.frequency.currentData()
        existing = [(self.frequency.itemText(i), self.frequency.itemData(i))
                    for i in range(self.frequency.count())]
        if existing != items:
            with QtCore.QSignalBlocker(self.frequency):
                self.frequency.clear()
                for label, value in items:
                    self.frequency.addItem(label, value)
                index = self.frequency.findData(current)
                self.frequency.setCurrentIndex(max(0, index))

    def redraw(self, *_):
        if not self.active:
            return
        self.c.source_signal.leave()
        self.update_signals()
        c = self.c
        c.curve_editor.detach()
        plot = c.ui.graphicsView
        plot.clear()
        if self.legend is None:
            self.legend = plot.addLegend(offset=(10, 10))
        equipment = 'new' if self.device.currentIndex() == 0 else 'old'
        measurement = getattr(c.state, equipment)
        table = measurement.signals
        plot.showGrid(x=True, y=True)
        plot.setLabel('bottom', 'Глубина', units='м')
        plot.setLabel('left', 'Амплитуда')
        plot.setTitle('Сравнение с эталонами · ' + self.device.currentText())
        c.ui.name_cement.setText('Сравнение с эталонами')
        if not table.empty:
            rows = slice(measurement.interval_min, measurement.interval_max)
            selected = self.frequency.currentData()
            groups = [(1, 2, 5, 6), (3, 4, 7, 8)] if equipment == 'new' else [
                (1, 2, 5, 6, 9, 10), (3, 4, 7, 8, 11, 12)]
            for index, (channels, color, label) in enumerate(zip(
                    groups, ('#54bfff', '#e7b65c'), ('Текущее · 5 кГц', 'Текущее · 2 кГц'))):
                if selected not in ('both', ('freq5', 'freq2')[index]):
                    continue
                columns = [str(n) + '_envelope' if equipment == 'new' else n for n in channels]
                if all(column in table for column in columns):
                    values = table[columns].mean(axis=1, skipna=False)
                    plot.plot(table['depth'].iloc[rows].to_numpy(), values.iloc[rows].to_numpy(),
                              pen=pg.mkPen(color, width=2), name=label)
            if isinstance(selected, int):
                column = str(selected) + '_envelope' if equipment == 'new' else selected
                plot.plot(table['depth'].iloc[rows].to_numpy(), table[column].iloc[rows].to_numpy(),
                          pen=pg.mkPen('#54bfff', width=2), name=self.frequency.currentText())
            c.ui.label_info.setText('Сравнение: ' + self.frequency.currentText() + ' с эталонами скв. 2093.')
        else:
            c.ui.label_info.setText('Загрузите исследование для выбранного оборудования. Показаны эталоны.')
        for index, (kind, color, label) in enumerate([
                ('good', '#58ce87', 'Хорошая · скв. 2093 после ремонта'),
                ('bad', '#fa7272', 'Плохая · скв. 2093 до ремонта')]):
            if self.selection.currentIndex() in (0, index + 1):
                plot.plot(getattr(reference, 'depth_' + equipment),
                          getattr(reference, kind + '_' + equipment),
                          pen=pg.mkPen(color, width=2, style=QtCore.Qt.DashLine), name=label)
        plot.enableAutoRange()
