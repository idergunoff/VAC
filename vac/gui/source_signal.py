"""Read-only source view; the editor and its plot remain intact in the stack."""
from PyQt5 import QtCore
import pyqtgraph as pg


class SourceSignalView:
    def __init__(self, controller):
        self.c = controller
        self.ui = controller.ui
        self.view = pg.GraphicsLayoutWidget()
        self.amplitude = self.view.addPlot(row=0, col=0)
        self.amplitude.addLegend()
        self.amplitude.setLabel('left', 'Амплитуда')
        self.amplitude.setLabel('bottom', 'Глубина', units='м')
        self.source = self.view.addPlot(row=1, col=0)
        self.source.setLabel('left', 'Значение')
        self.source.setLabel('bottom', 'Глубина', units='м')
        self.amplitude.setXLink(self.source)
        for plot in (self.amplitude, self.source):
            plot.showGrid(x=True, y=True)
        self.ui.plotStack.addWidget(self.view)
        self.ui.sourceSignalButton.toggled.connect(self.toggle)

    def leave(self):
        with QtCore.QSignalBlocker(self.ui.sourceSignalButton):
            self.ui.sourceSignalButton.setChecked(False)
        self.ui.sourceSignalButton.setText('Исходный сигнал')
        self.ui.plotStack.setCurrentWidget(self.ui.graphicsView)
        self.ui.editTools.setEnabled(True)
        self.c.curve_editor.update_status()

    def toggle(self, enabled):
        if not enabled:
            self.leave()
            return
        original = self.c.state.interpretation.automatic_table
        if original is None or original.empty or not {'depth', 'first_sig'} <= set(original.columns):
            self.leave()
            return
        if self.c.curve_editor.gesture is not None:
            self.c.curve_editor.cancel()
        self.amplitude.clear()
        self.source.clear()
        self.source.setTitle('Сигнал цементограммы до ручных правок')
        self.source.plot(original['depth'].to_numpy(), original['first_sig'].to_numpy(),
                         pen=pg.mkPen('#D285AE', width=2))
        identity = self.c.state.interpretation.signal
        plotted = False
        if identity is not None:
            measurements = getattr(self.c.state, identity.equipment)
            signals = measurements.signals
            token = next(iter(identity.signal.split()), '')
            number = int(token) if token.isdigit() else None
            columns = []
            if identity.signal == 'pdf':
                columns = [('pdf_envelope', 'PDF', '#54B9E6'), ('pdf_func', 'Функция затухания', '#58CE87')]
            elif number is not None:
                columns = [(number, 'Исходный сигнал', '#99AABB')]
                if identity.equipment == 'new':
                    columns.append((str(number) + '_envelope', 'Огибающая', '#54B9E6'))
                columns.append((str(number) + '_func', 'Функция затухания', '#58CE87'))
            if measurements.revision == identity.revision and 'depth' in signals:
                rows = signals['depth'].between(original['depth'].min(), original['depth'].max())
                for column, label, color in columns:
                    if column in signals and signals.loc[rows, column].notna().any():
                        self.amplitude.plot(signals.loc[rows, 'depth'].to_numpy(),
                                            signals.loc[rows, column].to_numpy(),
                                            pen=pg.mkPen(color, width=1.5), name=label)
                        plotted = True
        title = identity.signal if identity is not None else self.ui.name_cement.text()
        self.view.setToolTip(title)
        self.amplitude.setTitle('Исходный сигнал' if plotted else 'Исходные измерения недоступны')
        self.amplitude.enableAutoRange()
        self.source.enableAutoRange()
        self.ui.plotStack.setCurrentWidget(self.view)
        self.ui.sourceSignalButton.setText('К цементограмме')
        self.ui.editTools.setEnabled(False)
        self.c.curve_editor.update_status()
