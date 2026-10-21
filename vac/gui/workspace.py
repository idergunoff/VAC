"""Workspace state and bounded undo history for manual curve corrections."""
from copy import deepcopy

from PyQt5 import QtCore, QtGui, QtWidgets


class WorkspaceActions:
    def __init__(self, controller):
        self.c = controller
        self.undo_stack = []
        self.redo_stack = []
        self.interpretation = controller.state.interpretation
        ui = controller.ui
        ui.undoButton.clicked.connect(self.undo)
        ui.redoButton.clicked.connect(self.redo)
        self.shortcuts = []
        for key, slot in [('Ctrl+Z', self.undo), ('Ctrl+Shift+Z', self.redo)]:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), controller.window)
            shortcut.activated.connect(slot)
            self.shortcuts.append(shortcut)
        self.refresh()

    def reset(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.interpretation = self.c.state.interpretation
        self.refresh()

    def snapshot(self):
        return deepcopy(self.c.state.interpretation), self.c.ui.label_int_undefine.text()

    def checkpoint(self):
        if self.c.state.interpretation.table is None:
            return
        self.undo_stack.append(self.snapshot())
        del self.undo_stack[:-30]
        self.redo_stack.clear()
        self.refresh()

    def restore(self, snapshot):
        state, label = snapshot
        self.c.state.interpretation = state
        self.interpretation = state
        self.c.ui.label_int_undefine.setText(label)
        self.c.draw_cement()
        self.refresh()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.snapshot())
            self.restore(self.undo_stack.pop())

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.snapshot())
            self.restore(self.redo_stack.pop())

    def refresh_tooltips(self):
        ui = self.c.ui
        for group in [ui.new_device, ui.old_device]:
            for widget in group.findChildren(QtWidgets.QCheckBox):
                widget.setToolTip(widget.text() + '\nПоказать на графике. Для преобразования выберите один сигнал.')
        for widget in [ui.label_direct, ui.label_direct_old, ui.checkBox_noise, ui.checkBox_noise_old]:
            widget.setToolTip(widget.text())

    def refresh(self, show_cement=False):
        ui = self.c.ui
        self.refresh_tooltips()
        if self.interpretation is not self.c.state.interpretation:
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.interpretation = self.c.state.interpretation
        ready = self.c.state.interpretation.table is not None
        ui.editButton.setEnabled(ready)
        ui.pushButton_save_cement.setEnabled(ready)
        ui.pushButton_save_cement.setToolTip('Выгрузить текущую цементограмму в Word.' if ready else
                                           'Сначала преобразуйте выбранный сигнал в цементограмму.')
        ui.undoButton.setEnabled(ready and bool(self.undo_stack))
        ui.redoButton.setEnabled(ready and bool(self.redo_stack))
        ui.undoButton.setToolTip('Отменить последнюю коррекцию (Ctrl+Z).')
        ui.redoButton.setToolTip('Повторить отменённую коррекцию (Ctrl+Shift+Z).')
        for button, text in [(ui.undoButton, 'Нет коррекций для отмены.'),
                             (ui.redoButton, 'Нет отменённых коррекций для повтора.')]:
            if not button.isEnabled():
                button.setToolTip(text)
        ui.workspace.setVisible(ui.sidebarToggle.isChecked())
        ui.editTools.setVisible(ready and ui.editButton.isChecked())
        if not ready:
            ui.editButton.setChecked(False)
            ui.name_cement.setText('Сигналы')
        elif show_cement:
            ui.workspace.select(2)
