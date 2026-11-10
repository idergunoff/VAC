"""Persistent report text editor, available before measurements are loaded."""
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from ..reporting.template import DEFAULT_LABELS, ELEMENTS, TemplateStore, default_texts, validate_texts


def user_template_store():
    directory = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.GenericConfigLocation)
    return TemplateStore(Path(directory) / 'VAC' / 'report-template.json')


class TemplateEditor(QtWidgets.QDialog):
    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.defaults = default_texts()
        self.texts = self.defaults | store.load()
        self.saved = self.texts.copy()
        self.current_key = None
        self.setWindowTitle('Редактор шаблона отчёта')
        self.resize(900, 580)
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            'Выберите элемент и измените его текст. Сохранённый шаблон используется для всех новых отчётов.\n'
            'Реквизиты по умолчанию переносятся в параметры отчёта кнопкой «Из шаблона».')
        info.setWordWrap(True)
        layout.addWidget(info)
        splitter = QtWidgets.QSplitter()
        self.elements = QtWidgets.QListWidget()
        splitter.addWidget(self.elements)
        pane = QtWidgets.QWidget()
        pane_layout = QtWidgets.QVBoxLayout(pane)
        self.title = QtWidgets.QLabel()
        self.title.setWordWrap(True)
        pane_layout.addWidget(self.title)
        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setAccessibleName('Текст элемента шаблона')
        pane_layout.addWidget(self.editor)
        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True)
        self.hint.setTextFormat(QtCore.Qt.PlainText)
        pane_layout.addWidget(self.hint)
        reset = QtWidgets.QPushButton('Вернуть исходный текст элемента')
        reset.clicked.connect(self.reset_element)
        pane_layout.addWidget(reset)
        splitter.addWidget(pane)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 570])
        layout.addWidget(splitter, 1)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.setTextFormat(QtCore.Qt.PlainText)
        self.error.setStyleSheet('color: #b00020')
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox()
        self.save_button = buttons.addButton('Сохранить', QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.addButton('Отмена', QtWidgets.QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.elements.currentItemChanged.connect(self.select_element)
        self.editor.textChanged.connect(self.edit_text)
        for key, label in [(e.key, e.label) for e in ELEMENTS] + list(DEFAULT_LABELS.items()):
            item = QtWidgets.QListWidgetItem(label)
            item.setToolTip(label)
            item.setData(QtCore.Qt.UserRole, key)
            self.elements.addItem(item)
        self.elements.setCurrentRow(0)
        self.update_dirty()

    def select_element(self, item, previous=None):
        if item is None:
            return
        self.current_key = item.data(QtCore.Qt.UserRole)
        self.title.setText(item.text())
        blocker = QtCore.QSignalBlocker(self.editor)
        self.editor.setPlainText(self.texts[self.current_key])
        del blocker
        element = next((e for e in ELEMENTS if e.key == self.current_key), None)
        tokens = ', '.join('{' + name + '}' for _, name in element.tokens) if element else ''
        self.hint.setText('Обязательные подстановки (по одному разу): ' + tokens if tokens else
                          'Обычный текст. Можно оставить пустым.' if element is None else
                          'Обычный текст. Можно оставить пустым. Для фигурных скобок используйте {{ и }}.')

    def edit_text(self):
        if self.current_key:
            self.texts[self.current_key] = self.editor.toPlainText()
            self.error.clear()
            self.update_dirty()

    def update_dirty(self):
        dirty = self.texts != self.saved
        self.setWindowTitle('Редактор шаблона отчёта' + (' — есть изменения' if dirty else ''))
        self.save_button.setEnabled(dirty)

    def reset_element(self):
        if self.current_key:
            self.editor.setPlainText(self.defaults[self.current_key])

    def save(self):
        try:
            validate_texts(self.texts)
            self.store.save(self.texts)
        except (ValueError, OSError) as exc:
            self.error.setText('Шаблон не сохранён: ' + str(exc))
            return
        self.saved = self.texts.copy()
        self.accept()

    def reject(self):
        if self.texts != self.saved:
            result = QtWidgets.QMessageBox.question(
                self, 'Несохранённые изменения', 'Закрыть редактор без сохранения изменений?',
                QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
            if result != QtWidgets.QMessageBox.Discard:
                return
        super().reject()


class TemplateMenu:
    def __init__(self, controller):
        self.c = controller
        self.store = user_template_store()
        menu = controller.window.menuBar().addMenu('Шаблон отчёта')
        self.action = menu.addAction('Редактировать тексты…')
        self.action.triggered.connect(self.open_editor)

    def open_editor(self):
        try:
            dialog = TemplateEditor(self.store, self.c.window)
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self.c.window, 'Шаблон отчёта', str(exc))
            return
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.c.ui.label_info.setText('Шаблон сохранён. Изменения будут использованы в новых отчётах.')
            self.c.ui.label_info.setStyleSheet('color:green')
