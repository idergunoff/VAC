"""Small archive menu beside the existing form; file workflows remain available."""
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sqlite3
import pickle
import tempfile

from PyQt5 import QtCore, QtWidgets

from ..domain.state import ApplicationState
from ..services import archive as service
from ..storage import Archive
from ..reporting import figures, word
from ..resources import template_path


@contextmanager
def blocked(window):
    blockers = [QtCore.QSignalBlocker(w) for w in window.findChildren(QtCore.QObject)]
    try:
        yield
    finally:
        del blockers


def capture_widgets(ui):
    result = {}
    for name, widget in vars(ui).items():
        if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            value = {'type': 'number', 'min': widget.minimum(), 'max': widget.maximum(), 'value': widget.value()}
        elif isinstance(widget, QtWidgets.QAbstractButton):
            value = {'type': 'button', 'checked': widget.isChecked(), 'text': widget.text()}
        elif isinstance(widget, QtWidgets.QDateEdit):
            value = {'type': 'date', 'value': widget.date().toString(QtCore.Qt.ISODate)}
        elif isinstance(widget, QtWidgets.QComboBox):
            value = {'type': 'combo', 'value': widget.currentIndex()}
        elif isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QLabel)):
            value = {'type': 'text', 'value': widget.text()}
        else:
            continue
        result[name] = value | {'enabled': widget.isEnabled()}
    return result


def restore_widgets(ui, values):
    for name, data in values.items():
        widget = getattr(ui, name, None)
        if widget is None:
            raise ValueError('Неизвестное поле сохранённой формы: ' + name)
        kind = data['type']
        if kind == 'number':
            widget.setRange(data['min'], data['max']); widget.setValue(data['value'])
        elif kind == 'button':
            widget.setText(data['text']); widget.setChecked(data['checked'])
        elif kind == 'date':
            widget.setDate(QtCore.QDate.fromString(data['value'], QtCore.Qt.ISODate))
        elif kind == 'combo':
            widget.setCurrentIndex(data['value'])
        elif kind == 'text':
            widget.setText(data['value'])
        widget.setEnabled(data['enabled'])


class ArchiveMenu:
    def __init__(self, controller):
        self.c = controller
        self.path = None
        self.study_id = None
        self.run_id = None
        self.last_import = None
        menu = controller.window.menuBar().addMenu('Архив')
        for label, method in [
            ('Открыть / создать архив…', self.choose_archive),
            ('Импортировать LVM…', lambda: self.import_dialog('new')),
            ('Импортировать TWF…', lambda: self.import_dialog('old')),
            ('Открыть исследование…', self.choose_study),
            ('Рассчитать заново текущим алгоритмом', self.recalculate),
            ('Сохранить новую версию', self.save_current),
            ('Восстановить сохранённую версию…', self.choose_version),
            ('Отчёт выбранной интерпретации…', self.report_dialog),
            ('Перенести доверенный .vac…', self.legacy_dialog)]:
            action = menu.addAction(label)
            action.triggered.connect(lambda checked=False, method=method: self.guard(method))

    def guard(self, method):
        try:
            return method()
        except (ValueError, OSError, KeyError, IndexError, AssertionError, sqlite3.Error, pickle.PickleError, EOFError) as exc:
            self.c.ui.label_info.setText(str(exc))
            self.c.ui.label_info.setStyleSheet('color:red')

    def connect(self):
        if not self.path:
            raise ValueError('Сначала выберите файл архива')
        return Archive(self.path)

    def choose_archive(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.c.window, 'Открыть или создать архив', '', 'SQLite (*.sqlite)',
            options=QtWidgets.QFileDialog.DontConfirmOverwrite)
        if path:
            with Archive(path):
                pass
            self.path, self.study_id, self.run_id = path, None, None
            self.status('Архив: ' + path)

    def status(self, text):
        self.c.ui.label_info.setText(text)
        self.c.ui.label_info.setStyleSheet('color:green')

    def select(self, title, rows, label):
        labels = [label(row) + ' [' + row['id'][:8] + ']' for row in rows]
        if not labels:
            raise ValueError('Список пуст: ' + title)
        choice, ok = QtWidgets.QInputDialog.getItem(self.c.window, title, title, labels, 0, False)
        return rows[labels.index(choice)]['id'] if ok else None

    def choose_well(self, db, create=False):
        rows = db.find_wells()
        if create:
            labels = ['Создать новую скважину'] + [r['name'] + ' [' + r['id'][:8] + ']' for r in rows]
            choice, ok = QtWidgets.QInputDialog.getItem(self.c.window, 'Скважина', 'Выберите скважину', labels, 0, False)
            if not ok:
                return None
            if choice == labels[0]:
                name, ok = QtWidgets.QInputDialog.getText(self.c.window, 'Новая скважина', 'Название / номер')
                return db.create_well(name) if ok else None
            return rows[labels.index(choice) - 1]['id']
        return self.select('Скважина', rows, lambda r: r['name'])

    def import_dialog(self, equipment):
        with self.connect() as db:
            well_id = self.choose_well(db, create=True)
            if well_id is None:
                return
            directory = QtWidgets.QFileDialog.getExistingDirectory(self.c.window, 'Комплект измерений')
            if not directory:
                return
            try:
                report = service.import_directory(db, well_id, directory, equipment, self.c.ui.spinBox_mean.value())
            except service.ImportFailure as exc:
                self.last_import = exc.report
                self.show_import_report(exc.report)
                raise
            self.last_import = report
            self.open_study(report.study_id)
            self.show_import_report(report)

    def show_import_report(self, report):
        lines = ['Исследование: ' + str(report.study_id),
                 'Принято: ' + ', '.join(report.accepted),
                 'Пропущено: ' + ', '.join(report.skipped),
                 'Совпадения: ' + repr(report.duplicates),
                 'Назначение каналов: ' + repr(report.assignments),
                 'Ошибки: ' + '; '.join(report.errors),
                 'Повторный импорт создаёт новое исследование; существующее сохраняется.']
        box = QtWidgets.QMessageBox(self.c.window)
        box.setWindowTitle('Отчёт импорта')
        box.setText(f'Принято {len(report.accepted)}, пропущено {len(report.skipped)}, ошибок {len(report.errors)}')
        box.setDetailedText('\n'.join(lines)); box.exec_()

    def choose_study(self):
        with self.connect() as db:
            well = self.choose_well(db)
            if well is None:
                return
            identity = self.select('Исследование', db.list_studies(well), lambda r: r['created_at'] + ' ' + r['equipment'])
        if identity:
            self.open_study(identity)

    def open_study(self, identity):
        with self.connect() as db:
            measurements = service.open_measurements(db, identity)
        state = ApplicationState()
        setattr(state, measurements.equipment, measurements)
        with blocked(self.c.window):
            self.c.state = state
            for equipment in ('new', 'old'):
                group = self.c.ui.new_device if equipment == 'new' else self.c.ui.old_device
                for checkbox in group.findChildren(QtWidgets.QCheckBox):
                    number = int(checkbox.text()[:2])
                    names = {c.number: c.source for c in measurements.channels} if equipment == measurements.equipment else {}
                    checkbox.setChecked(False); checkbox.setEnabled(number in names)
                    checkbox.setText(str(number) + ' ' + names.get(number, ''))
                getattr(self.c.ui, 'checkBox_noise' + ('' if equipment == 'new' else '_old')).setEnabled(equipment == measurements.equipment)
                for number in (1, 2, 3):
                    getattr(self.c.ui, 'pushButton_sum' + str(number) + ('' if equipment == 'new' else '_old')).setEnabled(equipment == measurements.equipment)
            for i in range(9 if measurements.equipment == 'new' else 13, 12 if measurements.equipment == 'new' else 16):
                measurements.signals[str(i) + '_envelope' if measurements.equipment == 'new' else i] = 0
            self.c.set_int(measurements.signals)
        self.c.clear_interpretation()
        self.study_id, self.run_id = identity, None
        self.status('Исследование открыто. Для новой обработки выберите «Рассчитать заново».')

    def recalculate(self):
        if not self.study_id:
            raise ValueError('Сначала откройте исследование архива')
        with self.connect() as db:
            equipment = db.open_study(self.study_id).equipment
        self.c.clear_interpretation()
        (self.c.calc_to_int if equipment == 'new' else self.c.calc_to_int_old)()
        self.run_id = None
        self.status('Выполнен новый расчёт; сохраните его новой версией')

    def save_current(self):
        if not self.study_id:
            raise ValueError('Сначала откройте исследование архива')
        with self.connect() as db:
            provenance = db.open_run(self.run_id) if self.run_id else None
            self.run_id = service.save_version(db, self.study_id, self.c.state, capture_widgets(self.c.ui), provenance)
        self.status('Сохранена новая версия ' + self.run_id[:8])
        return self.run_id

    def choose_version(self):
        if not self.study_id:
            raise ValueError('Сначала выберите исследование')
        with self.connect() as db:
            identity = self.select('Версия обработки', db.list_runs(self.study_id), lambda r: r['created_at'])
        if identity:
            self.open_version(identity)

    def open_version(self, identity):
        with self.connect() as db:
            run = db.open_run(identity)
            state, values, _ = service.open_version(db, identity)
        # Decode completely before touching the open form; signals cannot launch a calculation.
        with blocked(self.c.window):
            restore_widgets(self.c.ui, values)
            self.c.state = state
        self.study_id, self.run_id = run['study_id'], identity
        self.c.ui.graphicsView.clear()
        if state.interpretation.table is None:
            self.c.choice_signal()
        else:
            self.c.draw_cement()
        self.status('Восстановлена версия ' + identity[:8] + ' без пересчёта')

    def save_report(self, path):
        table = self.c.state.interpretation.table
        if table is None:
            raise ValueError('Сначала выберите интерпретацию')
        ui = self.c.ui
        d1, d2 = ui.doubleSpinBox_defect1.value(), ui.doubleSpinBox_defect2.value()
        table['quality'] = self.c.calc_defect(table.corr_sig, d1, d2, self.c.state.interpretation.intervals)
        values = self.c.read_display_values(('checkBox_golfstrim.checkState', 'comboBox_pipe_type.currentText',
            'dateEdit.text', 'doubleSpinBox_int_max.value', 'lineEdit_client.text', 'lineEdit_client2.text',
            'lineEdit_interpreter.text', 'lineEdit_number_skv.text', 'lineEdit_pipe_diameter.text'))
        with tempfile.TemporaryDirectory(prefix='vac-report-') as tmp:
            image = Path(tmp) / 'cement.png'
            figures.cementogramma(table, ui.doubleSpinBox_min_cement.value(), ui.doubleSpinBox_max_cement.value(),
                                  d1, d2, ui.name_cement.text(), image)
            document = word.build_report(table, self.c.state.interpretation.frequency, values, template_path(), image)
            stream = BytesIO(); document.save(stream); content = stream.getvalue()
        # Always snapshot current edits and report fields; never attach changed data to an old version.
        run_id = self.save_current()
        with self.connect() as db:
            _, _, interpretation_id = db.open_gui_version(run_id)
            db.save_report(interpretation_id, content, details=values, template_version='legacy-phase-A',
                           template_hash=sha256(Path(template_path()).read_bytes()).hexdigest())
        Path(path).write_bytes(content)
        self.status('Отчёт сохранён в архиве и в ' + str(path))

    def report_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.c.window, 'Сохранить отчёт', '', 'Word (*.docx)')
        if path:
            self.save_report(path)

    def legacy_dialog(self):
        if not self.study_id:
            raise ValueError('Сначала выберите исследование')
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.c.window, 'Доверенные настройки .vac', '', 'VAC (*.vac)')
        if not path:
            return
        answer = QtWidgets.QMessageBox.question(self.c.window, 'Доверенный источник',
            'Файл .vac использует pickle и может выполнять код. Вы доверяете источнику?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        with self.connect() as db:
            _, limitations = service.import_trusted_vac(db, self.study_id, path)
        self.status('Настройки перенесены как неполное вложение: ' + '; '.join(limitations))
