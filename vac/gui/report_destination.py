"""Shared report destination and persistent last successful output directory."""
from pathlib import Path
from PyQt5 import QtCore, QtWidgets


def report_options_path(path, well):
    name = str(well).replace('/', '_').replace('\\', '_')
    return Path(path).with_name('options_' + name + '.vac')


def report_settings():
    root = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.GenericConfigLocation)
    return QtCore.QSettings(str(Path(root) / 'VAC' / 'report-output.ini'), QtCore.QSettings.IniFormat)


def choose_report_path(parent, well):
    settings = report_settings()
    directory = Path(settings.value('directory', '', type=str))
    if not directory.is_dir():
        directory = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation))
    name = str(well).replace('/', '_').replace('\\', '_')
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent, 'Сохранить отчёт', str(directory / ('цементограмма_по_скв_' + name + '.docx')), 'Word (*.docx)')
    if not path:
        return None
    result = Path(path)
    if result.suffix.lower() != '.docx':
        result = result.with_name(result.name + '.docx')
        if result.exists():
            answer = QtWidgets.QMessageBox.question(parent, 'Заменить файл?',
                'Файл уже существует. Заменить его?\n' + str(result),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return None
    return result


def remember_report_path(path):
    settings = report_settings()
    settings.setValue('directory', str(Path(path).resolve().parent))
    settings.sync()
    return settings.status() == QtCore.QSettings.NoError
