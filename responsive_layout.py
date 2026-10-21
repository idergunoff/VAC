"""Compact workspace assembled from Designer controls without changing identities.

Sections are registered in a stack; new tools can be added without growing the
window. Calculation code continues to use the original controls and groups.
"""
from PyQt5 import QtCore, QtGui, QtWidgets


def _column(parent):
    layout = QtWidgets.QVBoxLayout(parent)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(4)
    return layout


def _scroll(content):
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setWidget(content)
    return scroll


def _fold(title, content):
    panel = QtWidgets.QWidget()
    layout = _column(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    toggle = QtWidgets.QToolButton()
    toggle.setText(title)
    toggle.setCheckable(True)
    toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(QtCore.Qt.RightArrow)
    toggle.setToolTip('Развернуть или свернуть блок «' + title + '».')
    layout.addWidget(toggle)
    layout.addWidget(content)
    content.hide()
    def expand(checked):
        content.setVisible(checked)
        toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
    toggle.toggled.connect(expand)
    return panel


def _group(title, widgets):
    group = QtWidgets.QGroupBox(title)
    layout = _column(group)
    for widget in widgets:
        layout.addWidget(widget)
    return _fold(title, group) if title in ('PDF', 'Сравнение и вывод') else group


def _form(title, pairs):
    group = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QFormLayout(group)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(4)
    layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    layout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
    for label, widget in pairs:
        layout.addRow(label, widget)
    return _fold(title, group) if title in ('Обработка', 'Уровни дефектов') else group


def _stack_controls(container):
    """Reflow signal checkboxes vertically, retaining their device ancestry."""
    layout = container.layout()
    if layout is None:
        layout = _column(container)
        for child in container.findChildren(QtWidgets.QWidget, options=QtCore.Qt.FindDirectChildrenOnly):
            layout.addWidget(child)
    for child in container.findChildren(QtWidgets.QWidget, options=QtCore.Qt.FindDirectChildrenOnly):
        inner = child.layout()
        if isinstance(inner, QtWidgets.QGridLayout):
            def drain(layout):
                widgets = []
                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        widgets.append(item.widget())
                    elif item.layout():
                        widgets.extend(drain(item.layout()))
                return widgets
            items = drain(inner)
            for column in range(inner.columnCount()):
                inner.setColumnMinimumWidth(column, 0)
                inner.setColumnStretch(column, 0)
            checks = [w for w in items if isinstance(w, QtWidgets.QCheckBox)]
            checks.sort(key=lambda w: int(w.text().split()[0]))
            for row, widget in enumerate(checks):
                inner.addWidget(widget, row, 0)
                if 'sum' in widget.objectName():
                    button_name = widget.objectName().replace('checkBox', 'pushButton')
                    button = next((w for w in items if w.objectName() == button_name), None)
                    if button:
                        button.setMaximumWidth(30)
                        inner.addWidget(button, row, 1)
            inner.setColumnStretch(0, 1)


class Workspace(QtWidgets.QWidget):
    """Sidebar navigation with independent scroll positions for each section."""
    def __init__(self):
        super().__init__()
        layout = _column(self)
        self.navigation = QtWidgets.QComboBox()
        self.navigation.setToolTip('Выберите раздел работы. Параметры других разделов сохраняются.')
        self.pages = QtWidgets.QStackedWidget()
        layout.addWidget(self.navigation)
        layout.addWidget(self.pages, 1)
        self.navigation.currentIndexChanged.connect(self.pages.setCurrentIndex)
        self.setMinimumWidth(270)

    def add_section(self, title, content):
        self.navigation.addItem(title)
        self.pages.addWidget(_scroll(content))

    def select(self, index):
        self.navigation.setCurrentIndex(index)


def apply_responsive_layout(ui, window):
    central = ui.centralwidget
    # Keep the generated form as a widget catalogue, but own its presentation here.
    originals = central.findChildren(QtWidgets.QWidget)
    font = QtWidgets.QApplication.font()
    font.setPointSize(10)
    for widget in originals:
        widget.setMinimumSize(0, 0)
        widget.setFont(font)
        if isinstance(widget, QtWidgets.QPushButton):
            widget.setText(' '.join(widget.text().split()))
            widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            widget.setMinimumHeight(30)
        elif isinstance(widget, QtWidgets.QLabel):
            widget.setWordWrap(True)
        elif isinstance(widget, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QLineEdit)):
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    ui.Button_direct.setText('Выбрать папку…')
    ui.Button_direct_old.setText('Выбрать папку…')
    ui.Button_open_sig.setText('Открыть сигнал…')
    ui.Button_open_sig_old.setText('Открыть сигнал…')
    ui.pushButton_int.setText('Применить')
    ui.pushButton_int_cement.setText('Преобразовать выбранный сигнал')
    ui.pushButton_cement.setText('Печатный вид')
    ui.pushButton_save_cement.setText('Выгрузить отчёт')
    ui.pushButton_mean.setText('Пересчитать усреднение')
    ui.label_6.setText('Интервал:')
    for i in range(ui.horizontalLayout_2.count()):
        ui.horizontalLayout_2.setStretch(i, 0)
    ui.horizontalLayout_2.addStretch()
    ui.label_9.setText('м')
    ui.pushButton_int.setToolTip('Пересчитать загруженные данные для заданного интервала исследования.')
    ui.pushButton_int_cement.setToolTip('Выберите один сигнал в разделе «Данные», затем преобразуйте его в цементограмму.')

    root = _column(central)
    toolbar = QtWidgets.QHBoxLayout()
    ui.sidebarToggle = QtWidgets.QToolButton()
    ui.sidebarToggle.setText('Панель')
    ui.sidebarToggle.setCheckable(True)
    ui.sidebarToggle.setChecked(True)
    ui.sidebarToggle.setToolTip('Показать или скрыть рабочую панель (Ctrl+\\).')
    toolbar.addWidget(ui.sidebarToggle)
    toolbar.addWidget(ui.horizontalLayoutWidget, 1)
    ui.doubleSpinBox_int_min.setMaximumWidth(110)
    ui.doubleSpinBox_int_max.setMaximumWidth(110)
    toolbar.addWidget(ui.pushButton_save_cement)
    report_button = QtWidgets.QToolButton()
    report_button.setText('⋯')
    report_button.setToolTip('Параметры отчёта и настройки отображения.')
    menu = QtWidgets.QMenu(report_button)
    report_button.setMenu(menu)
    report_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
    toolbar.addWidget(report_button)
    root.addLayout(toolbar)

    ui.workspace = Workspace()
    data = QtWidgets.QWidget()
    data_layout = _column(data)
    ui.equipmentTabs = QtWidgets.QTabWidget()
    for suffix, title, device, header, path in [
        ('', 'Новое', ui.new_device, ui.horizontalLayoutWidget_2, ui.label_direct),
        ('_old', 'Старое', ui.old_device, ui.horizontalLayoutWidget_3, ui.label_direct_old),
    ]:
        page = QtWidgets.QWidget()
        layout = _column(page)
        loading = QtWidgets.QGroupBox(title + ' оборудование')
        loading_layout = _column(loading)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(getattr(ui, 'Button_direct' + suffix))
        buttons.addWidget(getattr(ui, 'Button_open_sig' + suffix))
        loading_layout.addLayout(buttons)
        path.setMaximumHeight(36)
        loading_layout.addWidget(path)
        layout.addWidget(loading)
        layout.addWidget(getattr(ui, 'checkBox_all_new' if not suffix else 'checkBox_all_old'))
        # Header labels describe the old multi-column layout. Signal names carry
        # their own numbers, so keep just the noise selector above the list.
        noise = ui.checkBox_noise if not suffix else ui.checkBox_noise_old
        layout.addWidget(noise)
        header.hide()
        _stack_controls(device)
        layout.addWidget(device)
        layout.addStretch()
        ui.equipmentTabs.addTab(_scroll(page), title)
    data_layout.addWidget(ui.equipmentTabs)
    loading_options = _group('Параметры загрузки', [_form('', [('Усреднение', ui.spinBox_mean)]), ui.pushButton_mean])
    data_layout.addWidget(_fold('Параметры загрузки', loading_options))
    data_layout.addStretch()
    ui.workspace.add_section('Данные', data)

    comparison = QtWidgets.QWidget()
    comparison_layout = _column(comparison)
    processing_tabs = QtWidgets.QTabWidget()
    for suffix, title in [('', 'Новое'), ('_old', 'Старое')]:
        page = QtWidgets.QWidget()
        layout = _column(page)
        names = ['checkBox_envelop', 'checkBox_mean', 'checkBox_func', 'checkBox_norm',
                 'checkBox_rel_ampl', 'checkBox_diff_result']
        if not suffix:
            names.insert(0, 'checkBox_origin_sig')
        layout.addWidget(_group('Отображение сигналов', [getattr(ui, n + suffix) for n in names]))
        layout.addWidget(_form('Обработка', [
            ('Окно усреднения', getattr(ui, 'spinBox_mean_win' + suffix)),
            ('Нормирование', getattr(ui, 'doubleSpinBox_coeff_norm' + suffix)),
            ('Коррекция функции', getattr(ui, 'doubleSpinBox_coeff_func' + suffix)),
            ('Масштаб производной', getattr(ui, 'spinBox_coeff_dif_res' + suffix)),
            ('Коэффициент A', getattr(ui, 'doubleSpinBox_kA' + suffix)),
            ('Коэффициент B', getattr(ui, 'doubleSpinBox_kB' + suffix)),
        ]))
        tag = 'new' if not suffix else 'old'
        layout.addWidget(_group('PDF', [getattr(ui, 'checkBox_pdf' + suffix),
                                       getattr(ui, 'pushButton_pdf' + suffix),
                                       getattr(ui, 'checkBox_diff_res_pdf_' + tag),
                                       getattr(ui, 'pushButton_cem_pdf_' + tag)]))
        layout.addWidget(_form('Уровни дефектов', [
            ('Дефект 1', getattr(ui, 'doubleSpinBox_defect1_' + tag)),
            ('Дефект 2', getattr(ui, 'doubleSpinBox_defect2_' + tag)),
        ]))
        layout.addWidget(_group('Сравнение и вывод', [getattr(ui, 'checkBox_gen_def_' + tag),
            getattr(ui, 'checkBox_match_def_' + tag), getattr(ui, 'pushButton_recalc_qual_' + tag),
            getattr(ui, 'pushButton_all_graphics' + suffix), getattr(ui, 'pushButton_all_cement' + suffix)]))
        layout.addStretch()
        processing_tabs.addTab(_scroll(page), title)
    comparison_layout.addWidget(processing_tabs)
    comparison_layout.addWidget(ui.pushButton_corr_matrix)
    comparison_layout.addWidget(ui.pushButton_choice_cement)
    comparison_layout.addWidget(ui.checkBox_abs_def)
    comparison_layout.addStretch()
    ui.workspace.add_section('Сравнение', comparison)

    cement = QtWidgets.QWidget()
    cement_layout = _column(cement)
    cement_layout.addWidget(ui.pushButton_int_cement)
    cement_layout.addWidget(_form('Масштаб и дефекты', [
        ('Минимум', ui.doubleSpinBox_min_cement), ('Максимум', ui.doubleSpinBox_max_cement),
        ('Дефект 1', ui.doubleSpinBox_defect1), ('Дефект 2', ui.doubleSpinBox_defect2),
    ]))
    cement_layout.addWidget(ui.pushButton_minmax_cem)
    ui.editTools = QtWidgets.QWidget()
    edit_layout = _column(ui.editTools)
    edit_layout.addWidget(_form('Участок коррекции', [('От, м', ui.doubleSpinBox_x1_line),
        ('До, м', ui.doubleSpinBox_x2_line), ('Уровень Y', ui.doubleSpinBox_y_line)]))
    ui.checkBox_lin_cor.setText('Линейно')
    ui.pushButton_fix_corr.setText('Зафиксировать')
    ui.pushButton_plus_func.setText('+ Функция')
    correction = QtWidgets.QGroupBox('Коррекция')
    correction_layout = QtWidgets.QGridLayout(correction)
    for i, widget in enumerate([ui.checkBox_lin_cor, ui.checkBox_shift_corr,
                               ui.pushButton_corr_sig, ui.pushButton_corr_sig_bottom,
                               ui.pushButton_fix_corr, ui.pushButton_plus_func]):
        correction_layout.addWidget(widget, i // 2, i % 2)
    edit_layout.addWidget(correction)
    manual = QtWidgets.QWidget()
    manual_layout = _column(manual)
    manual_layout.addWidget(_form('Интервал вручную', [('От, м', ui.doubleSpinBox_undefine_min),
        ('До, м', ui.doubleSpinBox_undefine_max)]))
    manual_layout.addWidget(ui.groupBox)
    # The radio group already has a horizontal layout wrapper.
    if ui.groupBox.layout() is None:
        _column(ui.groupBox).addWidget(ui.horizontalLayoutWidget_6)
    manual_layout.addWidget(ui.pushButton_add_undefine)
    manual_layout.addWidget(ui.label_int_undefine)
    edit_layout.addWidget(_fold('Интервалы вручную', manual))
    cement_layout.addWidget(ui.editTools)
    ui.editTools.hide()
    cement_layout.addWidget(ui.pushButton_cement)
    cement_layout.addStretch()
    ui.workspace.add_section('Цементограмма', cement)

    plot_panel = QtWidgets.QWidget()
    plot_layout = _column(plot_panel)
    plot_toolbar = QtWidgets.QHBoxLayout()
    ui.name_cement.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    plot_toolbar.addWidget(ui.name_cement, 1)
    ui.editButton = QtWidgets.QToolButton()
    ui.editButton.setText('Редактировать')
    ui.editButton.setCheckable(True)
    ui.editButton.setToolTip('Открыть инструменты редактирования преобразованного сигнала.')
    ui.undoButton = QtWidgets.QToolButton()
    ui.undoButton.setText('Отменить')
    ui.undoButton.setToolTip('Отменить последнюю коррекцию (Ctrl+Z).')
    ui.redoButton = QtWidgets.QToolButton()
    ui.redoButton.setText('Повторить')
    ui.redoButton.setToolTip('Повторить отменённую коррекцию (Ctrl+Shift+Z).')
    for button in [ui.editButton, ui.undoButton, ui.redoButton]:
        plot_toolbar.addWidget(button)
        button.setEnabled(False)
    plot_layout.addLayout(plot_toolbar)
    ui.graphicsView.setMinimumSize(200, 120)
    plot_layout.addWidget(ui.graphicsView, 1)
    splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    splitter.addWidget(ui.workspace)
    splitter.addWidget(plot_panel)
    splitter.setCollapsible(0, True)
    splitter.setCollapsible(1, False)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([320, 1000])
    root.addWidget(splitter, 1)
    ui.responsiveSplitter = splitter
    ui.sidebarToggle.toggled.connect(ui.workspace.setVisible)
    shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+\\'), window)
    shortcut.activated.connect(ui.sidebarToggle.toggle)
    ui.sidebarShortcut = shortcut

    def edit_mode(enabled):
        ui.editTools.setVisible(enabled)
        ui.checkBox_corr_mode.setChecked(enabled)
        if enabled:
            ui.workspace.select(2)
            ui.sidebarToggle.setChecked(True)
    ui.editButton.toggled.connect(edit_mode)
    ui.progressBar.setMaximumHeight(8)
    ui.progressBar.setTextVisible(False)
    ui.progressBar.hide()
    ui.progressBar.valueChanged.connect(lambda value: ui.progressBar.setVisible(0 <= value < ui.progressBar.maximum()))
    root.addWidget(ui.progressBar)
    status = QtWidgets.QHBoxLayout()
    status.addWidget(ui.label_info, 1)
    status.addWidget(ui.horizontalLayoutWidget_5)
    root.addLayout(status)
    ui.label_info.setMaximumHeight(36)

    ui.reportDialog = QtWidgets.QDialog(window)
    ui.reportDialog.setWindowTitle('Параметры отчёта')
    report_layout = _column(ui.reportDialog)
    # Move report fields without copying their values or signal connections.
    report_layout.addWidget(ui.verticalLayoutWidget_2)
    report_layout.addWidget(ui.checkBox_golfstrim)
    report_layout.addWidget(ui.pushButton_from_temp)
    close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
    close.button(QtWidgets.QDialogButtonBox.Close).setText('Закрыть')
    close.rejected.connect(ui.reportDialog.close)
    report_layout.addWidget(close)
    ui.reportDialog.resize(380, 600)
    menu.addAction('Параметры отчёта…', ui.reportDialog.show)
    menu.addAction('Загрузить настройки…', ui.pushButton_open_options.click)
    view_menu = window.menuBar().addMenu('Вид')
    view_action = view_menu.addAction('Рабочая панель')
    view_action.setCheckable(True)
    view_action.setChecked(True)
    view_action.toggled.connect(ui.sidebarToggle.setChecked)
    ui.sidebarToggle.toggled.connect(view_action.setChecked)
    background = QtWidgets.QWidgetAction(view_menu)
    background.setDefaultWidget(ui.checkBox_background_white)
    view_menu.addAction(background)
    view_menu.addAction('Параметры отчёта…', ui.reportDialog.show)

    # Hide emptied Designer wrappers and explanatory labels replaced by blocks.
    for widget in originals:
        if widget.parent() is central and widget not in [ui.progressBar, ui.label_info, ui.horizontalLayoutWidget_5,
                                                                       ui.horizontalLayoutWidget, ui.pushButton_save_cement]:
            widget.hide()
    ui.tabWidget.hide()
    for label in [ui.label_18, ui.label_22]:
        label.hide()
    hints = {
        'doubleSpinBox_int_min': 'Начало интервала исследования, м. Для пересчёта нажмите «Применить».',
        'doubleSpinBox_int_max': 'Конец интервала исследования, м. Для пересчёта нажмите «Применить».',
        'pushButton_mean': 'Заново загрузить данные с выбранным усреднением.',
        'checkBox_background_white': 'Переключить фон рабочего графика на белый.',
        'pushButton_add_undefine': 'Добавить выбранный тип цемента на заданном интервале глубин.',
        'lineEdit_number_skv': 'Номер скважины для заголовка и имени файла отчёта.',
        'comboBox_pipe_type': 'Тип трубы для формулировок в отчёте.',
        'lineEdit_pipe_diameter': 'Диаметр трубы в миллиметрах для отчёта.',
        'lineEdit_client': 'Наименование заказчика в отчёте.',
        'lineEdit_client2': 'Вторая строка сведений о заказчике в отчёте.',
        'lineEdit_interpreter': 'Имя специалиста, выполнившего интерпретацию.',
        'dateEdit': 'Дата исследования, которая попадёт в отчёт.',
        'pushButton_from_temp': 'Заполнить реквизиты значениями из шаблона Word.',
        'pushButton_open_options': 'Восстановить настройки из ранее сохранённого файла .vac.',
    }
    for name, hint in hints.items():
        getattr(ui, name).setToolTip(hint)
    for suffix in ['', '_old']:
        for index in range(1, 4):
            getattr(ui, 'pushButton_sum' + str(index) + suffix).setToolTip(
                'Сложить отмеченные сигналы и записать результат в эту сумму.')
    # Fill gaps in legacy tooltips with visible, local context.
    for parent in [central, ui.reportDialog]:
        for widget in parent.findChildren(QtWidgets.QWidget):
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            if isinstance(widget, (QtWidgets.QAbstractButton, QtWidgets.QAbstractSpinBox,
                                   QtWidgets.QComboBox, QtWidgets.QLineEdit)) and not widget.toolTip():
                text = widget.text() if isinstance(widget, QtWidgets.QAbstractButton) else ''
                if not text:
                    layout = widget.parentWidget().layout()
                    label = layout.labelForField(widget) if isinstance(layout, QtWidgets.QFormLayout) else None
                    text = label.text() if label else widget.accessibleName()
                widget.setToolTip(text or 'Изменить значение параметра.')
                widget.setAccessibleName(text or widget.objectName())
    screen = window.screen() or QtWidgets.QApplication.primaryScreen()
    if screen:
        available = screen.availableGeometry()
        window.resize(min(1440, available.width()), min(900, available.height()))
