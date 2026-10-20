# Карта функций и состояния

Сгенерировано `python tools/inventory.py` из AST основного файла. Строки относятся к неизменённому VAC_python.py. Чтение/запись имён — синтаксические; изменение столбцов и вызовы методов включены в «возможные изменения». Это не анализ потока выполнения: aliases, сигналы Qt и переданные аргументы требуют просмотра кода.

## Функции

| Функция:строка | Назначение | Прямые вызовы функций приложения |
| --- | --- | --- |
| `open_dir:56` | функция выбора директории с файлами измерений в формате *.lvm  | calc_to_int, set_int |
| `open_dir_old:125` | функция выбора директории с файлами измерений в формате *.TWF  | calc_to_int_old, set_int |
| `open_sig:191` | функция выбора файлов измерений в формате *.lvm по одному  | calc_to_int, set_int |
| `calc_mean:243` | Пересчет усреднения исходного сигнала для нового оборудования  | calc_to_int |
| `open_sig_old:263` | функция выбора файлов измерений в формате *.TWF по одному  | calc_to_int_old, set_int |
| `calc_to_int:322` | Функция пересчета исходного сигнала, расчет цементограмм  | calc_defect_to_tab, calc_max_int, choice_signal, func1, get_nearest_value, k_damp_detect |
| `calc_to_int_old:407` | Функция пересчета исходного сигнала, расчет цементограмм для старого оборудования | calc_defect_to_tab, calc_max_int, choice_signal, func1, get_nearest_value, k_damp_detect |
| `calc:491` | Запуск пересчета параметров сигнала для существующих измерений  | calc_to_int, calc_to_int_old |
| `calc_defect_to_tab:499` | Сохранение интервалов дефектов и процентов совпадения дефектов в таблице  | calc_defect, choice_signal, create_list_sig, match_defect |
| `recalc_qual_new:550` | Пересчет интервалов дефектов при изменении общих уровней дефектов для нового оборудования    | calc_defect_to_tab |
| `recalc_qual_old:555` | Пересчет интервалов дефектов при изменении общих уровней дефектов для старого оборудования    | calc_defect_to_tab |
| `calc_defect:560` | Функция расчета интервалов дефектов. Принимает сигнал и 2 уровня девектов  |  |
| `match_defect:623` | Расчет процента совпадений дефектов  | create_list_sig, remove_sum |
| `choice_signal:646` | выбор сигналов для построения  | plot_graph, plot_graph_old |
| `plot_graph:762` | выбор и построение кривых для выбранных сигналов | calc_max_int |
| `plot_graph_old:805` | выбор и построение кривых для выбранных сигналов | calc_max_int |
| `change_mean_win:843` | пересчёт кривых при изменении коэффициентов  | choice_signal |
| `change_mean_win_old:868` | пересчёт кривых при изменении коэффициентов  | choice_signal |
| `func1:895` | функция затухания  |  |
| `change_func:901` | пересчет для сдвига функции затухания  | choice_signal, func1 |
| `change_func_old:916` | пересчет для сдвига функции затухания  | choice_signal, func1 |
| `check_int_sig:931` | проверка максимума и минимума интервала сигнала  |  |
| `get_nearest_value:937` | функция поиска ближайщего значения  |  |
| `sum_signals:942` | Расчет суммы сигналов  | calc_defect_to_tab, calc_max_int, choice_signal, func1, get_nearest_value |
| `sum_signals_old:1025` | Расчет суммы сигналов  | calc_defect_to_tab, calc_max_int, choice_signal, func1, get_nearest_value |
| `sum1:1109` | Первый слот для суммы новое оборудование  | k_damp_detect, sum_signals, text_sum |
| `sum2:1126` | Второй слот для суммы новое оборудование  | k_damp_detect, sum_signals, text_sum |
| `sum3:1143` | Третий слот для суммы новое оборудование  | k_damp_detect, sum_signals, text_sum |
| `sum1_old:1160` | Первый слот для суммы старое оборудование  | k_damp_detect, sum_signals_old, text_sum_old |
| `sum2_old:1177` | Второй слот для суммы старое оборудование  | k_damp_detect, sum_signals_old, text_sum_old |
| `sum3_old:1194` | Третий слот для суммы старое оборудование  | k_damp_detect, sum_signals_old, text_sum_old |
| `calc_pdf:1211` | См. тело функции | choice_signal, func1 |
| `calc_pdf_old:1282` | См. тело функции | func1 |
| `text_sum:1343` | Текст для названия суммированного сигнала  |  |
| `text_sum_old:1355` | Текст для названия суммированного сигнала  |  |
| `k_damp_detect:1367` | Цвет названия сигнала, в зависимости от коэффициента затухания  |  |
| `calc_int_cement:1399` | Выбор сигнала для цементограммы  | calc_max_int, draw_cement |
| `cement_from_pdf_new:1482` | Выбор PDF сигнала для цементограммы  | draw_cement |
| `cement_from_pdf_old:1518` | Выбор PDF сигнала для цементограммы  | draw_cement |
| `draw_cement:1553` | отрисовка цементограммы в pyqtgraph (в окне программы обработки)  |  |
| `corr_sig:1581` | Функция вытягивания кривой цементограммы - по максимуму  | draw_cement, get_nearest_value |
| `corr_sig_bottom:1614` | Функция вытягивания кривой цементограммы - по минимуму  | draw_cement, get_nearest_value |
| `fix_corr:1647` | зафиксировать изменения цементограммы для последующей коррекции  | draw_cement |
| `plus_func:1654` | Сложение цементограммы с функцией затухания - поднимает начало сигнала  | draw_cement |
| `add_undefine:1660` | добавить интервал цемента вручную  | get_nearest_value |
| `cementogramma:1684` | построение цеметограммы в Matplotlib  | calc_defect |
| `check_int_corr:1783` | проверка интервала коррекции | draw_cement |
| `check_int_defect:1790` | проверка интервала уровней дефектов | draw_cement |
| `calc_maxmin_cem:1797` | См. тело функции | check_int_defect |
| `check_int_defect_general:1810` | проверка интервала уровней дефектов | choice_signal |
| `check_int_undefine:1819` | проверка интервала неопределенного цемента |  |
| `cement_temp:1825` | См. тело функции |  |
| `save_cement:1835` | Сохранение результирующей цементограммы в отчет в документ Word по шаблону 'template.docx'  |  |
| `draw_total_cement:2078` | функция отрисовки цементограммы по всей длине  |  |
| `draw_match_def:2140` | функция отрисовки кривой совпадения дефектов по всей длине  |  |
| `all_cement:2176` | Отрисовка всех цементограмм на одном листе новое оборудование  | create_list_sig, draw_match_def, draw_total_cement |
| `all_cement_old:2187` | Отрисовка всех цементограмм на одном листе старое оборудование  | create_list_sig, draw_match_def, draw_total_cement |
| `draw_graphics:2198` | Отрисовка всех графиков исходного сигнала на одном листе новое оборудование  |  |
| `draw_graphics_old:2399` | Отрисовка всех графиков исходного сигнала на одном листе старое оборудование  |  |
| `choice_cement:2625` | Отрисовка всех выбранных цеметограмм на одном листе  | draw_total_cement |
| `create_list_sig:2647` | Функция возвращает список существующих сигналов для итераций по ним |  |
| `create_list_sig_check:2666` | Функция возвращает список существующих и выбранных сигналов для итераций по ним |  |
| `background_white:2685` | переключение цвета фона графика |  |
| `open_options:2693` | Загрузка настроек обработки | calc, calc_int_cement, sum1, sum1_old |
| `set_int:2757` | Установка интервала исследование сигнала в спинбоксы + проверка  |  |
| `calc_max_int:2765` | Пересчет интералов отрисовки при загрузке сигналов с разной длиной по одному  | get_nearest_value |
| `check_all_new:2776` | Выбрать все сигналы  |  |
| `check_all_old:2789` | Выбрать все сигналы  |  |
| `remove_sum:2802` | Удалить суммы из списка сигналов  |  |
| `corr_matrix:2821` | строим корреляционную матрицу между сигналами  | create_list_sig_check |
| `mouseMoved:2871` | Отслеживаем координаты курсора  |  |

## Глобальное состояние

| Имя | Создание в модуле | Читают | Записывают имя / возможно меняют объект |
| --- | --- | --- | --- |
| `F` | Возникает позже | save_cement | calc_int_cement |
| `MainWindow` | 35 |  |  |
| `all_signals` | 43 | open_dir, open_sig, calc_mean, calc_to_int, calc_defect_to_tab, recalc_qual_new, match_defect, choice_signal, plot_graph, change_mean_win, change_func, sum_signals, calc_pdf, calc_int_cement, cement_from_pdf_new, draw_total_cement, draw_match_def, all_cement, draw_graphics, choice_cement, remove_sum, corr_matrix | open_dir, open_sig, calc_mean, calc_to_int, choice_signal, plot_graph, change_mean_win, change_func, sum_signals, calc_pdf, calc_int_cement, cement_from_pdf_new, draw_match_def, draw_graphics, corr_matrix |
| `all_signals_old` | 44 | open_dir_old, open_sig_old, calc_to_int_old, calc_defect_to_tab, recalc_qual_old, choice_signal, plot_graph_old, change_mean_win_old, change_func_old, sum_signals_old, calc_pdf_old, calc_int_cement, cement_from_pdf_old, draw_total_cement, draw_match_def, all_cement_old, draw_graphics_old, choice_cement, remove_sum, corr_matrix | open_dir_old, open_sig_old, calc_to_int_old, choice_signal, plot_graph_old, change_mean_win_old, change_func_old, sum_signals_old, calc_pdf_old, calc_int_cement, cement_from_pdf_old, draw_match_def, draw_graphics_old, corr_matrix |
| `all_stat` | 46 | open_dir, open_sig, calc_to_int, plot_graph, change_func, sum_signals, sum1, sum2, sum3, k_damp_detect, calc_int_cement, draw_total_cement | open_dir, open_sig, calc_to_int, sum_signals, sum1, sum2, sum3 |
| `all_stat_old` | 47 | open_dir_old, open_sig_old, calc_to_int_old, plot_graph_old, change_func_old, sum_signals_old, sum1_old, sum2_old, sum3_old, k_damp_detect, calc_int_cement, draw_total_cement | open_dir_old, open_sig_old, calc_to_int_old, sum_signals_old, sum1_old, sum2_old, sum3_old |
| `app` | 34 |  |  |
| `cement_sig` | Возникает позже | calc_int_cement, cement_from_pdf_new, cement_from_pdf_old, draw_cement, corr_sig, corr_sig_bottom, fix_corr, plus_func, add_undefine, cementogramma, save_cement, open_options | calc_int_cement, cement_from_pdf_new, cement_from_pdf_old, corr_sig, corr_sig_bottom, fix_corr, plus_func, add_undefine, cementogramma, save_cement, open_options |
| `columns` | 45 |  |  |
| `int_0_cement` | 49 | calc_defect, add_undefine | calc_int_cement, cement_from_pdf_new, cement_from_pdf_old, add_undefine |
| `int_1_cement` | 50 | calc_defect, add_undefine | calc_int_cement, cement_from_pdf_new, cement_from_pdf_old, add_undefine |
| `int_2_cement` | 51 | calc_defect, add_undefine | calc_int_cement, cement_from_pdf_new, cement_from_pdf_old, add_undefine |
| `int_max` | Возникает позже | calc_to_int, calc_defect_to_tab, choice_signal, plot_graph, change_mean_win, sum_signals, calc_pdf, calc_int_cement, cement_from_pdf_new, draw_total_cement, draw_match_def, draw_graphics, corr_matrix | calc_to_int, plot_graph, sum_signals, calc_int_cement |
| `int_max_old` | Возникает позже | calc_to_int_old, calc_defect_to_tab, choice_signal, plot_graph_old, change_mean_win_old, sum_signals_old, calc_pdf_old, calc_int_cement, cement_from_pdf_old, draw_total_cement, draw_match_def, draw_graphics_old, corr_matrix | calc_to_int_old, plot_graph_old, sum_signals_old |
| `int_min` | Возникает позже | calc_to_int, calc_defect_to_tab, choice_signal, plot_graph, sum_signals, calc_pdf, calc_int_cement, cement_from_pdf_new, draw_total_cement, draw_match_def, draw_graphics | calc_to_int |
| `int_min_old` | Возникает позже | calc_to_int_old, calc_defect_to_tab, choice_signal, plot_graph_old, sum_signals_old, calc_pdf_old, calc_int_cement, cement_from_pdf_old, draw_total_cement, draw_match_def, draw_graphics_old | calc_to_int_old |
| `int_undefine_cement` | 48 | calc_defect, add_undefine | calc_int_cement, cement_from_pdf_new, cement_from_pdf_old, add_undefine |
| `k_damp` | Возникает позже |  | calc_int_cement |
| `k_oldtonew` | 53 | open_dir_old, open_sig_old | open_dir_old, open_sig_old |
| `options` | Возникает позже | cementogramma, save_cement | cementogramma |
| `proxy` | 2882 |  |  |
| `ui` | 36 | open_dir, open_dir_old, open_sig, calc_mean, open_sig_old, calc_to_int, calc_to_int_old, calc, calc_defect_to_tab, calc_defect, choice_signal, plot_graph, plot_graph_old, change_mean_win, change_mean_win_old, change_func, change_func_old, check_int_sig, sum_signals, sum_signals_old, sum1, sum2, sum3, sum1_old, sum2_old, sum3_old, calc_pdf, calc_pdf_old, text_sum, text_sum_old, calc_int_cement, cement_from_pdf_new, cement_from_pdf_old, draw_cement, corr_sig, corr_sig_bottom, add_undefine, cementogramma, check_int_corr, check_int_defect, calc_maxmin_cem, check_int_defect_general, check_int_undefine, cement_temp, save_cement, draw_total_cement, draw_match_def, draw_graphics, draw_graphics_old, choice_cement, create_list_sig, create_list_sig_check, background_white, open_options, set_int, calc_max_int, check_all_new, check_all_old, corr_matrix, mouseMoved | open_dir, open_dir_old, open_sig, calc_mean, open_sig_old, calc_to_int, calc_to_int_old, calc, calc_defect_to_tab, calc_defect, choice_signal, plot_graph, plot_graph_old, change_mean_win, change_mean_win_old, change_func, change_func_old, check_int_sig, sum_signals, sum_signals_old, sum1, sum2, sum3, sum1_old, sum2_old, sum3_old, calc_pdf, calc_pdf_old, text_sum, text_sum_old, calc_int_cement, cement_from_pdf_new, cement_from_pdf_old, draw_cement, corr_sig, corr_sig_bottom, add_undefine, cementogramma, check_int_corr, check_int_defect, calc_maxmin_cem, check_int_defect_general, check_int_undefine, cement_temp, save_cement, draw_total_cement, draw_match_def, draw_graphics, draw_graphics_old, choice_cement, create_list_sig, create_list_sig_check, background_white, open_options, set_int, calc_max_int, check_all_new, check_all_old, corr_matrix, mouseMoved |
