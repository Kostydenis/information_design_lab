report_write('''
Обратим внимание, что в целевой функции коэффициент при переменной равен 1000,
что значительно больше любого из остальных коэффициентов целевой функции.
Тем самым использование судов третьего типа на второй регулярной линии "заблокировано",
так как при $x_{32} \\neq 0$ значение целевой функции резко возрастает, и алгоритм,
сформированный на основе метода симплекс-таблиц, выведет переменную $x_{32}$ из числа базисных переменных,
т.~е. определит значение  равным нулю.

В системе ограничений вида неравенств коэффициент при  равен нулю. Тем самым, отражено то,
что на судах 3-го типа по 2-ой регулярной линии количество перевозимого груза может быть только равным нулю.
''')