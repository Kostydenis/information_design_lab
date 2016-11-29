#######################################################
from IPython.display import display, Latex, Math, Markdown
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import math
import copy
from ipy_table import make_table, set_row_style
import re
import random
#######################################################

ROUND_ACCURACY = 4
MAX_ITERATIONS = 11
REPORT_PATH = 'report/solution.tex'

#######################################################

IS_OUTPUT = True

report_output = open(REPORT_PATH, 'w')
def report_write(s,debug=False):
	if IS_OUTPUT or debug:
		report_output.write(s.replace('&nbsp', ' ')+'\n\n')

def writetbl(t, caption='', debug=False):
	if IS_OUTPUT or debug:
		output =  '\\begin{table}[H]\n'
		output += '\t\\centering\n'
		output += '\t\\normalsize\n'
		output += '\t\\caption{' + str(caption) + '}\n'
		output += '\t\\label{tbl:' + str(printed_table_no) + '}\n'
		# printed_table_no += 1
		output += '\t\\begin{tabular}{|' + 'c|'*len(t[0]) + '}\n'
		output += '\t\t\\hline\n'

		for row in t:
			output += '\t\t'
			for item in row:
				output += str(item) + '&'
			output = output[:-1] + '\\\\ \\hline\n'

		output += '\n\\end{tabular}\n'
		output += '\n\\end{table}'
		report_write(output,debug)

def writeeq(eq,num=False,debug=False):
	if IS_OUTPUT or debug:
		output = '\\vspace{-\\baselineskip}'
		output += '\\begin{align}\n' if num else '\\begin{align*}\n'

		output += eq

		output += '\n\\end{align}' if num else '\n\\end{align*}'
		report_write(output,debug)
#######################################################

printed_table_no = 1

#######################################################

tons = [
	[25,20,50],
	[15,10,0],
	[10,40,8],
]
cost = [
	[15,10,30],
	[25, 6, 0],
	[30, 5,10],
]
min_transit = [
	500,
	200,
	100
]
vehicles = [55,95,30]

# test set source data
# tons = [
#   [25,20,50,50],
#   [20,12, 0,45],
#   [15,10, 0,40],
#   [10,40, 8,25],
# ]
# cost = [
#   [15,10,30,25],
#   [20, 8, 0,30],
#   [25, 6, 0,15],
#   [30, 5,10,45],
# ]
# min_transit = [
#   500,
#   200,
#   200,
#   100
# ]
# vehicles = [55,95,30,45]

table_1 = []
table_1.append([])
table_1[0].append('№ линии\\textbackslash № судна')
for i in range(len(vehicles)):
	table_1[0].append(i+1)
table_1[0].append('Минимальный объем перевозок')

for index,value in enumerate(tons):
	table_1.append([])
	table_1[index+1].append(index+1)
	for idx,item in enumerate(value):
		table_1[index+1].append(str(item)+'/'+str(cost[index][idx]))
	table_1[index+1].append(min_transit[index])

table_1.append([])
table_1[-1].append('Кол-во кораблей')

for item in vehicles:
	table_1[-1].append(item)
table_1[-1].append('')

writetbl(table_1, 'Исходные данные')

#######################################################
report_write('\\section{Математическая модель}')

report_write('''
Обозначим через $x_{ij}$ --- количество судов, перевозимого по линиям перевозки, $c_{i,j}$ --- стоимость перевозки.

Целевая функция --- $\min F(x_{ij}) = \min\sum_{i=1}^{m}\sum_{j=1}^{n} c_{ij} \cdot x_{ij}$.

Целевая функция отражает минимальные транспортные издержки, при которых запросы всех потребителей удовлетворены.

Требуется определить множество переменных $х_{ij} \\geqslant 0$, удовлетворяющих следующим условия:

$$\sum_{j=1}^{n} x_{ij} \cdot a_{ij} \geqslant a_i, \\text{, где } (i = 1, 2, \dots , m)$$

$$\sum_{i=1}^{m} x_{ij} = N_j, \\text{, где } (i = 1, 2, \dots , m)$$

В ограничениях $a_i$ --- минимальный объем перевозок на линии $i$; $N_j$ --- количество судов вида $j$.

\\subsection{Индивидуальное задание}

Обозначим через $x_{ij}$ число судов типа $j$ $(j = 1,2,3)$, которое планируется закрепить за регулярной линией $i (i = 1,2,3)$.

С учетом введенных обозначений математическая модель задачи:

Целевая функция:
''')

math_model = '\\min Z_{x_{ij}} = \\min_{x_{ij}} \\left( \\right.'
for row,value in enumerate(cost):
	for col,value in enumerate(value):
		math_model += str(value)+'x_{'+str(row+1)+str(col+1)+'}+'
	math_model += ' \\\\\n\t'
math_model = math_model[:-6] + '\\left. \\right)'

writeeq(math_model)

report_write(', при ограничениях:')

restrictions = '\\begin{cases}'

for row, value in enumerate(tons):
	for col, value in enumerate(value):
		restrictions += str(value) + 'x_{' + str(row+1) + str(col+1) + '} +'
	restrictions = restrictions[:-1]
	restrictions += '&\geqslant ' + str(min_transit[row])
	restrictions += ' \\\\ \n\t\t'

for row, value in enumerate(tons):
	for col, value in enumerate(value):
		restrictions += 'x_{' + str(col+1) + str(row+1) + '} +'
	restrictions = restrictions[:-1]
	restrictions += '&= ' + str(vehicles[row])
	restrictions += '\\\\ \n\t\t'

restrictions += 'x_{ij} \geqslant 0, (i=1(1)'+ str(len(tons)) +'), (j=1(1)' + str(len(tons[0])) + ')'
restrictions += '\n\t\end{cases}'

writeeq(restrictions)

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

#######################################################
report_write('\section{Решение}')
report_write('\subsection{Нахождение начального допустимого базисного решения. Метод Данцига.}')

report_write('Приведем индексы матрицы ограничений к такому виду:')

wided_table_cols = len(tons)*len(tons[0])+len(tons)+1
wided_table_rows = 2*len(tons)
wided_table = [[0]*wided_table_cols for i in range(wided_table_rows)]
wided_table_visible = [[0]*(wided_table_cols) for i in range(wided_table_rows+1)]

# header of the table
for i in range(wided_table_cols):
	wided_table_visible[0][i] = '$x_{' + str(i) + '}$'

# first col (restrictions) A_0
for i in range(wided_table_rows):
	try:
		wided_table[i][0] = min_transit[i]
		wided_table_visible[i+1][0] = '$' + str(min_transit[i]) + '$'
	except IndexError:
		wided_table[i][0] = vehicles[i-len(min_transit)]
		wided_table_visible[i+1][0] = '$' + str(vehicles[i-len(min_transit)]) + '$'

# coeffs of restrictions
for row, value in enumerate(tons):
	for col, item in enumerate(value):
		curr_col = len(tons)*col+row+1
		wided_table[row][curr_col] = item
		wided_table_visible[row+1][curr_col] = '$\\mathbf{' + str(item) + '}$'

# additional -1 to make equations with 0
for item in range(len(tons)):
	wided_table[item][len(tons)**2+item+1] = -1
	wided_table_visible[item+1][len(tons)**2+item+1] = '$\\mathbf{' + str(-1) + '}$'

# coeffs of vehicles restrictions
for row, value in enumerate(tons):
	for col, item in enumerate(value):
		curr_col = len(tons)*col+row+1
		wided_table[row+len(tons)][curr_col] = 1
		wided_table_visible[row+1+len(tons)][curr_col] = '$\\mathbf{' + str(1) + '}$'

writetbl(wided_table_visible)
# writetbl(wided_table)


#######################################################

basis_idx = []
basis_val = []
text = 'Вводим в базис произвольные переменные: '

for i in range(wided_table_rows):
	basis_idx.append(i)
	basis_val.append([ row[i] for row in wided_table ])
	text += '$A_{' + str(i) + '}$, '

not_basis_idx = sorted(list(set(range(wided_table_cols)) - set(basis_idx)))
not_basis_val = []
for idx in not_basis_idx:
	not_basis_val.append([ row[idx] for row in wided_table ])
text = text[:-2] + '.'
report_write(text)

#######################################################

aux_array = [random.randrange(0,10) for i in range(len(basis_idx))]

SUCCESS = False
iteration = 1
while iteration < MAX_ITERATIONS + 1:
	report_write('\\subsubsection{Шаг ' + str(iteration) + '}')

	# equations on step
	report_write('\\small')
	for idx, nbas in enumerate(not_basis_idx):
		text = ''
		for i in range(len(basis_val[0])):
			text += '\tA_{'+str(nbas)+'} &= ' + str(not_basis_val[not_basis_idx.index(nbas)][i]) + ' = '
			for bas in basis_idx:
				text += str(basis_val[basis_idx.index(bas)][i]) + 'x_{' + str(bas) + '-' + str(nbas) + '} + '
			text = str(text[:-3]) + ' \\\\ \n' # remove extra plus on the end
		text = str(text[:-5])
		writeeq(text)
	report_write('\\normalsize')

	#######################################################

	basis_solutions = []
	basis_val_transposed = list(np.array(basis_val).transpose())
	for item in not_basis_val:
		basis_solutions.append(
			list(np.linalg.solve(basis_val_transposed, item))
		)

	# rounding result
	# dirty hack with float(str())
	basis_solutions = [[float(str(round(item,ROUND_ACCURACY))) for item in sol] for sol in basis_solutions]


	report_write('Решения уравнений:')

	basis_solutions_visible = [['']+['$x_{'+str(i)+'}$' for i in basis_idx]]
	for nbas_idx,nbas in enumerate(not_basis_idx):
		text = '\\begin{array}{'+'c'*len(basis_solutions[0])+'}\n\t'
		for bas_idx,bas in enumerate(basis_idx):
			text += 'x_{' + str(bas) + '-' + str(nbas) + '} = '\
			+ str(basis_solutions[nbas_idx][bas_idx]) + ';\ '
			text += '\\\\ \n\t' if bas_idx != 0 and bas_idx%3 == 0 else ''
		basis_solutions_visible.append(['$A_{'+str(nbas)+'}$']+basis_solutions[nbas_idx])
		text += '\n\end{array}'
		# writeeq(text)

	writetbl(basis_solutions_visible,caption='Решения уравнений в виде таблицы')

	TO_COMPARE = 0
	report_write('Сравнивая решения при $x_' + str(TO_COMPARE) + '$:')

	text = '\t'
	min_solution = 999999
	min_idx = 0
	for sol_idx, sol in enumerate(basis_solutions):
		text += 'x_{0'+str(not_basis_idx[sol_idx])+'} = ' + str(sol[TO_COMPARE]) + ';\\ '
		text = text[:-3] + '\\\\ \n\t' if sol_idx != 0 and sol_idx%3 == 0 else text
		if sol[TO_COMPARE] > 0:
			if sol[TO_COMPARE] < min_solution:
				min_solution = sol[TO_COMPARE]
				min_idx = sol_idx

	text = text[:-3]
	writeeq(text)

	if min_solution != 999999:
		report_write('Минимальный элемент: $x_{'+\
			str(TO_COMPARE) + '-'+\
			str(not_basis_idx[min_idx])+'} = ' +\
			str(min_solution) + '$.')

	exclude_eq_idx = -1
	theta0 = -1
	for sol_idx,sol in enumerate(basis_solutions[min_idx]):
		# if sol_idx != 0:
		if sol > 0:
			if aux_array[sol_idx]/sol > 0:
				exclude_eq_idx = sol_idx
				theta0 = round(aux_array[sol_idx]/sol,ROUND_ACCURACY)
				break

	# if there's no positive coeffs, it means that we find optimal basis solution
	if (exclude_eq_idx == -1 and theta0 == -1) or min_solution == 999999:
		report_write('Нет положительных коэффициентов, соответственно допустимое базисное решение:')

		text = ''

		for sol_idx,sol in enumerate(basis_solutions[min_idx]):
			text += 'x_{'+str(not_basis_idx[min_idx])+'-'+str(basis_idx[sol_idx])+'}^{\\*} = '\
			+ str(sol) + ',\\ '
			text = text[:-3] + '\\\\ \n\t' if sol_idx != 0 and sol_idx%3 == 0 else text
		text = text[:-3]
		writeeq(text)
		SUCCESS = True
		break

	report_write('Вводим в базис вектор $A_{' + str(not_basis_idx[min_idx]) + '}$ и запишем для него уравнение:')

	report_write('\\small')
	text = ''
	text += 'A_{'+str(not_basis_idx[min_idx])+'} = '
	for sol_idx,sol in enumerate(basis_solutions[min_idx]):
		if sol_idx != 0:
			if sol > 0:
				text += '+'
		text += str(sol) +\
		'x_{'+str(not_basis_idx[min_idx])+'-'+str(basis_idx[sol_idx])+'}'
	writeeq(text)
	report_write('\\normalsize')

	#######################################################

	if iteration == 1:
		report_write('Введем вспомогательный вектор со случайными значениями:')
	else:
		report_write('Вспомогательный вектор на этом шаге:')

	text = ''
	for idx in range(len(aux_array)):
		text += '\\rho_{' if idx == 0 else '\\omega_{'
		text += str(idx)+'}'
		text += ' = '+str(aux_array[idx]) + ';\\ '
	text = text[:-3]
	writeeq(text)

	# including new vector
	basis_idx.append(not_basis_idx[min_idx])
	basis_val.append(not_basis_val[min_idx])
	not_basis_idx.append(basis_idx[exclude_eq_idx])
	not_basis_val.append(basis_val[exclude_eq_idx])
	del not_basis_idx[min_idx]
	del not_basis_val[min_idx]
	del basis_idx[exclude_eq_idx]
	del basis_val[exclude_eq_idx]

	# sorting like in wided table
	basis_idx, basis_val = [list(i) for i in zip(*sorted(zip(basis_idx, basis_val)))]
	not_basis_idx, not_basis_val = [list(i) for i in zip(*sorted(zip(not_basis_idx, not_basis_val)))]

	text = 'Выводим из базиса вектор $A_' +\
		str(basis_idx[exclude_eq_idx]) +\
		'$, т.к. $'+\
		'\\theta_0 = \\frac{'+\
		str(aux_array[exclude_eq_idx])+'}{'+\
		str(basis_solutions[min_idx][exclude_eq_idx])+'}$ = '\
		+ str(theta0)

	for idx,item in enumerate(aux_array):
		aux_array[idx] = round(item + theta0*basis_solutions[min_idx][idx],ROUND_ACCURACY)

	report_write(text)

	iteration += 1

if not SUCCESS:
	report_write('Не найдено решения за ' +\
		str(MAX_ITERATIONS) +\
		' шагов. Или что-то пошло не так или надо увеличить число шагов.')
	report_output.close()
	exit()

report_write('\\subsection{Переход от начального допустимого решения к первой симплекс-таблице}')

report_write('Разложим небазисные векторы по найденному методом Данцига базису:')

simplex_tbl = []
conversion_tbl = []

report_write('\\small')
for nbas_idx,nbas in enumerate(not_basis_val):
	text = ''
	for item_idx, item in enumerate(nbas):
		text += str(nbas[item_idx]) + ' &= '
		for bas_idx, bas in enumerate(basis_idx):
			if bas_idx != 0:
				if not basis_val[bas_idx][item_idx] < 0:
					text += '+'
			text += str(basis_val[bas_idx][item_idx]) +\
			'x_{' + str(bas) + '-' + str(not_basis_idx[nbas_idx]) + '}'

		text += ' \\\\ \n\t'
	text = text[:-3]
	writeeq(text)

report_write('\\normalsize')

report_write('Решая каждую из систем уравнений, получим:')
report_write('\\footnotesize')

for item in not_basis_val:
	conversion_tbl.append(
		list(np.linalg.solve(basis_val_transposed, item))
	)
conversion_tbl = [[float(str(round(j,ROUND_ACCURACY))) for j in i] for i in conversion_tbl]

text = '\\begin{array}{' + 'c'*len(conversion_tbl[0]) + '}\n'
for row_idx,row in enumerate(conversion_tbl):
	text += '\t'
	for item_idx,item in enumerate(row):
		text += 'x_{' +\
		str(basis_idx[item_idx]) +\
		'-' +\
		str(not_basis_idx[row_idx]) + '} = '\
		+ str(item) + ', & '
	text = text[:-4] + ' \\\\ \n'

text = text[:-4] + '\n\\end{array}'
writeeq(text)
report_write('\\normalsize')
############################################################

report_write('\\subsection{Решение методом полного исключения Гаусса}')

# flatten cost array and append zeros to simplex table length
# (-1 coz first item is empty)
cost_flat = list(np.array(cost).flatten())
cost_flat = [''] + cost_flat + [0]*((len(basis_idx)+len(not_basis_idx))-len(cost_flat)-1)

####
# Fill simplex table with values from conversion table if x is not in basis
# if x in basis so 1 at cross, 0 others
####
curr_sol = 0
for idx in range(len(wided_table[0])):
	if idx in basis_idx:
		simplex_tbl.append(
			[1 if i == basis_idx.index(idx) else 0 for i in range(len(basis_idx))] + [0]
		)
	else:
		simplex_tbl.append(conversion_tbl[curr_sol] + [cost_flat[idx]])
		curr_sol += 1

basis_left_col = []
for item in basis_idx:
	basis_left_col.append(float(cost_flat[item]))

simplex_tbl = list(np.array(simplex_tbl).transpose())
simplex_tbl = [list(item) for item in simplex_tbl]

simplex_tbl = [cost_flat] + simplex_tbl

for row_idx,row in enumerate(simplex_tbl):
	for item_idx,item in enumerate(row):
		simplex_tbl[row_idx][item_idx] = float(item) if item != '' else ''

init_target_func = 0
for item_idx,item in enumerate(simplex_tbl):
	if item_idx != 0 and item_idx != len(simplex_tbl)-1:
		init_target_func += float(item[0])*float(basis_left_col[item_idx-1])


simplex_tbl[len(simplex_tbl)-1][0] = init_target_func

direction_col = -1
direction_row = -1

# havePositive = False
SUCCESS = False
iteration = 1
while iteration < MAX_ITERATIONS + 1:
	################################################
	# find direction col and row
	min_col_val = float('NaN')
	for item_idx,item in enumerate(simplex_tbl[-1]):
		if item_idx != 0:
			if float(item) > 0:
				if not item < min_col_val:
					min_col_val = item
					direction_col = item_idx

	if math.isnan(min_col_val):
		direction_row = 0
		direction_col = 0

	if direction_col != 0:
		min_row_val = float('inf')
		for item_idx, item in enumerate(simplex_tbl):
			if item_idx != 0 and item_idx != len(simplex_tbl)-1:
				if simplex_tbl[item_idx][direction_col] > 0.0:

					divided = simplex_tbl[item_idx][0] / simplex_tbl[item_idx][direction_col]

					if divided < min_row_val:
						min_row_val = divided
						direction_row = item_idx

	if direction_col != 0:
		havePositive = False
		for row in simplex_tbl:
			if row[direction_col] > 0:
				havePositive = True
				break

	################################################
	# make visual representation of simplex table
	simplex_tbl_visible = []

	# first row
	simplex_tbl_visible.append(['$c$',''])
	simplex_tbl_visible[0] += ['$' + str(item) + '$' if item != '' else '' for item in simplex_tbl[0]]

	# second row
	simplex_tbl_visible.append(['','$B_x$','$a_{i0}$'])
	for i in range(len(simplex_tbl[0])-1):
		if i == direction_col-1 and direction_col != 0:
			simplex_tbl_visible[1].append('$\\mathbf{A_{'+str(i+1)+'}}$')
		else:
			simplex_tbl_visible[1].append('$A_{'+str(i+1)+'}$')

	# body of table
	for idx in range(len(simplex_tbl)):
		if idx != 0:
			if idx == len(simplex_tbl)-1:
				simplex_tbl_visible.append(
					['','$\\Delta$'] +\
					['$' + str(item) + '$' for item in simplex_tbl[idx]]
				)
			else:
				if idx == direction_row and direction_row != 0:
					simplex_tbl_visible.append(
						['$' + str(basis_left_col[idx-1]) + '$' ,
						'$\\mathbf{x_{'+ str(basis_idx[idx-1]) +'}}$'] +\
						['$\\mathbf{' + str(item) + '}$' if item_idx == direction_col else '$' + str(item) + '$' for item_idx,item in enumerate(simplex_tbl[idx])]
					)
				else:
					simplex_tbl_visible.append(
						[basis_left_col[idx-1] ,'$x_{'+ str(basis_idx[idx-1]) +'}$'] +\
						['$' + str(item) + '$' for item in simplex_tbl[idx]]
					)

	report_write('\\begin{landscape}')
	report_write('\\subsubsection{Шаг ' + str(iteration) + '}')

	writetbl(simplex_tbl_visible,debug=True)

	if direction_col != 0 and direction_row != 0:
		report_write('Направляющий столбец: ' + str(direction_col))
		report_write('Направляющая строка: ' + str(direction_row))
		report_write('Разрешающий элемент: ' + str(simplex_tbl[direction_row][direction_col]))


	report_write('\\end{landscape}')

	if not havePositive:
		SUCCESS = False
		break

	if math.isnan(min_col_val):
		target_func = 0
		for idx in range(len(simplex_tbl)-1):
			if idx != 0:
				target_func += simplex_tbl[idx][0]*basis_left_col[idx-1]
		simplex_tbl[row_idx][item_idx] = target_func
		SUCCESS = True
		break

	new_simplex_tbl = copy.deepcopy(simplex_tbl)

	# recount
	for row_idx,row in enumerate(simplex_tbl):
		# first row, restrictions
		if row_idx == 0:
			continue
		for item_idx,item in enumerate(row):
			if row_idx == direction_row:
				if item_idx == direction_col:
					new_simplex_tbl[row_idx][item_idx] = 1
				divided = float(np.divide(
					simplex_tbl[row_idx][item_idx],
					simplex_tbl[direction_row][direction_col]
				))
				if divided == float('nan') or math.fabs(divided) == float('inf'):
					new_simplex_tbl[row_idx][item_idx] = 0
				else:
					new_simplex_tbl[row_idx][item_idx] = round(divided,ROUND_ACCURACY)
				continue
			if item_idx == direction_col:
				new_simplex_tbl[row_idx][item_idx] = 0
				new_simplex_tbl[len(new_simplex_tbl)-1][item_idx] = 0
				continue

			divided = float(np.divide(
						simplex_tbl[direction_row][item_idx],
						simplex_tbl[direction_row][direction_col]
					))

			if divided == float('nan') or math.fabs(divided) == float('inf'):
				new_simplex_tbl[row_idx][item_idx] = round(item,ROUND_ACCURACY)
			else:
				new_simplex_tbl[row_idx][item_idx] = round(
					item - (simplex_tbl[row_idx][direction_col]*divided),
					ROUND_ACCURACY
				)

	basis_left_col[direction_row-1] = cost_flat[direction_col]
	basis_idx[direction_row-1] = direction_col

	target_func = 0
	for idx in range(len(simplex_tbl)-1):
		if idx != 0:
			target_func += new_simplex_tbl[idx][0]*basis_left_col[idx-1]
	new_simplex_tbl[len(new_simplex_tbl)-1][0] = target_func

	simplex_tbl = new_simplex_tbl

	iteration += 1

if not SUCCESS:
	if iteration == MAX_ITERATIONS:
		report_write('Не найдено решения за ' +\
			str(MAX_ITERATIONS) +\
			' шагов. Или что-то пошло не так или надо увеличить число шагов.')
	else:
		report_write('В направляющем столбце нет положительных элементов. Это значит, что целевая функция убывает.')

if SUCCESS:
	print('done')

report_output.close()