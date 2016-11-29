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