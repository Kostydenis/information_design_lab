def writetbl(t, caption='', debug=False):
	if IS_OUTPUT or debug:
		output =  '\\begin{table}[H]\n'
		output += '\t\\centering\n'
		output += '\t\\normalsize\n'
		output += '\t\\caption{' + str(caption) + '}\n'
		output += '\t\\label{tbl:' + str(printed_table_no) + '}\n'
		printed_table_no += 1
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