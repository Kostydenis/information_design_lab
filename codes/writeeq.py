def writeeq(eq,num=False,debug=False):
	if IS_OUTPUT or debug:
		output = '\\vspace{-\\baselineskip}'
		output += '\\begin{align}\n' if num else '\\begin{align*}\n'

		output += eq

		output += '\n\\end{align}' if num else '\n\\end{align*}'
		report_write(output,debug)