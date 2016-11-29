filename=main

pdf:
	pdflatex ${filename}.tex
	rm -f ${filename}.{ps,log,aux,out,dvi,bbl,blg,toc,tmp,xref,4tc,4ct}

links:
	pdflatex ${filename}.tex
	pdflatex ${filename}.tex
	rm -f ${filename}.{ps,log,aux,out,dvi,bbl,blg,toc,tmp,xref,4tc,4ct}

bib:
	pdflatex ${filename}
	bibtex ${filename}||true
	pdflatex ${filename}
	pdflatex ${filename}
	rm -f ${filename}.{ps,log,aux,out,dvi,bbl,blg,toc,tmp,xref,4tc,4ct}

open:
	open ${filename}.pdf -a Skim

clean:
	rm -f ${filename}.{ps,pdf,log,aux,out,dvi,bbl,blg,toc,tmp,xref,4tc,4ct}

# pdf: ps
# 	ps2pdf ${filename}.ps

# pdf-print: ps
# 	ps2pdf -dColorConversionStrategy=/LeaveColorUnchanged -dPDFSETTINGS=/printer ${filename}.ps

# text: html
# 	html2text -width 100 -style pretty ${filename}/${filename}.html | sed -n '/./,$$p' | head -n-2 >${filename}.txt

# html:
# 	@#latex2html -split +0 -info "" -no_navigation ${filename}
# 	htlatex ${filename}

# ps:	dvi
# 	dvips -t letter ${filename}.dvi

# dvi:
# 	latex ${filename}
# 	bibtex ${filename}||true
# 	latex ${filename}
# 	latex ${filename}
