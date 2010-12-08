# default is to just make a pdf (invoke twice to get references
# right)... if using a bibtex file, add an extra pdflatex and bibtex
# before the two pdflatex commands.

all:
	pdflatex nskhotos.tex
	bibtex nskhotos
	pdflatex nskhotos.tex
	pdflatex nskhotos.tex

# make an alright-looking HTML file (note the sed command is used to
# insert a space after footnotes (silly latex2html)).

html:
	cp nskhotos.tex sun_html.tex
	pdflatex sun_html.tex
	latex2html -split 0 -show_section_numbers -local_icons -no_navigation sun_html
	sed 's/<\/SUP><\/A>/<\/SUP><\/A> /g' < sun_html/index.html > sun_html/index2.html
	mv sun_html/index2.html sun_html/index.html
	rm sun_html.*

# when done, do "make" and copy pdf to a new file.

final:
	make all
	mv nskhotos.pdf sun.pdf

# makes a date directory, copies relevant files there, tars and zips
# the directory and then removes the directory.

archive:
	mkdir `date "+%F"`
	cp *.tex *.pdf *.txt Makefile ./`date "+%F"`
	tar cvf ./`date "+%F"`.tar ./`date "+%F"`
	bzip2 ./`date "+%F"`.tar
	rm -rf ./`date "+%F"`

# cleans up intermediate files

clean:
	rm *.log *.aux *.ent *.blg
