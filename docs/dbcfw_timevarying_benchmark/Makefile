TEX := main.tex
PDF := dbcfw_timevarying_benchmark_report.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX) references.bib figures/main_result_one_graph_with_arrows.png
	xelatex -interaction=nonstopmode -halt-on-error $(TEX)
	bibtex main
	xelatex -interaction=nonstopmode -halt-on-error $(TEX)
	xelatex -interaction=nonstopmode -halt-on-error $(TEX)
	mv main.pdf $(PDF)

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.pdf $(PDF)

