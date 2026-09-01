MAIN=main
OUT=../../output/pdf

.PHONY: all clean

all:
	mkdir -p $(OUT)
	xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN)
	xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	cp $(MAIN).pdf $(OUT)/dual_product_structure_fw.pdf

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf
