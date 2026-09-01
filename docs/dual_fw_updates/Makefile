MAIN=main
OUT=../../output/pdf

.PHONY: all clean

all:
	mkdir -p $(OUT)
	xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	cp $(MAIN).pdf $(OUT)/dual_fw_updates.pdf

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).pdf
