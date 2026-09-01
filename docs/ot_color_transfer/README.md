# OT color-transfer documentation

This folder contains the write-up for the semi-relaxed OT color-transfer setup:

- `main.tex` - main LaTeX source.
- `references.bib` - literature references.
- `figures/` - copied experiment figures used by the PDF.
- `dbcfw_ot_color_transfer.pdf` - compiled PDF output.

Build locally from this directory:

```bash
xelatex -interaction=nonstopmode -halt-on-error -jobname=dbcfw_ot_color_transfer main.tex
bibtex dbcfw_ot_color_transfer
xelatex -interaction=nonstopmode -halt-on-error -jobname=dbcfw_ot_color_transfer main.tex
xelatex -interaction=nonstopmode -halt-on-error -jobname=dbcfw_ot_color_transfer main.tex
```

The experiment results referenced in the document are under:

```text
../../experiments/dbcfw_timevarying_benchmark/runs_ot_color/figure9_rocket_to_coffee/
```
