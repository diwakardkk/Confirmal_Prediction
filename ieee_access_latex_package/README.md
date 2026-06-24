# IEEE Access LaTeX manuscript package

This package contains a complete manuscript drafted from the generated primary, Pima secondary-benchmark, and NHANES external-transportability outputs. The source manuscript is `main.tex`; bibliography entries are in `ref.bib`.

## Compile

Install the official IEEE Access LaTeX template so that `ieeeaccess.cls` is available, then run:

```bash
latexmk -pdf main.tex
```

Alternatively, use the IEEE Access Overleaf template and upload this folder's contents.

## Package contents

- `main.tex`: manuscript source.
- `ref.bib`: cited references.
- `figures/`: selected 600-dpi primary-analysis figures.
- `tables/`: selected CSV and XLSX tables cited in the manuscript.
- `results/`: the full generated primary-analysis table set and reproducibility manifest.

The NHANES figure cited by `main.tex` is included in `figures/`. The full NHANES and Pima output directories remain local generated artefacts and are deliberately excluded from version control; rerun commands are documented in the repository-level README.

## Required author edits before submission

Replace the author, affiliation, correspondence, funding, DOI, and ethics placeholders. Confirm the target journal template and review all citations against the publisher version. Update the linked GitHub repository to the final analysis commit and archive a tagged release before submission. Do not claim clinical deployment readiness: the NHANES results demonstrate that target-domain recalibration of probabilities, thresholds, and conformal scores is required.
