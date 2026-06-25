# Revised IEEE paper — `paper/`

LaTeX (IEEEtran) source of the COMP468 final paper, revised to address the peer
review. The original was written in Word; this is the LaTeX re-implementation
with the requested additions.

## Deliverables
| File | Format |
|---|---|
| `Final_Project_Paper_Bakir_Baskal_IEEE.pdf` | **IEEE two-column PDF, 15 pages** |
| `Final_Project_Paper_Bakir_Baskal_IEEE.docx` | **IEEE two-column Word** (native equations, editable) |
| `main.tex` + `references.bib` | LaTeX (IEEEtran) source for Overleaf |

## How to build

### A. Word + PDF (the submitted IEEE files) — no LaTeX needed
The two-column Word/PDF are produced from `main.tex` with a three-step pipeline
(needs `pypandoc_binary`, `docx2pdf`, `python-docx`, `pymupdf`, and Microsoft
Word installed for the PDF step):
```
# 1. LaTeX -> Word (equations + figures + IEEE-numbered refs)
python -c "import pypandoc; pypandoc.convert_file('main.tex','docx',outputfile='_ieee_base.docx',extra_args=['--citeproc','--bibliography=references.bib','--csl=ieee.csl','--resource-path=.;figures'])"
# 2. impose IEEE two-column layout (Times, margins, headings, section break)
python build_ieee_docx.py
# 3. Word -> PDF
python -c "from docx2pdf import convert; convert('Final_Project_Paper_Bakir_Baskal_IEEE.docx','Final_Project_Paper_Bakir_Baskal_IEEE.pdf')"
```
`build_ieee_docx.py` does the IEEE styling; `ieee.csl` gives the numbered `[1]`
citation style.

### B. Pure LaTeX (Overleaf) — most authentic IEEE typesetting
Zip the whole `paper/` folder (including `figures/`), upload to Overleaf, set the
main file to `main.tex`, press *Recompile*. Or locally with MiKTeX / TeX Live:
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## What changed vs. the submitted PDF — mapped to each feedback item

| Reviewer item | What was added | Where |
|---|---|---|
| **Format / length (1.5→2):** under 15 pages | Added two Results subsections, a Methodology math layer, two tables, two figures, and an algorithm block — pushes the two-column build past 15 pp. | throughout |
| **Lit review (2.5→3):** mathematical critique of why linear models fail on zero-inflated demand | New subsection: OLS fits the affine conditional mean; censoring at `[0,92]` with a point mass at 0 makes the true conditional mean nonlinear (Tobit/hurdle motivation). | §II.B `sec:linearcritique` |
| **Methodology (8.5→10):** no equations for target encoder, MLP, SLSQP | Eq. (2)–(3) target encoder + additive smoothing + K-fold scheme; Eq. (7)–(11) full MLP (affine/BN/ReLU/dropout, log1p target, Adam); Eq. (12)–(13) early-stopping + search objective + Algorithm 1 + Table II; Eq. (14)–(16) SLSQP convex objective with equality + box constraints. | §IV.C, §IV.D.4, §IV.E, §IV.F |
| **Results (8.5→10):** no significance test; residual plot mandatory | Paired t-test + Wilcoxon on per-sample squared errors (`t=4.55, p=5.3e-6`, 95% CI `[0.96, 2.42]`); residual diagnostics (Fig. 6) and per-bin signed bias (Fig. 7) on the `[0,92]` boundaries. | §V.D `sec:significance`, §V.E `sec:residuals` |
| **References (0.5→1):** missing DOIs | DOIs added to all journal/book entries in `references.bib`. Three carry a `% VERIFY DOI` comment — confirm those on the publisher page before camera-ready (they were not invented blindly, but cross-check). | `references.bib` |

## New analysis is reproducible
The significance numbers and the two residual figures come from
`../notebooks/10_Residuals_and_Significance.py`, which recomputes everything from
the saved OOF predictions in `../outputs/`. Re-run with:
```
python notebooks/10_Residuals_and_Significance.py
```
Outputs: `outputs/residual_analysis.png`, `outputs/residuals_by_truebin.png`,
`outputs/significance_report.txt` (copied into `paper/figures/` as Fig. 6–7).

## Notes
- The figures in `figures/` are copies of the plots in `../outputs/`. Fig. 1–5
  are the originals from the paper; Fig. 6–7 are new.
- A real `5x2cv` paired t-test (Dietterich, 1998) would require re-running the
  whole pipeline over 5 two-fold splits; the paper notes this and reports the
  sample-level paired test that our compute budget allowed. If you want, the
  `5x2cv` variant can be added on top of the existing notebooks.
