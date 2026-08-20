# Supervisor methods presentation

This presentation explains every reliability source and downstream decision method
separately. Its figures are newly drawn vector schematics: they do not consume old runs,
plots, summaries or fitted artifacts and they make no empirical claim.

Build:

```bash
cd experiments/usable_observation/supervisor_comparison/presentation
pdflatex -interaction=nonstopmode -halt-on-error supervisor_methods_deck.tex
pdflatex -interaction=nonstopmode -halt-on-error supervisor_methods_deck.tex
```

Outputs:

- `supervisor_methods_deck.pdf` — presentation-ready 16:9 deck.
- `supervisor_methods_deck.tex` — editable Beamer source.

The deck deliberately separates:

1. sources that estimate `p_use`;
2. camera selection and fusion;
3. belief-update approximations;
4. empirical evidence, which remains blocked pending fresh collection.
