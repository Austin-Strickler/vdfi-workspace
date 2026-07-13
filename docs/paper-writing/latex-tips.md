# LaTeX tips — units, symbols, macros

Companion notes to `macros.tex`. Explains the *why* behind the macros so you can extend them yourself.

## Why `\AA` eats the space after it

`\AA` is a control **word** (letters only, no number/symbol). LaTeX's tokenizer swallows any whitespace immediately following a control word — that's true of every letter-named command (`\LaTeX`, `\alpha`, your own `\lya`), not just `\AA`. So:

```
5007\AA emission   ->   5007Åemission
```

Three ways to fix it, in order of how much you want to think about it:

1. **`\xspace`** (what `macros.tex` uses) — end the macro definition with `\xspace` and it inserts a space automatically, except before punctuation (`.`, `,`, `)`) or another space, where a space would look wrong. This is why `\Ang` is defined as `\AAns\xspace` rather than plain `\AA`.
2. **Manual escape** — type `\AA\ ` (backslash-space) or `\AA{}` every single time. Works, but you will forget it constantly.
3. **`siunitx`** — sidesteps the whole issue; see below.

Gotcha with `\xspace`: don't follow the macro with `{}` or another macro that itself starts with a letter and no space — `\xspace` occasionally over- or under-guesses in unusual token sequences. Rare in practice, but if spacing looks wrong once in a while, that's the first thing to check.

## `\sim` vs `\approx` vs `\simeq`

All three render as squiggle-family symbols and get mixed up constantly. Rough convention in astronomy papers:

- **`\sim`** (∼) — "roughly," order-of-magnitude, a value you haven't pinned down precisely. This is the one you want for redshift ranges, rough counts, ballpark quantities: *"the sample spans z ∼ 2–3"*, *"∼50 objects."* By far the most common of the three in running text.
- **`\approx`** (≈) — two specific, computed quantities that are numerically close. Reads more like a weakened `=`: *"x ≈ 3.14."* Use this when you're making an actual approximation claim about a calculation, not just being vague about a value.
- **`\simeq`** (≃) — "approximately equal to" in a more formal/structural sense (also read as "similar to" or, in stats, "distributed as"). Less common in body text; you'll mostly reach for `\sim` or `\approx` instead.
- **`\propto`** (∝) — proportional to, not really part of this family but gets confused with it sometimes.

`macros.tex` wraps `\sim` and `\approx` in `\ensuremath{...}\xspace` (as `\sm` and `\ap`) specifically so you can type `z\sm2` directly in text without dropping into `$...$` by hand, and it won't break if you happen to use it inside an existing equation.

## Is a macro even worth it?

Worth defining a macro for:
- Anything you'll type more than a handful of times per paper (emission lines, recurring units, recurring symbols).
- Anything with fiddly syntax you'll forget (ion notation, subscripts/superscripts, spacing rules).
- Anything you want to be able to change globally later — e.g. if you decide `\lya` should render as "Lyα" instead of "Ly$\alpha$" everywhere, you edit one line instead of hunting through the manuscript.

Not worth it: a symbol you use once or twice — just type it inline.

## Naming collisions

`\newcommand` throws an error if the name is already taken; that's a feature, not a bug — it stops you from silently clobbering something. A few names to watch for:

- `\ang` — reserved by `siunitx` for angles (degrees), so the Angstrom macro here is `\Ang` instead.
- Anything already defined by your document class (AASTeX defines quite a bit — `\ion`, various journal-style commands). If `\newcommand` errors on a name you want, either pick a different name or use `\renewcommand` deliberately if you actually mean to override it.
- `\providecommand{name}{...}` defines a command only if it isn't already defined — handy for the `\ion` fallback in `macros.tex`, since it lets the same file work whether or not you're using an AASTeX class.

## The alternative: `siunitx`

For anything unit-heavy, `siunitx` is the standard package astronomers reach for instead of (or alongside) hand-rolled macros. It handles spacing, upright roman unit fonts, and text/math-mode agnosticism automatically:

```latex
\usepackage{siunitx}
\DeclareSIUnit\angstrom{\text{\AA}}   % if not already defined by your siunitx version

The line sits at \SI{5007}{\angstrom}.
Flux: \SI{3.2e-17}{\erg\per\second\per\cm\squared\per\angstrom}.
```

`\SI{value}{unit}` and `\si{unit}` insert the correct spacing and symbol regardless of math/text mode — no `\xspace` bookkeeping needed. Tradeoff: more setup/syntax to learn, and you lose the terse `\Ang`/`\kms`-style shortcuts unless you also define those as wrappers around `\SI`. Reasonable approach: use the custom macros in `macros.tex` for the things you type constantly (line names, `\Ang`, `\sm`), and reach for `siunitx` directly for one-off or compound units that don't have a macro yet.

## Keeping `master.bib` in sync

Simple workflow: when you add or edit a reference while writing a specific paper, copy that entry back into `master.bib` before you forget which paper it lived in first. Alphabetize by BibTeX key (`Author:Year`) so diffs between the master and a paper's local copy are easy to eyeball.
