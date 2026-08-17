# Workflow & tooling ideas

**Status: proposals, none implemented.** Captured 2026-08-01 from a working session on where
day-to-day research time actually goes. Nothing here is a commitment. Each item has a "verdict"
line so we can kill the bad ones cheaply.

These are *workflow* proposals, not pipeline specs. If one graduates to implementation it gets a
real spec in `01-lya_halos/specs/` and this entry becomes a pointer.

**Ordering.** Items 2 and 4 are cheap and use infrastructure that already exists — do those first.
Item 1 needs a one-time cleanup before it's worth building. Items 3 and 5 are larger and
conditional.

---

## Table of contents

| # | Idea | Cost | Verdict |
|---|---|---|---|
| 1 | Obsidian literature-note loop | Medium (blocked on vault cleanup) | **Doing first** — cleanup session, then loop |
| 2 | `guide.py` tunables registry | Low | Build next |
| 3 | `WHY.md` per module | Low each, needs discipline | Conditional — prototype one |
| 4 | Unified plot style | Low | Build, in `formatting.py` |
| 5 | Interactive 3D Lyα halo explainer | High | Worth it, but stage it |
| 6 | Subscription / capacity strategy | — | **Settled: no change.** Stay on Pro. |

---

## 1. Obsidian literature-note loop

### The problem

Reading papers is manual for a good reason — you need them in your head, not in a summary. What's
lossy is not the reading, it's the **reaction**. Six months later, "what did I think about
Niemeyer 2022, and did I trust their systematics?" is unanswerable. The judgment evaporates; only
the citation survives.

The Obsidian vault was started in summer 2026 with roughly this intent and has accumulated
duplicate literature entries since.

### Proposed loop (two-pass, decided 2026-08-01)

Runs as a **separate Cowork project mounted on the Obsidian vault**, not this workspace.

**Pass 1 — react**

1. **You read the paper.** No PDF highlighting required — that was an assumption, not a
   requirement. Quotes can be dictated, pasted, or pulled by Claude from the PDF when you point at
   a section. Whatever's least friction that day.
2. **You write/dictate your thoughts** into the note, using the template.
3. **Claude reads both the paper and your thoughts** and responds **in chat only**:
   - a quick reaction to what you said — where it holds, where it doesn't
   - things the paper says that you did *not* flag, and might care about
   - related work along the lines you pointed at (fenced — see below)

**Pass 2 — settle**

4. **You go look into the flagged items** on your own time.
5. **Next session, you give the final read.** Claude writes the finished note to the vault from the
   template.

### The template contains only your material

**Hard rule: Claude's suggestions never enter the template.** They live in the chat/discussion
layer. Anything of theirs that survives does so because *you* restated it in pass 2 — at which
point it's your note, in your words.

This is what makes the fencing below almost automatic: there is no mechanism by which an unverified
suggestion lands in the vault, because Claude never writes during pass 1.

### Note schema (fixed — the point is comparability across notes)

```markdown
---
authors:      
year:         
arxiv:        
doi:          
tags:         [lya-halo, kinematics, ...]
relevance-to: [centroid-vs-radius, LIS-calibration, ...]
read-date:    
---

## What they measured
One paragraph, factual, no interpretation.

## My reaction
Your dictation, cleaned of disfluency but NOT sanitized or rephrased.

## Do I trust it
The referee read. Systematics you doubt, sample issues, overreach.
*This is the field nobody writes and everybody later wishes they had.*

## Where it touches VDFI
Explicit wiki-links: [[centroid-vs-radius]], [[air-vacuum-correction]].
State the relationship — supports / contradicts / method we could borrow.

## Quotes
Page- or section-anchored. Verbatim.

## Follow up
Things to chase. Yours, written in pass 2.
```

Template lives in an Obsidian templates folder inside the vault (not `vdfi-workspace/templates/`),
so Templater/QuickAdd can instantiate it natively.

### The verification step — is it trustworthy? (open question, partially resolved)

Split into two very different operations:

**(a) Verifying a claim you made against a paper you already have.** Reliable. You say "I think
they used a 2\" aperture" — the paper is fetchable, the answer is checkable. Low risk. Do this.

**(b) Finding "related interesting ideas."** This is where it can go wrong, in a specific way:
plausible-sounding connections that don't survive reading the actual paper, and — worse —
citations that look right and aren't. This failure is quiet, which is what makes it dangerous in a
literature note you'll trust in six months.

**Mitigations, all mandatory if we build this:**

- Every suggested item must carry a **resolvable identifier** (arXiv ID or DOI) that was actually
  fetched during the session. No identifier → don't mention it.
- Each carries a provenance tag in chat: `[verified: abstract read]`, `[verified: full text]`, or
  `[suggested: unread — go check]`.
- **The two-pass structure does most of the work.** Suggestions live in chat and only reach the
  vault by passing through you in pass 2. There is no path by which an unverified citation is
  silently written.

**Verdict:** valid under the two-pass design. The earlier single-pass version (Claude writes
directly, suggestions quarantined in a `Related (unverified)` section) was the riskier shape —
quarantined text still ends up in the vault, and in six months a tag is easy to miss. Two-pass is
strictly better and simpler. Superseded.

### Blocker — do this first

The vault already has duplicate literature entries from summer. A one-time consolidation pass —
inventory, find duplicates and orphans, propose a single structure — has to happen before any
template or loop is worth building. Otherwise this creates a third copy of everything.

**This is a separate session in a new project mounted on the vault.** Nothing else in this document
should start before it.

### Folder plumbing

Cowork mounts one folder per session, which matches the intended split:

- project mounted on `vdfi-workspace` → pipeline work
- project mounted on the Obsidian vault → literature work
- Claude Code can see both simultaneously if cross-referencing is ever needed

Do **not** merge the vault into `vdfi-workspace`.

### Not required

Earlier draft assumed Obsidian Annotator / in-PDF highlighting. **Dropped** — highlighting inside
PDFs is friction you don't want. Quotes get into the note by whatever route is cheapest in the
moment; the loop doesn't depend on any particular one.

---

## 2. `guide.py` tunables registry

### The problem

Reading a function tells you what it does. It does not tell you **which knobs matter, what happens
when you turn them, and which ones are degenerate with each other.** That is the actual daily
friction: "I want to try something different and I don't know what I'm allowed to change."

Asking for "explain this function" reliably produces a description of the happy path, which is not
the same thing.

### The infrastructure already exists

`guide.py` (2287 lines) registers entries via:

```python
_e(name, module, section, sub, desc, example)
```

`check_guide()` already audits coverage. Today this is a *what exists* index. The proposal is to
make it also a *what can I turn* index.

### Proposed extension

Add two optional fields to the entry: `tunables` and `wont_work_if`.

```python
_e("measure_twohalo_luminosity", "measure", "measurement", "luminosity",
   "Integrates flux beyond the fitted crossover radius.",
   example="...",
   tunables={
       "r_c_fixed":   "400.0 kpc. Fixes the outer transition. Set None to fit it — "
                      "fits invert when A1 < A2; see the A2 = f*A1 ordering fix (2026-07-21).",
       "gamma_fixed": "0.8. Degenerate with r_c. Only unfix if r_c is also fixed.",
       "n_boot":      "500. Drop to 50 for a fast look; errors come out ~20% optimistic.",
   },
   wont_work_if="bin_mode is 'virial' — the crossover radius is in kpc.")
```

### Why this shape

- **Machine-checkable.** `check_guide()` can assert that every keyword argument in the signature
  appears in `tunables`. That is the only thing that stops this rotting within two months.
- **Describes the effect, not the type.** "500 → 50 gives ~20% optimistic errors" is usable.
  "n_boot: int, number of bootstrap draws" is not.
- **`wont_work_if` is where the "can I try something new?" answer lives** — it encodes the
  preconditions that are currently only in your head or in a memory file.

### Better prompt to use in the meantime

Instead of "explain this function":

> For each parameter, tell me what *visibly changes in the output* if I move it, which parameters
> are degenerate with each other, and what would break.

Different question, materially better answer. Worth using even before the registry exists.

### Cost

Low for the mechanism. The real cost is backfilling ~25 modules, which should be done lazily —
fill in `tunables` for a function the next time you touch it, not in one sitting.

---

## 3. `WHY.md` per module

### The problem it solves

Physics and statistics decisions are baked into the code and recorded nowhere durable. Some live in
memory files, some in the systematics markdown, some nowhere. When the Methods section gets
written, they have to be reconstructed.

### The objection (valid)

A doc that takes 5 minutes to generate and 50 minutes to read is a net loss. Unbounded AI-written
"why" text is exactly that.

### The fix is structural, not cosmetic

Prettiness does not solve this. Three hard constraints do:

1. **One page. Hard cap.** If it exceeds one page, the module is doing too much — that is the
   signal, not "trim the doc."
2. **Fixed table, falsifiable columns.** The "would be wrong if" column cannot be padded; you
   either know the failure condition or you don't.
3. **Written in Methods voice.** It has to survive being pasted into the paper. If it reads like
   filler, it also reads like a bad Methods section, and you catch it immediately.

### Template

```markdown
# Why: <module>.py

| Decision | What we do | Why | Would be wrong if |
|---|---|---|---|
| Core boundary | Fitted crossover per subsample | A fixed radius biases low-mass bins | h1 varies >2x across the sample |
| ... | | | |

## The one thing a referee will ask


## What we tried and abandoned

```

### Verdict

Conditional. Prototype exactly one — `measure.py` is the obvious candidate — and judge the real
artifact rather than the idea. If the prototype is not something you'd paste into a Methods
section, kill the whole proposal.

---

## 4. Unified plot style

### Home: `formatting.py` (decided 2026-08-01)

`formatting.py` is currently LaTeX/text table formatting only — `rows_to_latex_table`,
`make_subsample_latex_table`, sig-fig and asymmetric-error handling — and has no matplotlib
dependency today.

**Decision: the plot style goes in `formatting.py` anyway.** The module's scope is *formatting
outputs*, and figure formatting is that. At 368 lines it has plenty of room, and the repo already
has a lone-file / fragmentation problem that a new `plotstyle.py` would worsen.

Consequence to be aware of: this adds matplotlib as an import dependency of `formatting.py`, so
anything importing it purely for LaTeX tables pulls matplotlib in. Minor, but not zero.

`plotting.py` (2472 lines) and `analysis.py` (which also draws figures — `plot_flux_profile_fit`,
`plot_centroid_profile`) both import from `formatting.py`. So the style must be a module-level
rcParams application plus shared constants that both import — **not** per-function styling.

### What a style layer can reliably do

- **Font sizes correct at final print scale.** The single most common amateur tell is 6pt axis
  labels because the figure was drawn at one size and shrunk to column width.
- **Stable color assignment across figures.** "Low-density subsample" is the same color in every
  panel of every talk and the paper. This is the thing that makes figures read as one person's work.
- **Consistent error-bar / upper-limit convention.** Bars for sparse points, bands only for dense
  curves. Currently inconsistent.
- **Correct single- vs double-column widths** for the target journal.
- Colorblind-safe ramps.

### What it cannot do

Make a badly-conceived figure good. Axis choice, panel layout, and what to show remain yours.

### The "will this look weird" concern

The fix is: **do not invent a style.** Derive it from the target journal (ApJ / MNRAS) plus three
published figures you actually admire. Realistic ceiling — figures that look like they came from a
careful person who publishes in ApJ. Nobody should notice them; they should just read them.

**Action before building:** pick three reference figures from real papers. The style module is
derived from those, not from taste.

---

## 5. Interactive 3D Lyα halo explainer

### What this is (and is not)

**Not a simulation.** The reference class is a good astronomical illustration — an artist's
rendering of Saturn's rings where the artist genuinely understood the ring dynamics, or the
schematic BAO 100 Mpc overdensity-shell visualizations. Those are stylized and exaggerated on
purpose. That is exactly why they teach.

The standard is: **every element is grounded in the radiative transfer literature (Dijkstra,
Verhamme, and the hardcore RT simulation work), even though nothing is being simulated live.** The
exaggeration is in the presentation, never in the physics being represented.

### The object being depicted

A cosmic-noon galaxy at three nested scales, all in one view:

- **Core** — UV-bright, compact, the Lyα production site
- **Scattered exponential envelope** — the halo, at the measured h1 (~16–17 kpc)
- **2-halo / clustering regime** — satellite galaxies at the outer transition scale
- **Gas structure throughout** — H I column, H II regions, small-scale inflows and outflows per
  sub-galaxy

And layered on top: **the observed spectral shape, and how it forms.**

### Interaction model

- **Orbit** the object freely.
- **Radius slider** — highlights the corresponding annulus in 3D *and* updates the spectrum panel
  below to the measured stack at that radius. This directly renders the Paper I headline: the
  centroid drifting with radius, and the transition at R_vir.
- **Column density slider** — drives the double-peak separation. As N_HI rises, Lyα scatters
  further out of resonance into the wings and the peaks separate; as it falls, they merge toward
  one. *Feasibility: realistic.* The peak separation vs. N_HI relation for expanding-shell models
  is tabulated in the RT literature; the slider interpolates a precomputed grid rather than solving
  anything. Must be labelled as a shell-model illustration, not a fit to our data.
- Component toggles (core / envelope / satellites / gas).

### Staging

| Stage | Scope | Cost | Value |
|---|---|---|---|
| A | Schematic 3D + radius slider + measured spectrum panel | ~1 week | 80% of the talk value; directly serves Paper I |
| B | Add N_HI slider driving double-peak separation from a precomputed RT grid | +1–2 weeks | Explains *how the spectrum forms*, which is the part that's verbally awkward |
| C | Live toy Monte Carlo RT (photons random-walking in an expanding shell) | weeks | Pedagogical, not research-grade |
| D | Real simulation snapshot | subproject / paper | Do not start here |

Stage A is the commitment; B is the one that makes it genuinely good. Browser + Three.js, so it
runs from a link in a talk with no install.

### Non-negotiable

Label it "schematic — not to scale" on the slide. That both prevents anyone mistaking it for a
simulation and frees the illustration to exaggerate for clarity.

### Grounding work required before building

A short reference pass through Dijkstra's Lyα RT review and Verhamme's shell-model papers, to fix:
peak separation vs. N_HI, the outflow-velocity dependence, and the geometry conventions. This is
the "understand Saturn's rings before drawing them" step and it is not optional.

---

## 6. Subscription / capacity strategy

Recorded because it constrains everything above. **Not a build item.**

### Situation (2026-08-01)

- Currently on Claude Pro ($20/mo). Hitting weekly caps; the 5-hour rolling cap is tolerable.
- Summer (~40 hr/wk) is ending; fall load is ~10 hr/wk.
- Observed self-pattern: short, dense sessions are more productive than 8-hour days. Even at 10
  hr/wk, an idea can go from concept to agentically-built artifact in one sitting.
- API pay-as-you-go was tried and rejected — $100 of free credit burned fast with poor
  value-per-dollar versus a month of Pro.

### Options and honest assessment

| Option | Cost/yr | Assessment |
|---|---|---|
| Stay on Pro | $240 | Likely sufficient at 10 hr/wk. Default. |
| Pro + ChatGPT Pro | $480 | Only helps if the offloaded work is what burns the cap. Splits the workflow across two vendors. |
| Max 5x | $1200 | Fixes the actual constraint if the constraint is real. Revisit only if grant-funded. |
| Two Claude Pro accounts | $480 | **Not recommended — see below.** |
| API overflow | variable | Already tried, rejected. |
| Institutional access (Penn State) | possibly $0 | **Unexplored. Check this first.** |

### On two Pro accounts

Multiple accounts are not per se banned — Anthropic's stated concern is account *sharing* and
token *reselling*. But two practical problems:

1. **Abuse detection.** Running two subscriptions from one machine produces the same behavioral
   fingerprint as reselling. There are documented reports of usage allowance being shared or
   rate-limited across separate accounts on the same machine — i.e. it may not even deliver the
   extra capacity.
2. **Workflow fragmentation.** Memory files, skills, project context, and scheduled tasks are
   per-account. The whole value of the setup built over the summer is continuity. Splitting it in
   half to save $60/yr over Max-adjacent capacity is a bad trade.

### Separate two things that keep getting blended

- **"I want a second opinion from a different model"** — free. Free-tier GPT/Gemini is fine for
  "does this reasoning hold up." Different models fail differently, which is exactly the value for
  a referee-proofing pass. No subscription needed.
- **"I need more capacity"** — paid. And a second $20 chat plan only helps if the offloaded work is
  what's actually burning the cap. Long agentic sessions cannot move to a chat model.

### Penn State institutional access — researched 2026-08-01

**Verdict: does not solve the problem. Your read on AI Studio was right.**

*AI Studio* is Penn State's enterprise portal, free to employees, Provost-funded, student rollout
in fall 2026. It serves **Claude Sonnet 4.6 and Haiku 4.5**, GPT 4.1/5.4, Gemini 3 flash / 3.1 pro.

Why it doesn't help:

- **Not the frontier models.** Sonnet 4.6 and Haiku 4.5, no Opus. The gap matters for the work
  you'd actually want it for.
- **Web portal only.** No agentic loop, no filesystem, no repo. It's a chat box. The entire value
  of the current setup is that the model reads and writes `utils_lya_halo/` directly.

**The one non-obvious angle: Amazon Bedrock.** Penn State's "Platforms for Building with AI" page
lists Bedrock (which serves Anthropic models) as a cloud platform faculty/staff can request via
ServiceNow. Claude Code supports Bedrock as a backend. So in principle: Claude Code, Anthropic
models, billed to a Penn State cloud account.

Caveats, and they're significant:

- **"Purchase or paid cloud usage may be required"** — this is not free access, it's a provisioned
  account you or the grant pays into. Token economics identical to the API path you already
  rejected.
- Frontier-model availability on Bedrock typically lags Anthropic's direct API.
- Requires a ServiceNow cloud-account request; may need PI involvement.

**Where it changes the calculus:** only if the grant pays. The objection to API pricing was
value-per-*your*-dollar. If it's grant money and the PI is fine with it, the same token cost is a
different decision. Worth one conversation with your advisor — not worth setting up on your own
dime.

### Free levers to pull first

1. ~~Check Penn State institutional access~~ — done, see above. AI Studio is not a fit; Bedrock is
   a grant conversation, not a free lever.
2. **Delegate mechanical work to Haiku subagents.** Model can be specified per-subagent even when
   it can't be switched in the main Cowork session. File sweeps, "find every use of this
   parameter," doc audits, bulk reformatting — none need Opus. Just say "use a haiku subagent."
3. **Batch sessions.** Cap burn is dominated by repeated context re-reads. One focused session
   costs far less than ten small ones that each re-read the same modules. The memory files help
   here; using them deliberately helps more.

### Recommendation

Stay on Pro through the fall. Pull the three free levers. Track *what you were doing* when you hit
a cap for two weeks — if it's mostly agentic, the answer is Max or nothing, and the Max question
becomes a grant question rather than a personal-budget one.

---

## Sources consulted

- Claude Code pricing and limits, 2026: <https://www.ssdnodes.com/blog/claude-code-pricing-in-2026-every-plan-explained-pro-max-api-teams/>
- Claude usage-limit change timeline: <https://explainx.ai/blog/claude-usage-limits-2026-timeline-explained>
- Multi-account policy and enforcement: <https://dev.to/vainamoinen/two-multi-account-claude-code-architectures-one-anthropic-accepts-one-they-ban-2om7>
