# Changelog

Short, dated log of work across the VDFI workspace. One or two sentences per entry — a change doesn't have to mean a new function (that's what `guide.py` is for). Newest at top.

## 2026-07-09
- Ran `check_guide()`: renamed the `bootstrap_measurements` guide entry to `bootstrap_all` (new one-pass version that also folds in `bootstrap_stack_error`), and moved `bootstrap_measurements`, `bootstrap_stack_error`, `run_header` (measure), `resolve_core_sn_fields` (selection), and `resolve_galaxy_index` (stack) into `_BACKEND` as internal helpers.
- Set up `vdfi-workspace` as the main git repo/hub for all VDFI subprojects.
- Built out `01-lya_halos/` (notebooks + `functions/` package) plus shared `docs/`, `templates/`, `presentations/` at the workspace root; scaffolded `02-lya_escape/` and `03-lae_catalog/` for upcoming subprojects.
- Dropped in the real `functions/__init__.py` and `guide.py` from the working copy.
