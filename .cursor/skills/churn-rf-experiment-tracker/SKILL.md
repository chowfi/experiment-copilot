---
name: churn-rf-experiment-tracker
description: >-
  Runs customer_churn_random_forest.py with uv, reads customer_churn_random_forest.log,
  diffs or summarizes the script, writes Intent as change-plus-motivation using chat, prior
  EXPERIMENTS rows, and train/test metrics (Acc, AUC, weighted P/R/F1), appends one row to EXPERIMENTS.md,
  and echoes that row in chat. Use for churn RF experiments, tuning, logs, or tracking.
---

# Churn RF experiment tracker

Single-agent workflow: follow these steps in order in the **same** chat (no subagents unless the user explicitly asks).

## When to use

Apply when the user wants to **execute** the churn script and **record** a run, or when they reference this workflow, the log, or post-run analysis.

## Prerequisites

- Workspace is the repo root containing `customer_churn_random_forest.py`.
- Training and test CSVs exist under `./data/` (or `DATA_DIR` if the user overrides it).
- Dependencies: `uv sync` then run via `uv run` (see Run step).

## Steps

1. **Run the script** from the repository root:
   - Default: `uv run python customer_churn_random_forest.py`
   - Honor user-requested env (e.g. `LOG_FILE`, `LOG_LEVEL`, `DATA_DIR`) by exporting or prefixing the command.
2. **Capture run outcome**: exit code; stderr if useful.
3. **Locate the log file**: path from the latest `Log file: ...` line in output, else default `customer_churn_random_forest.log` beside the script.
4. **Read the log**; isolate the **latest run** (see Parsing the log).
5. **Code changes**: `git diff HEAD -- customer_churn_random_forest.py` (or ref the user names). If untracked or empty diff, one **Action** cell summarizing current tunables (`RF_PARAMS`, `RF_FIXED`, paths, preprocessor) as the “change” for this baseline.
6. **Intent cell**: write per **Intent cell (experiment hypothesis)** below (after reading **EXPERIMENTS.md**’s previous row when `Run#` > 1).
7. **Recommendation**: **one** short phrase for the Recommendation cell, grounded in metrics (train vs test gap, class balance in the log), and the commented `GridSearchCV` block (~223–241).
8. **Output (concise)**:
   - In chat, paste **only** the new markdown **table row** you appended (the `| … |` line) plus one line: `EXPERIMENTS.md updated.` No long prose.
9. **Journal (required)**:
   - Update **`EXPERIMENTS.md`** at the **repository root** using the **table format** below.
   - **If the file is missing or has no table**: create `# Churn RF experiments`, then the **header row**, **separator row**, then the first **data row** for `Run#1`.
   - **If the table exists**: append **exactly one new data row** after the last data row. **Do not** duplicate the header or title. **Do not** edit prior rows.
   - **Run#**: integer in the first column = one plus the largest existing `Run#` in the table (if none, `1`).
   - **Migrating legacy tables**:
     - If a **CM** column exists (old skill), replace header + separator with the current schema and drop CM from each data row (pad or delete that cell only).
     - If the table has **no PRF1** column yet, replace the **header** and **separator** with the current set in **Journal format**, then insert **`train: n/a; test: n/a`** in the new **PRF1** position on **every existing data row** so pipe counts match (do not re-run old experiments to backfill unless the user asks).
   - Before finalizing **Intent**, read **`EXPERIMENTS.md`** (if it exists): use the **last row’s** Action, Acc/AUC, and Recommendation as “what came before” when this run is not the first.

### Failed runs

Append one table row: put exit code / error snippet in **Action** or **Acc**/**AUC** cells as `train: —; test: —` or `n/a`; **PRF1** as `train: n/a; test: n/a` if no report; **Recommendation** = fix-first.

## Parsing the log

Append-only log: **latest run** = from the **last** `Log file:` line through that run’s metrics and `classification_report` blocks for **both** `[train]` and `[test]`, or through the error. If there is no `Log file:` in the tail, start at the **last** `Experiment:` line.

From that block, read:

- `Results [train]: accuracy=…` and `Results [train]: roc_auc=…` (or `roc_auc skipped` → put `train AUC: n/a` in the AUC cell).
- `Results [test]: accuracy=…` and `Results [test]: roc_auc=…` (same).
- After each `Results [split]: classification_report` line, parse the following sklearn table: take the **weighted avg** row’s **precision**, **recall**, and **f1-score** for that split (`train` / `test`). If **weighted avg** is missing, use **macro avg**. Use the numeric values as printed (typically 2 decimals).

**Legacy logs** (only `Results:` without `[train]`/`[test]`): treat those values as **test** only; use `train: n/a` in Acc, AUC, and PRF1 unless train lines appear elsewhere in the same run.

**Logs without classification_report** (e.g. failure before metrics): set **PRF1** to `train: n/a; test: n/a`.

## Intent cell (experiment hypothesis)

The **Intent** column must read like a **short experiment rationale**: what you are trying this run **and why**, tied to **evidence or stated goals**—not a description of invoking this skill or filling the journal.

**Required shape (one sentence, one table cell):**

- **Change + motivation**, e.g. *“Reduced `n_estimators` 200→100 because train AUC stayed ~1.0 while test AUC lagged, suggesting overfitting.”*
- Or *“Set `class_weight='balanced'` after prior run predicted almost all positives on test despite high train scores.”*

**How to build it (order of priority):**

1. **Cursor chat**: If the user said why they changed something, **lead with that** (paraphrase tightly; keep their causal wording when it is clear).
2. **Diff / Action vs previous run**: Name the concrete change (hyperparameter, preprocessor, data path) relative to **the prior `EXPERIMENTS.md` row** or the previous logged `RF_PARAMS` if inferable.
3. **This run’s metrics vs last run’s Acc/AUC cells**: Use **train vs test** gap, collapse on test, or flat test metrics to justify the move (e.g. overfitting, underfitting, no improvement).
4. **Prior Recommendation column**: If the last row’s recommendation suggested a specific next step and this run implements it, Intent may reference that (*“Applied prior suggestion to tune threshold / add `class_weight` / run smaller forest.”*).

**Run#1 / no prior row:** Use a baseline intent, e.g. *“Establish baseline RF pipeline and metrics on the fixed train/test split.”* Do **not** mention the skill or `EXPERIMENTS.md`.

**Anti-patterns (never use as Intent):**

- Meta text: *“Invoked `/churn-rf-experiment-tracker`”*, *“journal run”*, *“record in this file”*.
- Vague placeholders: *“Inferred: user wanted to experiment”* without **what** and **why**.
- Duplicating the **Action** cell verbatim; Intent must add **motivation** or **hypothesis**, not only the edit list.

If chat is silent and motivation is entirely from metrics/diff, start with **Inferred:** then the same change+motivation pattern in one clause.

## Journal format (markdown table)

**Columns (fixed order — no CM column):**

`| Run# | Timestamp | Action | Acc | AUC | PRF1 | Intent | Recommendation |`

**Separator row (copy exactly when creating the table):**

`|-----:|-----------|--------|-----|-----|------|--------|------------------|`

**Acc** and **AUC** cells must include **both** training and test numbers parsed from the latest run, in this exact shape (compact, one line per cell):

- **Acc:** `train: <float>; test: <float>` (use `n/a` if missing for that split).
- **AUC:** `train: <float>; test: <float>` (use `n/a` if skipped/missing).

Example: `train: 1.000000; test: 0.497313` in **Acc**, and `train: 1.000000; test: 0.610912` in **AUC`.

**PRF1** cell — **weighted** precision / recall / F1 from each split’s `classification_report` (see Parsing the log). One line, no raw report dump:

- Format: `train: P=<p> R=<r> F1=<f1>; test: P=<p> R=<r> F1=<f1>` (match log precision; use `n/a` for a split if missing).
- Example: `train: P=0.99 R=0.99 F1=0.99; test: P=0.75 R=0.47 F1=0.31`

**Data row template:**

```markdown
| N | YYYY-MM-DD HH:MM:SS | …one line… | train: …; test: … | train: …; test: … | train: P=… R=… F1=…; test: P=… R=… F1=… | … | … |
```

**Rules:**

- Keep **Action**, **Intent**, **PRF1**, and **Recommendation** on **one table row** each (no embedded newlines). Trim wording; use backticks for code tokens.
- **Intent** must follow **Intent cell (experiment hypothesis)** above.
- Do not place `---` horizontal rules between runs; table rows are the history.

## Report template (for agents; output must stay minimal)

Append the row to `EXPERIMENTS.md`, then in chat output **only** that same `| … |` line plus `EXPERIMENTS.md updated.`
