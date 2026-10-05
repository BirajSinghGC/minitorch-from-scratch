# Homework 2 - Minitorch Implementation

**COMPSCI 532**
**Due: 10/5/2026 11:59 PM EST**

## General Instructions

Work in groups of two for this assignment.

## Assignment

Implement elements of MiniTorch, a simplified teaching implementation of the Torch deep learning API (Torch is developed by Meta AI). Later homework uses PyTorch to *apply* the API; this one is about *implementing* it. Many parts are borrowed from the open-source [MiniTorch](https://minitorch.github.io/) assignment repository.

## Academic Integrity (unique to HW2)

- MiniTorch solutions are public. You **may** use them as reference for the **programming** sections, but every group member must understand both the syntax and the underlying concepts.
- The **analysis and conclusion report may NOT use outside resources** and must not be discussed outside your group.
- Questions about the policy go to course staff on Piazza.

## Learning Goals

- Installing and setting up a programming environment
- Using git version control
- Working in groups on software and written tasks
- Coding within an existing codebase
- Testing and debugging with unit tests
- Implementing machine learning operations
- Writing higher-order functions in functional Python
- Implementing a tree structure to support autodifferentiation

## Part 1: Environment Setup

Unit tests are expected to mostly **fail** after setup. They pass as you complete the tasks.

1. Install [VSCode](https://code.visualstudio.com/).
2. Install [Python 3.14.3](https://www.python.org/downloads/) (later versions will likely work).
3. Download the starter code from [Canvas](https://umamherst.instructure.com/courses/38708/files/17950401?wrap=1) and unzip it. Open the folder `umass-compsci-532-HW-MiniTorch` in VSCode.
4. Command Palette → **Python: Create Environment** → **Venv** → **Python 3.14** → check both `requirements.txt` and `requirements.extra.txt`. Wait for it to finish.
5. Verify: Terminal → New Terminal (prompt should show `(.venv)`), then:
   ```bash
   python            # should report 3.14
   >>> import minitorch   # no errors = installed correctly
   ```

## Tasks

| Task | File(s) | What to do | Test command |
|---|---|---|---|
| 1 — Operators | `minitorch/operators.py` | Complete `id`, `add`, `neg`, …, `inv_back`, `relu_back` (see `mul` example; TODOs mark the spots) | `pytest tests/test_operators.py -m task0_1` |
| 2 — Property tests | `tests/test_operators.py` | Implement each TODO'd unit/property test **following the given strategy** (hard-coded passing tests get no credit; pytest is required). Fix Task 1 code if these expose bugs, then re-run Task 1 tests. | `pytest tests/test_operators.py -m task0_2` |
| 3 — Functional programming | `minitorch/operators.py` (Task 0.3 TODO) | Implement `map`, `zipWith`, `reduce`, then use them with Task 1 operators. Notes: [functional](https://minitorch.github.io/module0/functional/) | `pytest tests/test_operators.py -m task0_3` |
| 4 — Module class | `minitorch/module.py` | Complete `train`, `eval`, `named_parameters`, `parameters`. Background: [modules](https://minitorch.github.io/module0/modules/) | `pytest tests/test_module.py -m task0_4` |
| 5 — Visualization | — | Manually tune a linear model in the GUI (below) | — |

**Task 4 note:** the expected passing result is **4 passed, 1 deselected, 1 xfailed**. `xfail` marks a test that is *expected* to fail ([pytest skipping/xfail docs](https://docs.pytest.org/en/stable/how-to/skipping.html)). Optional Module tree visualizer: `streamlit run app.py -- 0` → "Module Sandbox". This task is the hardest. Study the already-completed `Module` methods and the instance variables they use, and connect them to tree concepts.

**Task 5:** run `streamlit run project/app.py -- 0`, open page **"Module 0: Manual"**, scroll to **Model Parameters**, select the **Diag** dataset with **50 points**, and adjust the sliders until red X's sit mostly on red background and blue circles on blue. Close by closing the browser tab, then `Ctrl-C` in the terminal.

**Screenshots required:** terminal output of passing tests for Tasks 1–4, and the Task 5 screenshot showing **both the parameter sliders and the classification image**.

## Deliverables

1. **Report:** use the [HW2 Report Template](https://docs.google.com/document/d/17Gye9wV9E8hmfocSgivTgKCxiX98Rn5M_TBTr6rM2-s/edit?usp=sharing) (File → Make a Copy). Fill in each section, then export to PDF (File → Download → PDF). Submit to Gradescope **"HW2 - Minitorch Implementation"**.
2. **Code:** zip the project folder and submit to Gradescope **"HW2 - Project Code Zip Upload"**.
3. Submit **one joint submission** per group using the [Gradescope group tool](https://guides.gradescope.com/hc/en-us/articles/21863861823373-Adding-Group-Members-to-a-Submission). Add all members.

## Useful Docs

- [MiniTorch main page](https://minitorch.github.io/)
- [MiniTorch Module 0 (Fundamentals)](https://minitorch.github.io/module0/module0/). It differs from these instructions in places, so treat it as a secondary reference only.
- [pytest usage](https://docs.pytest.org/en/6.2.x/usage.html)
