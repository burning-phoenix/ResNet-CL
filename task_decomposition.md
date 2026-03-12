# Task Decomposition & Parallel Workflow

**Repo structure agreed upon before anyone writes code.**

---

## Repo Structure

```
project/
├── config.py                  # shared constants (everyone reads, Shashwat owns)
├── data/
│   └── cifar100.py            # dataset + loaders (Shashwat)
├── models/
│   ├── resnet18.py            # multi-head ResNet-18 (Yash)
│   └── logreg.py              # multi-head logistic regression (Shashwat)
├── training/
│   ├── trainer.py             # generic training loop (Emre)
│   ├── ewc.py                 # EWC Fisher + penalty (Himanshu)
│   └── l2_cl.py               # L2 CL penalty (Himanshu)
├── evaluation/
│   ├── metrics.py             # BWT, FWT, forgetting, F1 (Emre)
│   ├── embeddings.py          # extraction + silhouette (Yash)
│   └── visualize.py           # t-SNE plots, Fisher histograms (Yash)
├── experiments/
│   ├── run_trajectory.py      # main entry point, orchestrates everything (Emre)
│   └── configs/               # YAML/JSON per-experiment configs
├── results/                   # auto-populated by runs
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
└── tests/
    ├── test_data.py
    ├── test_model.py
    ├── test_ewc.py
    └── test_metrics.py
```

---

## Shared Contract: `config.py`

Everyone imports from this file. Shashwat writes it first. Nobody modifies it without posting in the group chat.

```python
# config.py — single source of truth

# Dataset
NUM_COARSE_CLASSES = 20
NUM_FINE_CLASSES = 100
INPUT_CHANNELS = 3
INPUT_SIZE = 32
FEATURE_DIM = 512          # ResNet-18 penultimate layer

# Training defaults
DEFAULT_LR = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS_PER_TASK = 50

# Fisher
DEFAULT_FISHER_SAMPLES = 2000

# Paths
CHECKPOINT_DIR = "results/checkpoints"
LOG_DIR = "results/logs"
FIGURE_DIR = "results/figures"
```

---

## Interface Contracts

These are the function signatures everyone must code against. Agree on these before writing implementations. If you need to change a signature, post in the group chat and get confirmation before pushing.

### Contract 1: Data Loader (Shashwat → everyone)

```python
def get_cifar100_loaders(batch_size=128, augment=True):
    """
    Returns:
        train_loader: yields (x, y_coarse, y_fine)
            x: Tensor (B, 3, 32, 32), normalized
            y_coarse: Tensor (B,), values in 0..2
            y_fine: Tensor (B,), values in 0..9
        test_loader: same format, no augmentation
    """
```

### Contract 2: Model (Yash → everyone)

```python
class MultiHeadResNet18(nn.Module):
    def forward(self, x, task_id):
        """
        Args:
            x: Tensor (B, 3, 32, 32)
            task_id: str, either "coarse" or "fine"
        Returns:
            logits: Tensor (B, 20) or (B, 100)
        """

    def get_backbone_params(self):
        """Returns iterator over shared backbone parameters only."""

    def get_features(self, x):
        """Returns 512-dim feature vectors (B, 512). For embedding extraction."""

    def freeze_head(self, task_id):
        """Zeros grad for the specified head."""

    def init_coarse_from_fine(self):
        """Aggregates fine-head weights into coarse head (for F→C transition)."""
```

### Contract 3: Fisher + Penalties (Himanshu → Emre's trainer)

```python
def compute_fisher(model, data_loader, n_samples, variant="empirical"):
    """
    Args:
        model: MultiHeadResNet18 (or MultiHeadLogReg)
        data_loader: yields (x, y_coarse, y_fine)
        n_samples: int
        variant: "empirical" | "sampled_true"
    Returns:
        fisher_dict: {param_name: Tensor of same shape as param}
    """

def ewc_penalty(model, fisher_dict, theta_star, lam):
    """
    Returns:
        scalar loss term: (λ/2) * Σ F_i * (θ_i - θ*_i)²
    """

def l2_cl_penalty(model, theta_star, lam):
    """
    Returns:
        scalar loss term: (λ/2) * Σ (θ_i - θ*_i)²
    NOTE: This is NOT weight decay. Do not use optimizer weight_decay.
    """

def diagnose_fisher(fisher_dict):
    """
    Prints/logs mean, max, fraction near zero.
    Returns: dict of diagnostics
    """
```

### Contract 4: Training Loop (Emre → experiments)

```python
def train_one_phase(model, loader, task_id, optimizer, epochs,
                    penalty_fn=None):
    """
    Args:
        model: any model following Contract 2
        loader: from Contract 1
        task_id: "coarse" or "fine"
        optimizer: torch optimizer
        epochs: int
        penalty_fn: callable(model) → scalar loss, or None
    Returns:
        training_log: list of {epoch, loss, acc}
    """
```

### Contract 5: Metrics (Emre → experiments)

```python
def evaluate(model, test_loader, task_id):
    """
    Returns:
        accuracy: float
        per_class_f1: dict {class_id: f1_score}
        macro_f1: float
    """

def compute_cl_metrics(R):
    """
    Args:
        R: 2x2 numpy array, R[i,j] = accuracy on task j after training task i
    Returns:
        dict: {bwt, fwt, forgetting}
    """
```

### Contract 6: Embeddings + Cluster Metrics (Yash → experiments)

```python
def extract_embeddings(model, test_loader):
    """
    Returns:
        embeddings: Tensor (N, 512)
        labels_coarse: Tensor (N,)
        labels_fine: Tensor (N,)
    """

def compute_cluster_metrics(embeddings, labels):
    """
    Args:
        embeddings: (N, 512), raw high-dim (NOT t-SNE projected)
        labels: (N,), superclass-level labels for clustering eval
    Returns:
        dict: {silhouette_score, davies_bouldin_index}
    """

def plot_tsne(embeddings, labels, title, save_path):
    """Saves a scatter plot to save_path. Fixed seed=42, perplexity=30."""
```

---

## Person-by-Person Breakdown

### Shashwat Krishna — Data + Logistic Regression + Paper

**Blocked by:** nothing (starts immediately)
**Blocks:** everyone (data loader is first dependency)

| Task | Priority | Deliverable | Est. Time |
|------|----------|-------------|-----------|
| S1: Write `config.py` | P0 — Day 1 morning | `config.py` committed | 15 min |
| S2: CIFAR-100 loader with dual labels | P0 — Day 1 morning | `data/cifar100.py` + `tests/test_data.py` | 1–2 hrs |
| S3: Verify coarse mapping | P0 — Day 1 | Unit test printing superclass→fine-class mappings | 30 min |
| S4: `MultiHeadLogReg` | P1 — Day 1 afternoon | `models/logreg.py` | 1 hr |
| S5: Run LogReg flat baseline | P1 — Day 2 | First numbers in `results/logs/` | 1 hr |
| S6: Run LogReg on both trajectories (unreg) | P2 — Day 2 | Sanity check that pipeline end-to-end works | 2 hrs |
| S7: Begin drafting paper (abstract, intro, methodology) | P1 — Day 3 | Overleaf/LaTeX draft | ongoing |
| S8: Integrate results into paper tables | P2 — Day 4–5 | Final paper | ongoing |

**Definition of done for S2:** Emre, Yash, and Himanshu can each independently run:
```python
from data.cifar100 import get_cifar100_loaders
train_loader, test_loader = get_cifar100_loaders()
x, y_c, y_f = next(iter(train_loader))
assert x.shape == (128, 3, 32, 32)
assert y_c.max() <= 1
assert y_f.max() <= 9
```

**Post in chat when ready for review:** "REVIEW REQUEST: S2, branch `shashwat/data-loader`. Data loader with dual labels. Run `python -m pytest tests/test_data.py`. Reviewer: Emre."

---

### Yash Vijay — Model Architecture + Representation Analysis

**Blocked by:** S2 (data loader) for integration testing; can write model code without it
**Blocks:** everyone needs the model to train

| Task | Priority | Deliverable | Est. Time |
|------|----------|-------------|-----------|
| Y1: `MultiHeadResNet18` skeleton | P0 — Day 1 | `models/resnet18.py` | 2 hrs |
| Y2: Test forward pass (both heads, feature extraction) | P0 — Day 1 | `tests/test_model.py` | 1 hr |
| Y3: `init_coarse_from_fine()` method | P1 — Day 1 | Tested weight aggregation | 1 hr |
| Y4: Embedding extraction function | P1 — Day 2 | `evaluation/embeddings.py` | 1 hr |
| Y5: Silhouette + Davies-Bouldin computation | P1 — Day 2 | In `evaluation/embeddings.py` | 1 hr |
| Y6: t-SNE visualization | P2 — Day 3 | `evaluation/visualize.py`, generates PNGs | 1–2 hrs |
| Y7: Run representation analysis on saved checkpoints | P2 — Day 4 | Figures in `results/figures/` | 2 hrs |

**Y1 details — what the model must do:**
- `torchvision.models.resnet18(pretrained=False)` as the backbone
- Strip the default `fc` layer
- Two `nn.Linear(512, K)` heads
- `forward(x, task_id)` routes to the correct head
- `get_features(x)` returns the 512-dim vector (use a forward hook or split the forward pass)
- `get_backbone_params()` returns only the shared parameters (everything except the two heads)
- `freeze_head(task_id)` sets `requires_grad=False` on the specified head

**Y3 details — weight aggregation:**
- Needs the CIFAR-100 superclass→fine-class mapping (get this from Shashwat's `data/cifar100.py`)
- For each superclass s: `coarse_head.weight[s] = mean(fine_head.weight[children_of_s])`
- Same for bias

**Post in chat when ready for review:** "REVIEW REQUEST: Y1+Y2, branch `yash/resnet-model`. Multi-head ResNet-18. Run `python -m pytest tests/test_model.py`. Reviewer: Himanshu." Include a 5-line usage example.

---

### Himanshu Janmeda — EWC + L2 + Fisher Analysis

**Blocked by:** Y1 (needs model interface to compute gradients); can write Fisher logic against the contract before Y1 lands
**Blocks:** Emre's full experiment runs

| Task | Priority | Deliverable | Est. Time |
|------|----------|-------------|-----------|
| H1: Empirical Fisher computation (variant A) | P0 — Day 1–2 | `training/ewc.py` | 2–3 hrs |
| H2: Sampled true Fisher computation (variant B) | P1 — Day 2 | Same file, `variant` parameter | 30 min (diff from H1 is one line) |
| H3: Fisher diagnostic logging | P1 — Day 2 | `diagnose_fisher()` in same file | 30 min |
| H4: `ewc_penalty()` function | P0 — Day 2 | In `training/ewc.py` | 30 min |
| H5: `l2_cl_penalty()` function | P0 — Day 2 | `training/l2_cl.py` | 30 min |
| H6: Unit test: Fisher on a toy model | P0 — Day 2 | `tests/test_ewc.py` | 1 hr |
| H7: Fisher magnitude analysis script | P2 — Day 3–4 | Histogram plots, layer-wise stats | 2 hrs |
| H8: λ grid search execution | P1 — Day 3 | Run experiments, log results | 3–4 hrs (wall time, mostly waiting) |

**H1 critical implementation detail:**
```
Process samples ONE AT A TIME in the Fisher loop.
Do NOT square the batch-aggregated gradient.
    Wrong: loss = mean_over_batch(); loss.backward(); grad**2
    Right: for each sample: loss_i.backward(); grad_i**2; accumulate
```

**H6 sanity check:** Compute Fisher on a 2-layer MLP with known weights. Verify that parameters with high curvature get high Fisher values. Also verify: `l2_cl_penalty` with `theta_star = current_params` returns 0.

**Post in chat when ready for review:** "REVIEW REQUEST: H1+H4+H5, branch `himanshu/ewc`. Fisher computation + EWC/L2 penalties. Run `python -m pytest tests/test_ewc.py`. Reviewer: Yash."

---

### Emre Alyamac — Training Loop + Metrics + Orchestration

**Blocked by:** S2 (data), Y1 (model) for integration; can write trainer and metrics against contracts immediately
**Blocks:** nothing downstream; produces final results

| Task | Priority | Deliverable | Est. Time |
|------|----------|-------------|-----------|
| E1: Generic `train_one_phase()` | P0 — Day 1–2 | `training/trainer.py` | 2 hrs |
| E2: `evaluate()` function (accuracy + F1) | P0 — Day 1–2 | `evaluation/metrics.py` | 1–2 hrs |
| E3: `compute_cl_metrics()` | P1 — Day 2 | Same file | 30 min |
| E4: Checkpoint save/load utilities | P1 — Day 2 | In `training/trainer.py` | 1 hr |
| E5: `run_trajectory.py` orchestrator | P0 — Day 2–3 | `experiments/run_trajectory.py` | 2–3 hrs |
| E6: Run unregularized baselines (both trajectories) | P1 — Day 3 | First real BWT numbers | 2 hrs wall |
| E7: Run EWC + L2 experiments | P1 — Day 3–4 | Full results grid | 4–6 hrs wall |
| E8: Run joint training reference | P1 — Day 3 | Single accuracy number | 1 hr wall |
| E9: Aggregate results into tables | P2 — Day 4 | CSV/JSON for Shashwat's paper | 1 hr |

**E1 details — the trainer must accept an optional penalty:**
```python
def train_one_phase(model, loader, task_id, optimizer, epochs, penalty_fn=None):
    for epoch in range(epochs):
        for x, y_coarse, y_fine in loader:
            y = y_coarse if task_id == "coarse" else y_fine
            logits = model(x, task_id)
            loss = F.cross_entropy(logits, y)
            if penalty_fn is not None:
                loss += penalty_fn(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**E5 details — the orchestrator ties everything together:**
```
run_trajectory.py --trajectory coarse_to_fine --condition ewc --lambda 400 --seed 42
```
It should:
1. Load data (Contract 1)
2. Init model (Contract 2)
3. Train T1 via `train_one_phase` (no penalty)
4. Evaluate after T1 → R[1,1], R[1,2]
5. Save θ*, compute Fisher (Contract 3)
6. Extract + save embeddings (Contract 6)
7. If Fine→Coarse: call `model.init_coarse_from_fine()`
8. Freeze old head
9. Build penalty_fn (ewc_penalty or l2_cl_penalty or None)
10. Train T2 via `train_one_phase` (with penalty)
11. Evaluate after T2 → R[2,1], R[2,2]
12. Extract + save embeddings again
13. Compute CL metrics (Contract 5)
14. Save everything to results/

**Post in chat when ready for review:** "REVIEW REQUEST: E1+E2+E5, branch `emre/trainer`. Trainer + metrics + orchestrator. Run `python -m pytest tests/test_metrics.py` and `python experiments/run_trajectory.py --trajectory coarse_to_fine --condition unreg --seed 42 --epochs 1` (smoke test with 1 epoch). Reviewer: Shashwat."

---

## Dependency Graph (what blocks what)

```
Day 1:

    Morning:
        Shashwat: S1 config.py ──→ S2 data loader
        Yash: Y1 model (write against contract, use dummy tensors to test shapes)
        Himanshu: H1 Fisher (write against contract, use a tiny nn.Linear as mock model)
        Emre: E1 trainer + E2 metrics (write against contracts)

    Afternoon — REVIEW WINDOW 1:
        Shashwat posts REVIEW REQUEST for S2 → Emre reviews
        Yash posts REVIEW REQUEST for Y1+Y2 → Himanshu reviews
        While waiting for review, authors continue on next tasks.

    End of Day 1 goal: S2 and Y1 both APPROVED and merged to main.

Day 2:

    Morning — Integration checkpoint:
        Everyone pulls main. Run: tests/test_data.py, tests/test_model.py
        If anything breaks, fix immediately.

        Shashwat: S4 LogReg model ──→ S5 first baseline run
        Yash: Y3 head init + Y4 embeddings + Y5 cluster metrics
        Himanshu: H2 true Fisher + H4 ewc_penalty + H5 l2_cl_penalty
        Emre: E4 checkpointing + E5 orchestrator

    Afternoon — REVIEW WINDOW 2:
        Himanshu posts REVIEW REQUEST for H1+H4+H5 → Yash reviews
        Emre posts REVIEW REQUEST for E1+E2+E5 → Shashwat reviews
        Yash posts REVIEW REQUEST for Y4+Y5 → Himanshu reviews

    End of Day 2 goal: all core modules APPROVED and merged.
        run_trajectory.py can execute an end-to-end dummy run.

Day 3:

    Morning — Experiments begin:
        Emre: E6 unreg baselines (both trajectories, 3 seeds)
        Shashwat: S7 start writing paper (methodology, math from proposal)

    Afternoon:
        Emre: E7 EWC + L2 runs begin (start with best-guess λ)
        Himanshu: H7 Fisher magnitude analysis (on checkpoints from E6)
        Yash: Y6 t-SNE code (REVIEW REQUEST → Himanshu reviews)

    End of Day 3 goal: unregularized BWT numbers for both trajectories.

Day 4:

    Emre: E7 continues ──→ E8 joint reference ──→ E9 aggregate results
    Yash: Y7 run representation analysis on saved checkpoints
    Himanshu: H8 λ grid search (if E7 is still running, help run seeds in parallel)
    Shashwat: S8 integrate results into paper tables and figures

    End of Day 4 goal: all numbers that will appear in the paper are computed.

Day 5:

    Everyone: review paper draft, fix figures, check tables, submit.
    No new code unless something is broken.
    Emergency merge override allowed today only (verbal confirmation on a call).
```

---

## Communication Protocol

**Channel: Team group chat (Slack/Discord/Teams)**

**Required posts (copy-paste these templates):**

When you merge something:
```
MERGED: [task ID] [file path]
Reviewed by: [reviewer name]
Status: ready for integration
Usage: [2-line code example]
```

When you request a review:
```
REVIEW REQUEST: [task ID] [branch name]
What it does: [1 sentence]
How to test: [command to run tests]
Files changed: [list]
Reviewer: [assigned reviewer from table]
```

When you finish reviewing:
```
APPROVED: [task ID]
Tests passed: [author's tests + N edge cases I added]
Edge case tests added to: tests/test_<module>_edge.py
Notes: [any observations, minor suggestions, or things to watch out for]
```
or
```
CHANGES REQUESTED: [task ID]
Issue: [what's wrong]
Repro: [command or steps to trigger the bug]
Severity: [blocks merge / minor, can fix after]
```

When you're blocked:
```
BLOCKED: I need [task ID] from [person] to continue [my task ID]
Can work on [alternative task] in the meantime
```

When you find a bug in someone else's code:
```
BUG: [file path] [line number or function name]
Expected: [what should happen]
Actual: [what happens]
Repro: [command to reproduce]
```

When you change a shared interface:
```
INTERFACE CHANGE: [function name] in [file]
Old: [old signature]
New: [new signature]
Reason: [why]
Who's affected: [names]
```
Wait for confirmation from affected people before merging.

---

## Git Workflow

- `main` branch: always working. Never push broken code here.
- Feature branches: `shashwat/data-loader`, `yash/resnet-model`, `himanshu/ewc`, `emre/trainer`
- If two people need to touch the same file, coordinate in chat first.
- Commit messages: `[TASK_ID] description` (e.g., `[S2] CIFAR-100 loader with dual labels`)

### Mandatory Cross-Review Rule

**No code merges to `main` unless it has been reviewed and tested by someone who did not write it.** No exceptions, even under time pressure. A bug that reaches `main` costs everyone time; catching it in review costs one person 20 minutes.

The process for every merge:

```
1. AUTHOR finishes code on their feature branch.
2. AUTHOR runs their own unit tests, confirms they pass.
3. AUTHOR posts in chat:
       REVIEW REQUEST: [task ID] [branch name]
       What it does: [1 sentence]
       How to test: [command to run tests]
       Files changed: [list]
4. REVIEWER (assigned below) pulls the branch.
5. REVIEWER runs the author's unit tests.
6. REVIEWER writes at least 2 additional edge-case tests and runs them.
       (See "What the reviewer must check" below.)
7. REVIEWER posts one of:
       APPROVED: [task ID] — tested, works. [any notes]
       CHANGES REQUESTED: [task ID] — [what's wrong, how to repro]
8. Only after APPROVED: author merges to main and posts the MERGED message.
```

**Assigned reviewers (fixed pairings to avoid confusion):**

| Author | Reviewer | Rationale |
|--------|----------|-----------|
| Shashwat (data, logreg) | Emre | Emre consumes the data loader directly in the trainer; he'll catch format issues fast |
| Yash (model, embeddings) | Himanshu | Himanshu needs the model to compute Fisher; he'll catch interface mismatches |
| Himanshu (EWC, L2, Fisher) | Yash | Yash understands the architecture and can verify gradients flow correctly through the backbone |
| Emre (trainer, metrics, orchestrator) | Shashwat | Shashwat is paper lead and needs the results pipeline to work correctly for reporting |

**What the reviewer must check:**

The reviewer is not just running the author's tests. The reviewer is actively trying to break the code. Specifically:

*Normal operation checks:*
- Run the author's existing unit tests. They must all pass.
- Read the code and verify it matches the interface contract from this document.
- Run a quick integration check: can you import the module and call it from your own code?

*Edge case stress tests (reviewer writes these):*
- **Data:** What happens with batch_size=1? What about the last incomplete batch? Feed in a single sample and verify shapes.
- **Model:** Pass a zero tensor. Pass a very large tensor. Call `forward` with the wrong `task_id` string — does it fail cleanly with a clear error, or silently return garbage?
- **Fisher:** What if `n_samples` is larger than the dataset? What if the model is randomly initialized (not trained) — does Fisher still compute without NaN/Inf?
- **Penalties:** What if λ=0? What if λ is extremely large (1e8)? Does the penalty blow up or stay numerically stable?
- **Trainer:** What if epochs=0? What if the loader is empty? Does it handle keyboard interrupt gracefully (so you don't lose a half-finished run)?
- **Metrics:** What if accuracy is 0 on all classes (degenerate model)? Does F1 handle division by zero?

*The reviewer should add their edge-case tests to `tests/` with the naming convention `test_<module>_edge.py` so they're preserved for future runs.*

**Time budget:** A review should take 20–40 minutes. If it's taking longer, the code probably needs to be refactored. Tell the author.

**Emergency override (Day 5 only):** On the final day, if a merge is blocking paper submission, the author can merge with only a verbal confirmation over a call. But this should never happen if the schedule is followed.

---

## Testing Checklist

These are the minimum checks. The **author** must verify all items in their section before requesting review. The **reviewer** must independently re-verify all items plus add edge cases.

**Status tracking:** Copy this checklist into your PR description. Mark items with `[x]` as you verify them. The reviewer marks their own column.

### Shashwat — data (Reviewer: Emre)

| Check | Author | Reviewer |
|-------|--------|----------|
| Batch shape is (B, 3, 32, 32), (B,), (B,) | [ ] | [ ] |
| y_coarse values ∈ {0..1}, y_fine values ∈ {0..9} | [ ] | [ ] |
| Every fine class maps to exactly one coarse class | [ ] | [ ] |
| Augmentation applies only to train, not test | [ ] | [ ] |
| Train and test sets have the expected sample counts (50k / 10k) | [ ] | [ ] |

*Reviewer edge cases to add:*
- [ ] batch_size=1 produces correct shapes
- [ ] Last incomplete batch doesn't crash
- [ ] Same image appears in both loaders with same coarse/fine labels
- [ ] Normalization values are correct (CIFAR-100 mean/std, not CIFAR-10 or ImageNet)

### Yash — model (Reviewer: Himanshu)

| Check | Author | Reviewer |
|-------|--------|----------|
| `forward(x, "coarse")` returns (B, 20) | [ ] | [ ] |
| `forward(x, "fine")` returns (B, 100) | [ ] | [ ] |
| `get_features(x)` returns (B, 512) | [ ] | [ ] |
| `get_backbone_params()` does not include head parameters | [ ] | [ ] |
| `freeze_head("coarse")` sets `requires_grad=False` on coarse head | [ ] | [ ] |
| `init_coarse_from_fine()` produces mean of child fine weights | [ ] | [ ] |
| After `init_coarse_from_fine()`, coarse head accuracy > random on a trained model | [ ] | [ ] |

*Reviewer edge cases to add:*
- [ ] `forward(x, "invalid")` raises a clear error, not a silent wrong output
- [ ] `get_backbone_params()` count matches expected ResNet-18 param count minus two head layers
- [ ] `freeze_head` then `unfreeze` (if implemented) restores grad correctly
- [ ] Zero input tensor doesn't produce NaN in features or logits
- [ ] `get_features()` output is detachable and moveable to CPU for embedding storage

### Himanshu — EWC/L2 (Reviewer: Yash)

| Check | Author | Reviewer |
|-------|--------|----------|
| `compute_fisher()` returns dict with same keys as `model.get_backbone_params()` | [ ] | [ ] |
| Fisher values are all non-negative | [ ] | [ ] |
| `variant="empirical"` and `variant="sampled_true"` produce different values | [ ] | [ ] |
| `ewc_penalty()` returns 0 when θ == θ* | [ ] | [ ] |
| `l2_cl_penalty()` returns 0 when θ == θ* | [ ] | [ ] |
| `l2_cl_penalty()` is NOT using `optimizer.weight_decay` | [ ] | [ ] |
| Fisher is computed per-sample (batch_size=1), not on aggregated gradients | [ ] | [ ] |
| `diagnose_fisher()` logs mean, max, and near-zero fraction | [ ] | [ ] |

*Reviewer edge cases to add:*
- [ ] Fisher with `n_samples` > dataset size doesn't crash (should just use all data)
- [ ] Fisher on a randomly initialized (untrained) model doesn't produce NaN/Inf
- [ ] `ewc_penalty()` with λ=0 returns exactly 0.0
- [ ] `ewc_penalty()` with λ=1e8 doesn't overflow to Inf
- [ ] Fisher dict is on the correct device (CPU/GPU matches model)
- [ ] Verify per-sample computation: manually compute Fisher for 3 samples, compare against function output

### Emre — trainer/metrics (Reviewer: Shashwat)

| Check | Author | Reviewer |
|-------|--------|----------|
| `train_one_phase()` works with `penalty_fn=None` | [ ] | [ ] |
| `train_one_phase()` works with a penalty_fn returning a scalar tensor | [ ] | [ ] |
| `evaluate()` returns accuracy as float in [0, 1] | [ ] | [ ] |
| `compute_cl_metrics()` returns negative BWT when R[2,1] < R[1,1] | [ ] | [ ] |
| `run_trajectory.py` saves checkpoints, logs, and metrics to correct dirs | [ ] | [ ] |
| Results JSON contains: trajectory, condition, λ, seed, R matrix, BWT, FWT, forgetting, macro_F1 | [ ] | [ ] |

*Reviewer edge cases to add:*
- [ ] `train_one_phase()` with epochs=0 returns empty log, doesn't crash
- [ ] `evaluate()` on a random model returns ~chance accuracy (sanity check)
- [ ] `evaluate()` handles division-by-zero in F1 when a class has 0 predictions
- [ ] `run_trajectory.py` with an invalid `--trajectory` flag fails with a clear error message
- [ ] Checkpoint can be loaded and produces identical `evaluate()` output as when it was saved
- [ ] Results JSON is valid JSON (parseable by `json.load`)
