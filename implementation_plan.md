# Implementation Plan: Hierarchical Task Granularity in Continual Learning

**Companion documents:**
- **Task Decomposition & Parallel Workflow** (`task_decomposition.md`): repo structure, interface contracts, per-person task breakdown with priorities and deadlines, day-by-day schedule, and the **mandatory cross-review rule** (no code merges to `main` without review and edge-case stress testing by a second person — fixed reviewer pairings and full testing checklists are in that document). All workflow, process, and communication protocol questions are answered there. We will use MS teams for communication, and to update each other on progress.
- **This document** covers the *what* and *why*: algorithms, pseudocode, mathematical setup, and experimental design. The task decomposition covers the *who*, *when*, and *how*.

---

## 1. Dataset Pipeline

**Dataset:** CIFAR-100 (60,000 images, 100 fine classes, 20 superclasses)

The dataset provides a native two-level hierarchy. Every image carries both a fine label (0–99) and a coarse label (0–19). No manual mapping is needed.

```
LOAD_DATASET():
    train_data, test_data ← torchvision.CIFAR100(download=True)

    For each sample (x, y_fine):
        y_coarse ← CIFAR100_COARSE_MAP[y_fine]
        store (x, y_coarse, y_fine)

    Apply standard augmentation to training set:
        - 4-pixel zero-padding
        - Random 32×32 crop
        - Random horizontal flip
        - Normalize per-channel (CIFAR-100 mean/std)

    Return train_loader, test_loader
        where each batch yields (x, y_coarse, y_fine)
```

**Note on the coarse map:** CIFAR-100's `torchvision` loader exposes `targets` (fine) and `coarse_targets` is not directly available. You must either load from the raw pickle (which contains `coarse_labels`) or define the 100→20 mapping manually from the CIFAR-100 documentation.

> **Implementation:** See Contract 1 in `task_decomposition.md` for the exact function signature (`get_cifar100_loaders`) that Shashwat must implement and Emre must review.

---

## 2. Model Architecture

### 2.1 ResNet-18 with Multi-Head Output

The backbone is a standard ResNet-18 pretrained weights are NOT used (train from scratch). The final fully-connected layer is removed and replaced by two independent classification heads.

```
CLASS MultiHeadResNet18:

    INIT():
        backbone ← ResNet18(num_classes=None)   # Remove default FC
        feature_dim ← 512                        # ResNet-18 penultimate dim

        head_coarse ← Linear(feature_dim, 20)
        head_fine   ← Linear(feature_dim, 100)

    FORWARD(x, task_id):
        features ← backbone(x)                   # shared feature extractor

        IF task_id == "coarse":
            RETURN head_coarse(features)
        ELSE IF task_id == "fine":
            RETURN head_fine(features)
```

**Design decisions:**
- Both heads exist from initialization. Only the active head receives gradients during a given task.
- The inactive head is frozen (zero grad) during training.
- `features` (the 512-dim vector before the heads) is what gets saved for t-SNE and silhouette analysis.

> **Implementation:** See Contract 2 in `task_decomposition.md` for the exact method signatures (`forward`, `get_backbone_params`, `get_features`, `freeze_head`, `init_coarse_from_fine`) that Yash must implement and Himanshu must review.

### 2.2 Head Initialization at Transition (Fine-to-Coarse only)

When transitioning from fine (100-class) to coarse (20-class), randomly initializing the coarse head discards the deterministic mapping between fine and coarse labels. Instead, aggregate the fine-class weights.

```
INITIALIZE_COARSE_FROM_FINE(head_fine, head_coarse):
    For each superclass s in 0..19:
        child_indices ← list of fine-class indices belonging to superclass s
        head_coarse.weight[s] ← MEAN(head_fine.weight[child_indices], dim=0)
        head_coarse.bias[s]   ← MEAN(head_fine.bias[child_indices])
```

For Coarse-to-Fine, the fine head is initialized randomly (no meaningful aggregation exists in that direction).

### 2.3 Multinomial Logistic Regression Baseline

A simple linear model operating on flattened 32×32×3 = 3072-dim input vectors. Same multi-head structure.

```
CLASS MultiHeadLogReg:

    INIT():
        W_coarse ← Matrix(3072, 20), init zeros
        W_fine   ← Matrix(3072, 100), init zeros

    FORWARD(x, task_id):
        x_flat ← flatten(x)    # (batch, 3072)

        IF task_id == "coarse":
            logits ← x_flat @ W_coarse
        ELSE:
            logits ← x_flat @ W_fine

        RETURN logits    # raw logits, same as ResNet — CrossEntropy applies softmax internally
```

---

## 3. Training Protocol

> **Implementation:** See Contracts 3–4 in `task_decomposition.md` for the exact function signatures (`compute_fisher`, `ewc_penalty`, `l2_cl_penalty`, `train_one_phase`) and the `run_trajectory.py` orchestrator logic (Emre's E5).

### 3.1 Shared Hyperparameters (all trajectories)

```
HYPERPARAMETERS:
    optimizer       = SGD
    learning_rate   = 0.01
    momentum        = 0.9
    weight_decay    = 0       # (except for L2 baseline)
    batch_size      = 128
    epochs_per_task = 50      # may need tuning
    λ_EWC           = [100, 400, 1000]   # grid search
    λ_L2            = [0.01, 0.1, 1.0]   # grid search
    fisher_samples  = 2000
    fisher_variant  = "empirical"  # or "sampled_true" or "exact_true"
                                   # start with empirical; add sampled_true if time permits
```

### 3.2 Learning Trajectories

All trajectories use the same data, same backbone, same optimizer. The only variable is the task ordering and the regularization method.

```
TRAJECTORY: COARSE-TO-FINE
    T1: Train on Y_coarse (20 classes) for E epochs
        → active head: head_coarse
        → compute Fisher / save θ* at end of T1
    T2: Train on Y_fine (100 classes) for E epochs
        → active head: head_fine (randomly initialized)
        → apply EWC penalty anchored to θ* from T1

TRAJECTORY: FINE-TO-COARSE
    T1: Train on Y_fine (100 classes) for E epochs
        → active head: head_fine
        → compute Fisher / save θ* at end of T1
    T2: Train on Y_coarse (20 classes) for E epochs
        → active head: head_coarse (initialized via weight aggregation)
        → apply EWC penalty anchored to θ* from T1

TRAJECTORY: JOINT TRAINING REFERENCE
    Single task: Train on Y_fine (100 classes) for 2E epochs
        → standard cross-entropy, no regularization
        → no sequential transition (i.i.d. baseline)
        → BWT/forgetting metrics are undefined here
```

### 3.3 Continual Learning Algorithms (applied to each trajectory)

Each trajectory is run under three regularization conditions:

**Condition A — Unregularized Sequential Fine-Tuning (lower bound)**

```
TRAIN_UNREGULARIZED(model, T1_loader, T2_loader):
    # Phase 1
    For epoch in 1..E:
        For (x, y) in T1_loader:
            loss ← CrossEntropy(model(x, task=T1), y)
            loss.backward()
            optimizer.step()

    # Phase 2 — no protection of T1 weights
    For epoch in 1..E:
        For (x, y) in T2_loader:
            loss ← CrossEntropy(model(x, task=T2), y)
            loss.backward()
            optimizer.step()
```

**Condition B — L2 Regularization (strong baseline, per Hsu et al. 2018)**

```
TRAIN_L2(model, T1_loader, T2_loader, λ_L2):
    # Phase 1: identical to unregularized
    Train on T1 as above
    θ_star ← copy(model.backbone.parameters())

    # Phase 2: penalize deviation from T1 params (identity matrix replaces Fisher)
    For epoch in 1..E:
        For (x, y) in T2_loader:
            loss_new ← CrossEntropy(model(x, task=T2), y)
            loss_reg ← (λ_L2 / 2) * SUM( (θ_i - θ_star_i)^2 )
            loss ← loss_new + loss_reg
            loss.backward()
            optimizer.step()
```

**Condition C — Elastic Weight Consolidation (EWC)**

```
TRAIN_EWC(model, T1_loader, T2_loader, λ_EWC, n_fisher_samples, fisher_variant):
    # Phase 1
    Train on T1 as above
    θ_star ← copy(model.backbone.parameters())

    # Compute Fisher on T1 data (see Section 4 for both variants)
    F ← COMPUTE_FISHER(model, T1_loader, n_fisher_samples, variant=fisher_variant)

    # Phase 2: Fisher-weighted penalty
    For epoch in 1..E:
        For (x, y) in T2_loader:
            loss_new ← CrossEntropy(model(x, task=T2), y)
            loss_ewc ← (λ_EWC / 2) * SUM( F_i * (θ_i - θ_star_i)^2 )
            loss ← loss_new + loss_ewc
            loss.backward()
            optimizer.step()
```

---

## 4. Fisher Information Computation

> **Implementation:** See Contract 3 in `task_decomposition.md` for `compute_fisher()`, `ewc_penalty()`, `l2_cl_penalty()`, and `diagnose_fisher()` signatures. Himanshu implements; Yash reviews. The per-sample gradient requirement and the L2-vs-weight-decay distinction are flagged in Himanshu's testing checklist.

This is applied only to the shared backbone parameters (not the classification heads, since heads are task-specific and frozen).

**Background:** The original EWC paper (Kirkpatrick et al., 2017) does not specify which variant of the Fisher to compute. In the optimization literature, Kunstner et al. (2019) argue against the empirical Fisher. In the continual learning literature, van de Ven (2025) found that the exact/true Fisher performs substantially better, but the gap between the sampled true Fisher and the empirical Fisher is small. Most open-source EWC implementations and major CL libraries (Avalanche, PyCIL) default to the empirical Fisher or even cruder batch-level approximations.

**Recommendation:** Implement both variants. The code difference is a single line (the label source). Report results for both if time permits; if not, use the empirical Fisher and state this explicitly.

### 4.1 Variant A — Empirical Fisher (ground-truth labels)

Uses the actual dataset labels. Simpler, widely used, but technically a different mathematical object than the true Fisher.

```
COMPUTE_EMPIRICAL_FISHER(model, data_loader, n_samples):
    F ← zeros(num_backbone_params)
    count ← 0

    model.eval()
    For (x, y_true) in data_loader:
        IF count >= n_samples: BREAK

        # CRITICAL: process one sample at a time (batch_size=1)
        # PyTorch aggregates gradients across a batch, so squaring
        # the aggregated gradient ≠ averaging the squared per-sample gradients.
        For each sample (x_i, y_i) in the batch:
            logits ← model(x_i, task=current_task)
            log_probs ← log_softmax(logits)

            loss ← NLLLoss(log_probs, y_i)       # ← uses TRUE label

            model.zero_grad()
            loss.backward()

            For each param p in model.backbone.parameters():
                F[p] += p.grad.data ** 2

            count += 1

    F ← F / count
    RETURN F
```

### 4.2 Variant B — Sampled True Fisher (model's own distribution)

Samples labels from the model's predicted distribution rather than using ground truth. This is the theoretically correct Fisher information as defined in statistics: the expectation of the squared score under the model's own output distribution.

```
COMPUTE_TRUE_FISHER_SAMPLED(model, data_loader, n_samples):
    F ← zeros(num_backbone_params)
    count ← 0

    model.eval()
    For (x, _) in data_loader:                    # ground-truth labels are IGNORED
        IF count >= n_samples: BREAK

        For each sample x_i in the batch:
            logits ← model(x_i, task=current_task)
            probs ← softmax(logits)
            log_probs ← log_softmax(logits)

            # Sample a label from the model's own predicted distribution
            y_sampled ← Categorical(probs).sample()

            loss ← NLLLoss(log_probs, y_sampled)  # ← uses SAMPLED label

            model.zero_grad()
            loss.backward()

            For each param p in model.backbone.parameters():
                F[p] += p.grad.data ** 2

            count += 1

    F ← F / count
    RETURN F
```

### 4.3 Variant C — Exact True Fisher (full expectation, optional)

Instead of sampling one label, sum over all output classes weighted by their predicted probability. More expensive (K forward-backward passes per sample, where K = number of classes) but gives the exact diagonal of the true Fisher without sampling noise. Only feasible for the 20-class coarse task; likely too expensive for the 100-class fine task.

```
COMPUTE_TRUE_FISHER_EXACT(model, data_loader, n_samples):
    F ← zeros(num_backbone_params)
    count ← 0
    K ← number of output classes for current task

    model.eval()
    For (x, _) in data_loader:
        IF count >= n_samples: BREAK

        For each sample x_i in the batch:
            logits ← model(x_i, task=current_task)
            probs ← softmax(logits)
            log_probs ← log_softmax(logits)

            grad_outer_sum ← zeros(num_backbone_params)

            For each class k in 0..K-1:
                loss_k ← -log_probs[k]            # NLL for class k

                model.zero_grad()
                loss_k.backward(retain_graph=True)

                For each param p in model.backbone.parameters():
                    grad_outer_sum[p] += probs[k] * (p.grad.data ** 2)

            F += grad_outer_sum
            count += 1

    F ← F / count
    RETURN F
```

### 4.4 Diagnostic: Gradient Norm Monitoring

When the model has fully converged on T1, gradients near the minimum are very small, making all Fisher values tiny. In this regime, EWC's penalty term effectively vanishes regardless of λ, and the algorithm provides no protection against forgetting.

```
DIAGNOSE_FISHER(F):
    mean_F ← mean(F)
    max_F  ← max(F)
    frac_near_zero ← fraction of F_i < 1e-8

    LOG("Fisher mean: {mean_F}, max: {max_F}, near-zero: {frac_near_zero}")

    IF frac_near_zero > 0.95:
        WARNING: "Fisher is near-degenerate. EWC penalty will be ineffective."
        SUGGESTION: "Consider computing Fisher a few epochs before full
                     convergence, or normalizing F by its mean/max."
```

**Practical mitigation options if Fisher is degenerate:**
1. Compute Fisher 5-10 epochs before T1 convergence (gradients are still informative).
2. Normalize: replace F with F / mean(F), so the relative importance structure is preserved even if absolute magnitudes are small.
3. Increase λ_EWC substantially (but this risks over-constraining T2 learning).

### 4.5 L2 vs. EWC vs. Weight Decay — Implementation Warning

These three things look similar but are mathematically distinct. Ensure the codebase keeps them separate:

```
# STANDARD WEIGHT DECAY (pulls toward zero — NOT what you want for CL)
# This is what PyTorch's optimizer weight_decay parameter does.
loss_wd = (λ / 2) * SUM( θ_i^2 )
# DO NOT use optimizer = SGD(..., weight_decay=λ) for the L2 CL baseline.

# L2 CL BASELINE (pulls toward θ* from T1 — identity matrix replaces Fisher)
loss_l2 = (λ / 2) * SUM( (θ_i - θ_star_i)^2 )

# EWC (pulls toward θ* from T1, weighted by Fisher importance)
loss_ewc = (λ / 2) * SUM( F_i * (θ_i - θ_star_i)^2 )
```

The L2 CL baseline is EWC with F_i = 1 for all i. This is the correct interpretation from Hsu et al. (2018), and it is the comparison that isolates whether Fisher-weighted protection actually matters versus uniform protection.

---

## 5. Evaluation Protocol

> **Implementation:** See Contract 5 in `task_decomposition.md` for `evaluate()` and `compute_cl_metrics()` signatures (Emre's E2 and E3).

### 5.1 Accuracy Matrix

Build a 2×2 matrix R where R[i,j] = test accuracy on Task j after training through Task i.

```
EVALUATE(model, T1_test, T2_test, phase):
    # phase ∈ {"after_T1", "after_T2"}

    acc_T1 ← accuracy(model(T1_test, task=T1), T1_labels)
    acc_T2 ← accuracy(model(T2_test, task=T2), T2_labels)

    IF phase == "after_T1":
        R[1,1] ← acc_T1
        R[1,2] ← acc_T2     # zero-shot / random init performance
    ELSE:
        R[2,1] ← acc_T1     # retention of T1 knowledge
        R[2,2] ← acc_T2     # T2 final performance
```

**Evaluate at these checkpoints:**
1. After T1 training completes (before transition)
2. After T2 training completes

### 5.2 Continual Learning Metrics (per Lopez-Paz & Ranzato, 2017)

```
COMPUTE_CL_METRICS(R):
    BWT ← R[2,1] - R[1,1]
        # Negative = catastrophic forgetting
        # Positive = backward facilitation

    Forgetting ← R[1,1] - R[2,1]
        # = -BWT; reports as a positive number when forgetting occurs

    FWT ← R[1,2] - random_chance
        # random_chance = 1/20 for coarse, 1/100 for fine
        # NOTE: limited analytical value with 2 tasks (see discussion)

    RETURN BWT, Forgetting, FWT
```

**Read the papers for better formulas for FWT and BWT**
**The joint training reference** only reports final accuracy on Y_fine. CL metrics are undefined for it.

### 5.3 Standard Classification Metrics

At each checkpoint, also compute per-class precision, recall, and F1 on the active task's test set.

```
COMPUTE_CLASSIFICATION_METRICS(predictions, labels):
    For each class k:
        TP ← count(pred == k AND label == k)
        FP ← count(pred == k AND label != k)
        FN ← count(pred != k AND label == k)

        precision[k] ← TP / (TP + FP)
        recall[k]    ← TP / (TP + FN)
        F1[k]        ← 2 * precision[k] * recall[k] / (precision[k] + recall[k])

    macro_F1 ← mean(F1)
    RETURN macro_F1, per_class_F1
```

---

## 6. Representation Analysis

> **Implementation:** See Contract 6 in `task_decomposition.md` for `extract_embeddings()`, `compute_cluster_metrics()`, and `plot_tsne()` signatures (Yash's Y4–Y7).

### 6.1 Embedding Extraction

```
EXTRACT_EMBEDDINGS(model, test_loader):
    embeddings ← []
    labels_fine ← []
    labels_coarse ← []

    model.eval()
    For (x, y_coarse, y_fine) in test_loader:
        features ← model.backbone(x)    # 512-dim vectors
        embeddings.append(features.detach())
        labels_fine.append(y_fine)
        labels_coarse.append(y_coarse)

    RETURN stack(embeddings), stack(labels_fine), stack(labels_coarse)
```

**Extract at these checkpoints:**
1. After T1 training (before transition)
2. After T2 training (after transition)

### 6.2 t-SNE Visualization (qualitative)

```
VISUALIZE_TSNE(embeddings, labels, title):
    # Use consistent hyperparameters across all plots
    perplexity ← 30
    n_iter ← 1000
    random_seed ← 42          # fix seed for reproducibility

    projection ← TSNE(embeddings, perplexity, n_iter, seed=random_seed)
    scatter_plot(projection, colored_by=labels, title=title)
```

**Generate four plots per trajectory:**
1. After T1, colored by coarse labels (20 colors)
2. After T1, colored by fine labels (100 colors)
3. After T2, colored by coarse labels
4. After T2, colored by fine labels

### 6.3 Silhouette Score (quantitative)

Computed on the raw 512-dim embeddings, NOT on the t-SNE projection.

```
COMPUTE_CLUSTER_QUALITY(embeddings, labels_coarse):
    # Evaluate at the 20-superclass level
    silhouette ← silhouette_score(embeddings, labels_coarse, metric="cosine")
    db_index   ← davies_bouldin_index(embeddings, labels_coarse)

    RETURN silhouette, db_index
```

**Hypothesis to test:** Coarse-to-Fine should maintain higher superclass-level silhouette scores after T2 compared to Fine-to-Coarse, indicating better preservation of broad category structure.

### 6.4 Fisher Magnitude Analysis

```
ANALYZE_FISHER(F_coarse_to_fine, F_fine_to_coarse):
    # F values computed at the end of T1 for each trajectory

    For each trajectory's F:
        mean_F    ← mean(F)
        var_F     ← variance(F)
        sparsity  ← fraction of F_i below some threshold (e.g., 1e-6)
        top_k     ← indices of largest F_i values

    # Compare distributions
    Plot histogram of log(F_i) for both trajectories
    Report mean, variance, sparsity

    # Optionally: layer-wise analysis
    For each ResNet block (layer1, layer2, layer3, layer4):
        Report mean(F_i) for parameters in that block
```

**Hypothesis to test:** Coarse-grained T1 should distribute Fisher importance more uniformly across the network. Fine-grained T1 should concentrate importance in later layers.

---

## 7. Experimental Matrix (total runs)

Each cell is one full training run (T1 → T2):

| Trajectory | Unregularized | L2 (3 λ values) | EWC (3 λ values) |
|---|---|---|---|
| Coarse-to-Fine | 1 | 3 | 3 |
| Fine-to-Coarse | 1 | 3 | 3 |
| Joint Reference | 1 | — | — |

**Total (minimum): 15 runs** (plus the joint reference)

Each run should be repeated with 3 random seeds to report mean ± std.

**Total with repeats: 15 × 3 + 1 × 3 = 48 runs**

Estimated time per run (ResNet-18, CIFAR-100, 100 epochs total, single GPU): ~15–25 minutes.

**Optional extension (if time permits):** Re-run best-λ EWC conditions with the sampled true Fisher variant (Section 4.2) for Coarse-to-Fine and Fine-to-Coarse. This adds 2 × 3 = 6 runs (with 3 seeds). Report as a supplementary comparison to isolate whether the Fisher variant affects BWT.

> **Schedule:** See the day-by-day dependency graph in `task_decomposition.md` for when each experiment runs and who is responsible. If experiments are incomplete by Day 4, see the Minimum Viable Paper tiers in that document to determine what to submit.

---

## 8. Task Assignments, Workflow, and Schedule

**See `task_decomposition.md` for the full breakdown.** That document contains:

- **Repo structure:** exact file paths for every module
- **Interface contracts:** function signatures everyone codes against (Contracts 1–6)
- **Per-person task tables:** prioritized task lists with estimated time and dependencies
- **Dependency graph:** what blocks what, and when review windows happen
- **Mandatory cross-review rule:** no code merges to `main` without review and edge-case testing by a second person (fixed reviewer pairings are assigned there)
- **Communication protocol:** message templates for review requests, approvals, bug reports, and interface changes
- **Git workflow:** branching, merging, and the emergency override rule for Day 5
- **Testing checklists:** per-module checklists with both author and reviewer columns
- **Minimum Viable Paper tiers:** what to submit if experiments are incomplete

Summary of role assignments (details in task decomposition):

| Person | Primary Responsibility | Reviewer For |
|--------|----------------------|--------------|
| Shashwat | Data pipeline, LogReg baseline, paper lead | Emre's trainer/metrics |
| Yash | ResNet-18, embeddings, t-SNE, silhouette | Himanshu's EWC/Fisher |
| Himanshu | Fisher (both variants), EWC, L2 CL penalty | Yash's model/embeddings |
| Emre | Trainer, CL metrics, experiment orchestrator | Shashwat's data/LogReg |

---

## 9. References to Add to the paper

- Lopez-Paz & Ranzato (2017). Gradient Episodic Memory for Continual Learning. NeurIPS.
- Hsu, Liu, Ramasamy & Kira (2018). Re-evaluating Continual Learning Scenarios. NeurIPS CL Workshop.
- van de Ven & Tolias (2022). Three types of incremental learning. Nature Machine Intelligence.
- Chen, Li et al. (2023). Taxonomic Class Incremental Learning. arXiv:2304.05547.
- van de Ven (2025). On the Computation of the Fisher Information in Continual Learning. arXiv:2502.11756. *(for Fisher variant discussion)*
