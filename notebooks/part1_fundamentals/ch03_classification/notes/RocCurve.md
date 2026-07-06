# Roc Curve
How the same model's performance changes when the **threshold** is adjusted, and why the ROC sometimes **deceives**

---

## `FixedThreshold Classifier`
A wrapper that changes the decision threshold of an already trained model (without retraining, without modifying the model).

- The model continues to calculate the same scores. Nothing changes internally.

- The only thing that changes is the rule for converting that score into a class.

### Same score = 1.8 (two thresholds)

**decision_score: 1.8**

- thresholds = 0 (default)
1.8 ≥ 0 → ✅ Positive

- thresholds = 2.5
1.8 < 2.5 → ✖️ Negative

> Same score, same network. **Only the cutoff point changes**

---

## `TunedThresholdClassifierCV`
Scikit-learn **finds the best threshold for you** using cross-validation, optimizing the metric of your choice.

### 1. Model Scores

| Sample | Model Score | True Class |
| ------ | ----------: | ---------: |
| 1      |         2.8 |          1 |
| 2      |         1.6 |          1 |
| 3      |         0.7 |          0 |
| 4      |        -0.5 |          0 |
| 5      |         3.4 |          1 |

The model outputs a **score** for each sample. Higher scores indicate a higher likelihood of belonging to the positive class (`1`).

### 2. Threshold Evaluation

| Decision Threshold | F1 Score |
| -----------------: | -------: |
|               -1.0 |     0.71 |
|                0.0 |     0.82 |
|                0.7 |     0.87 |
|          **1.5** ⭐ | **0.91** |
|                2.5 |     0.84 |

**Selected Threshold:** **1.5**

**Reason:** This threshold achieved the **highest F1 score** during **cross-validation (CV)**, providing the best balance between precision and recall.


> Unlike FixedThresholdClassifier (where you choose), here **scikit-learn finds it automatically** and uses **cross-validation** to avoid over-adjusting.

---

## The metric depends on your objective (Which metric do you want to optimize?)

By default, it optimizes **balanced_accuracy**, but you can request any scorer from scikit-learn.

### ✅ I want to reduce
False positives → accuracy

### 🕵🏽 I want to detect
The maximum number of positives → recall

### ⚖️ I want a
balance between both → f1

Supported scores:
- accuracy
- balanced_accuracy (default)
- precision
- recall
- f1
- roc_auc

---

## Fixed vs. Tuned Thresholds: Which Should You Use?

Choosing between these two approaches depends on whether you already know the optimal decision threshold.

| Feature | `FixedThresholdClassifier` | `TunedThresholdClassifierCV` |
|--------|-----------------------------|-------------------------------|
| Who chooses the threshold? | 🫵 You | 🤖 Scikit-learn |
| Searches for the best threshold | ❌ No | ✅ Yes |
| Uses cross-validation | ❌ No | ✅ Yes |
| Best use case | You already know the optimal threshold. | You want the model to find the optimal threshold automatically. |

### `FixedThresholdClassifier`

Use this classifier when you have already determined the decision threshold (for example, from previous experiments or business requirements).

**Characteristics:**
- Uses the threshold you specify.
- Does not evaluate alternative thresholds.
- Does not perform cross-validation.

### `TunedThresholdClassifierCV`

Use this classifier when the optimal threshold is unknown.

**Characteristics:**
- Evaluates multiple candidate thresholds.
- Uses cross-validation to estimate performance.
- Selects the threshold that maximizes the chosen evaluation metric (e.g., F1 score).

> **Rule of thumb**
>
> - Use **`FixedThresholdClassifier`** when the threshold is already known.
> - Use **`TunedThresholdClassifierCV`** when you want Scikit-learn to automatically determine the best threshold.


---

