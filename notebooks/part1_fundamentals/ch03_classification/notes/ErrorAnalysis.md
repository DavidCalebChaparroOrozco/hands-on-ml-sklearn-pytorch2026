# Error Analysis (Scikit-learn)
Not just **how** your model fails (where and why).
Dissecting the `confusion matrix` with scikit-learn

- SGDClassifier
- MNIST: 60,000 images
- Accuracy: 89.7%

---

## A number is not a diagnosis (Definition)
Error analysis = studying **which** errors your model makes and **why** (don't just focus on the quantity).

![alt text](../images/ANumberIsNotaDiagnosis.jpg)

---

## The 5 Typical Tools (Toolbox)
Each one answers a different question about the same errors.

### confusion_matrix
Raw, absolute counts
> Answers: where the errors fall in **absolute** terms.

### `normalize = true`
by row (actual class), row total
> Answers: **recall** of the actual errors, what did it find?

### `normalize = "pred"`
by column (predicted), column total
> Answers: **accuracy** of what it predicted, what was correct?

### `sample_weight`
errors only, diagonal with weight 0
> Answers: Which class **contaminates** the others the most.

### `plt.imshow(...)`
individual inspection, would a human fail?

> Answers: **Reasonable** error or **structural** failure?

Error analysis = **structured curiosity** about your failures. Don't accept **"82% recall is 5s"** as an isolated fact (ask **where** those errors are going and **what mechanism** in the model explains it).

![alt text](../images/TheErrorAnalysisToolbox.jpg)


---

## We're at 89.7% (Now what?)
A promising SGDClassifier with scaled data. Two possible paths

### Current status: 89.7% accuracy (cross-val)
- **MNIST:** 60,000 train images
- **`StandardScaler`:** scaled features
- **`SGDClassifier`:** final, multi-class model

89.7% How do I improve?

- Reflexive approach: Try another model (Random Forest?, SVM?, tuning?) → Limited gain: if the problem is in the data or features **🎲blindly**

- Today we're going this way: Error analysis: study the errors first → Model or dataset?: that distinction changes everything you do afterward **🎯 directed**

> ⚠️ If **40%** of your errors come from a systematic confusion (5↔8), **no new hyperparameter will fix it**.

![alt text](../images/TwoPathsAfterYourFirstGoodModel.jpg)

---

## Anatomy of the Confusion Matrix
What the model **predicted** vs what **actually was** (counting every combination)

![alt text](../images/AnatomyoftheConfusionMatrix.png)

---
## Rows don't carry the same weight (the counting trap)

*"Which digit does the model recognize the worst?"*  
With raw counts, **you cannot answer.**

| Real digit | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | Total |
|------------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|------:|
| **1** | 0 | 6400 | 37 | 24 | 4 | 44 | 4 | 7 | 212 | 10 | **6742** |
| **5** | 27 | 15 | 30 | 168 | 53 | 4444 | 75 | 14 | 535 | 60 | **5421** |

### △ ≈ 1300

Difference in the number of examples between classes. This is **normal**; MNIST simply doesn't contain the same number of samples for every digit.

### Comparing raw counts = comparing different denominators

- Digit **1**: **6400 / 6742**
- Digit **5**: **4444 / 5421**

Does **6400 > 4444** mean that digit **1** performs better?

**You don't know.** Each row has a different total number of samples.

The fair comparison is to use **percentages**, not raw counts.

> **Solution:** Divide every cell by the total of its row → `normalize="true"`.

---

## Dividing by the row calculates **Recall** (`normalize="true"`)

Instead of asking:

> *"How many images were classified correctly?"*

we ask:

> *"Of all the **real** images of a class, what percentage received each prediction?"*

Each row is divided by its own total, so every row becomes a probability distribution that sums to **100%**.

```python
ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_train_pred,
    normalize="true",
    values_format=".0%"
)
```

### Reading the normalized confusion matrix

- Every **row adds up to 100%**.
- The **diagonal** now represents the **recall** of each class.
- Off-diagonal cells show **where the model confuses that digit**.

For example:

- **Recall(1) ≈ 95%** → the model correctly recognizes almost every real **1**.
- **Recall(5) ≈ 82%** → **5** is the most difficult digit for the model.

### Confusion is directional

One interesting observation is that confusion is **not symmetric**.

- **5 → 8 ≈ 10%**
- **8 → 5 ≈ 2%**

The model mistakes many more **5s as 8s** than **8s as 5s**.

> Confusing **A → B** is **not** the same as confusing **B → A**. Each row answers a different question because it starts from a different **true class**.