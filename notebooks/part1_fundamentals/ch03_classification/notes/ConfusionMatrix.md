# Matrix Confusion
## Precision and Recall
How to truly measure a classifier: beyond **accuracy**, to the **F1 score.**

- cross-validation
- precision
- recall
- F1 score

![alt text](../images/WhatIsAConfusionMatrix.png)

---

## A basic example: Is a binary classifier a 3?
Handwritten MNIST digits. The simplest possible task: answer yes or no.

![alt text](../images/MNISTExample.png)

- **Yes** it is a 3 → 1
- **No** it is a 3 → 0

> Based on this example we will build **the entire** class: matrix, precision, recall and F1.

---

## `cross_val_score()` VS `cross_val_predict()`
Both use the same cross-validation strategy internally. The difference lies in what they return at the end.

### `cross_val_score()`
Returns an evaluation metric. How well it performed.
> [0.95, 0.96, 0.94]

### `cross_val_predict()`
Returns the predictions. What it predicted for each data point.

> [1, 0, 1, 0, 1, 0, ...]

> ### One stores scores, the other predictions.

---

