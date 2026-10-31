# Early Stopping

Early Stopping is a regularization technique that prevents a model from overfitting by stopping training when performance on the validation set stops improving.

Instead of modifying the cost function or adding a penalty term, it controls how long the model trains.

Supported by:

- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Keras

## Monitoring Validation Performance and Stopping at the Optimal Point

Early Stopping is a regularization technique used in iterative training algorithms, such as gradient descent, neural networks, and boosting.

The main idea is to monitor the model's performance on a validation set after each training iteration (or epoch) and stop when validation performance no longer improves.

### Ridge, Lasso, and Elastic Net

These regularization methods modify the cost function by adding a penalty that discourages excessively large model weights.

\\[ \text{Cost} = \text{MSE} + \alpha \times \text{Penalty} \\]

- Ridge (L2): Penalizes large weights by adding the sum of their squared values.
- Lasso (L1): Penalizes large weights by adding the sum of their absolute values.
- Elastic Net: Combines L1 and L2 penalties.

### Early Stopping

Early Stopping achieves a similar regularization effect through a different mechanism: it limits the number of training iterations.

- The model trains normally without modifying its cost function.
- Its performance is evaluated on a validation set.
- Training stops when validation performance stops improving for a specified number of iterations.

The training objective remains unchanged, but the model is prevented from training long enough to start memorizing noise in the training data.

Common applications:

- Gradient descent
- Neural networks
- Gradient boosting

### Understanding the Error vs. Epoch Graph

![alt text](../images/UnderstandingtheErrorvsEpochGraph.png) 

- Training error: Usually continues decreasing as the model learns the training data.
- Validation error: Initially decreases as the model learns useful patterns, but eventually starts increasing when overfitting occurs.
- Optimal stopping point: The epoch where validation error reaches its minimum.

## The Problem: Predicting House Prices Based on Square Footage

Imagine we have a dataset containing 1,000 houses, including their actual selling prices (in millions).

We want to train a machine learning model to predict house prices using square footage.

Input feature (X): Square footage, ranging from 40 to 200.

Target (y): Actual house price, measured in millions.

### Understanding the Relationship

Generally, larger houses tend to have higher prices. This is the underlying pattern we want the model to learn because it helps predict prices for other houses.

However, some houses may have unusually high selling prices due to exceptional circumstances, such as luxury features, special locations, or unusual transactions.

These observations may introduce noise or outliers into the dataset. Not every unusual price represents a general relationship between square footage and house prices.

### What Happens When We Train for Too Long?

A highly flexible model may initially learn the general relationship between house size and price.

However, if training continues for too long, it may start adapting to individual outliers and random fluctuations in the training data instead of learning patterns that generalize to new houses.

This is called overfitting.

Early Stopping helps prevent this by monitoring validation error and stopping training before the model begins to memorize noise.

---

# Data Splitting

Data splitting is the process of dividing a dataset into separate subsets, each with a specific purpose: training, validation, and testing.

The test set is separated first and kept untouched until the final evaluation. The remaining data is then divided into training and validation sets.

### Example: 1,000 Houses

Suppose we have a dataset containing 1,000 houses.

![alt text](../images/Example1000Houses.png)

- Test set: 200 houses (20%) — Set aside and kept untouched until the final evaluation.
- Remaining data: 800 houses (80%)
  - Training set: 640 houses (80% of the remaining data) — Used to train the model.
  - Validation set: 160 houses (20% of the remaining data) — Used to monitor performance and determine when to stop training through Early Stopping.

## Three Roles

Each dataset has a specific purpose during the model development process.

The three subsets remain the same across all epochs. What changes during training are the model's parameters $(\theta)$ and, consequently, its predictions.

### 1. Training Set: 640 Houses (Learning)

The training set is where the model learns patterns from the data.

- Gradient descent uses these houses to calculate gradients and update the model's parameters $(\theta)$.
- The training loss, typically measured using Mean Squared Error (MSE), is the objective the model tries to minimize.
- As training progresses, the training loss generally decreases.

> Usage: Used during every training epoch to update the model's parameters.

### 2. Validation Set: 160 Houses (Monitoring)

The validation set is used to evaluate how well the model performs on data that was not used to update its parameters.

- The model makes predictions using these houses, and the resulting validation loss is calculated.
- This loss is monitored after each epoch to track the model's ability to generalize to unseen data.
- Early Stopping uses validation performance to determine when training should stop.
- The validation set does not directly participate in gradient calculations or parameter updates.

> Usage: Evaluated after every epoch to monitor performance and guide training decisions. It does not directly update the model's parameters.

### 3. Test Set: 200 Houses (Final Evaluation)

The test set provides an unbiased estimate of how well the final model performs on previously unseen data.

- It is kept separate throughout training and model selection.
- It is not used for gradient descent, validation monitoring, or Early Stopping.
- Once training and model selection are complete, the final model is evaluated using the test set.
- Its results provide an estimate of how well the model may perform on new, unseen houses.

> Usage: Evaluated at the very end, ideally only once, after all model decisions have been finalized.

## What Happens Across Epochs?

The same 640 training houses and 160 validation houses are used throughout training. The 200 test houses remain untouched.

The model's parameters $(\theta)$ change as gradient descent updates them using the training data. Consequently, its predictions and errors also change.

| Component                       | Changes between epochs? |
| ------------------------------- | ----------------------- |
| Training data (640)             | No                      |
| Validation data (160)           | No                      |
| Test data (200)                 | No                      |
| Model parameters $(\theta)$     | Yes                     |
| Model predictions               | Yes                     |
| Training loss                   | Yes                     |
| Validation loss                 | Yes                     |

Key distinction: The training set teaches the model, the validation set helps monitor and select the model, and the test set evaluates its final performance.