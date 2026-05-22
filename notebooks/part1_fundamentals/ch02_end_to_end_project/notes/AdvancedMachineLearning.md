# Feature Importance, Bootstrap & IC, Deployment with CloudPickle

Understanding models, quantifying uncertainty, and deploying to production

## Beyond RMSE

### "Which variables are most influencing the predictions?"
Inspecting the fine-tuned model answers this question.

> The overall RMSE tells **how well** the model predicts; Feature Importance tells **why**.

- **Interpretability:** Understanding what the model has learned internally
- **Variable Selection:** Eliminating irrelevant features to simplify
- **Iterative Improvement:** Guiding feature engineering in future cycles

---

## .base_estimator_

### RandomizedSearchCV
Tests N combinations of hyperparameters

### CV Evaluation
Compares results using cross-validation

### .best_estimator_
The winning model, already trained with the best hyperparameters

- If you passed a simple model `RandomForestRegressor(best hyperparameters)`:
Returns the estimator with the best hyperparameters, already trained

- If you passed a Pipeline `Pipeline(steps= [preprocessor, model])`:
Returns the complete pipeline with all optimized steps

Feature importance tells us:

> "Which variables helped the model make good decisions most often?"

---

## Importance from the Trees

### How it's calculated
- The tree splits the data using different variables
- Measures how much each split improves prediction quality
- Accumulates this improvement per variable across all trees in the forest

### `.feature_importances_`
Array with one value per feature; the total sum is always 1.0
> Available in RandomForest, GradientBoosting, XGBoost


## Example: Feature Importance in a Random Forest

Imagine we want to predict if a house is expensive or cheap using these variables:

- House Size
- Number of Rooms
- Distance to Downtown
- Age of the House

When the Random Forest builds its trees, each tree asks questions like:

- "Is the house size bigger than 2000 sqft?"
- "Is the house close to downtown?"

Every time a question helps separate the data better, the model gives credit to that variable.

After training many trees, the model adds all the credit together and calculates how important each variable was.

Example result from `.feature_importances_`:

```python
[0.55, 0.20, 0.15, 0.10]
````

Meaning:

| Feature              | Importance |
| -------------------- | ---------- |
| House Size           | 55%        |
| Number of Rooms      | 20%        |
| Distance to Downtown | 15%        |
| Age of the House     | 10%        |

### Easy Interpretation

* **House Size** was the most useful variable for predictions.
* **Age of the House** helped a little, but not as much.
* All importance values together always add up to **1.0**.

---

## `SelectFromModel`

### 1. **Train the model:**
RandomForest or any estimator with importance values

### 2. **Get importance values:**
Read the .feature_importances_ file for each variable

### 3. **Apply threshold:**
Compare each importance value to the defined threshold

### 4. **Eliminate variables:**
Only the features above the threshold remain

### Threshold
- mean: average of all importance values
- median: median of the importance values
- 0.05: fixed numerical value

### Why use it?

- Automatically reduces dimensionality
- Improves inference speed
- Prevents overfitting due to noisy variables

---

How reliable is your RMSE?

⚠️ RMSE = 50
But how stable is that number?

> If the test data changes, the RMSE changes.

Same model, different test sets:

| Test Set              | Medium error |
| -------------------- | ---------- |
| A | 47 |
| B | 55 |
| C | 50 |
| D | **61** |

> Which of these is the model's true RMSE?

---

> # I've included a more detailed explanation of Bootstrap in a notebook.  
> [Open the Bootstrap explanation notebook](bootstrap_explained.ipynb)

---

## Model Deployment

From Training to Production

### **1. Train the model:** In the development environment

### **2. Serialize:** Convert to bytes that can be stored on disk

### **3. Move:** Move the model to the production server

### **4. Load and predict:** Deserialize and use for inference

### Why does standard pickle fail?

- **Functions within functions:** Pickle cannot serialize closures
- **Lambdas:** Anonymous functions cause errors
- **Dynamic objects:** Created at runtime, they are not serializable
- **Complex pipelines:** They fail with custom transformations