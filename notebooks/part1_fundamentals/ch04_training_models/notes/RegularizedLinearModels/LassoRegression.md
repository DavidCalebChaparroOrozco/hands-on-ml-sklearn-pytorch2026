# Lasso Regression (L1 Regularization)

Lasso Regression is a linear regression method that uses an L1 penalty to reduce the size of model coefficients. It can force some coefficients to become exactly zero, effectively removing those features from the model.

- L1 penalty: Adds a penalty based on the sum of the absolute values of the model coefficients.
- Alpha ($\alpha$): Controls the strength of the penalty. A larger alpha applies stronger regularization and may set more coefficients to zero.
- Feature selection: Automatically selects relevant features by setting the coefficients of less useful features to zero.
- 3D geometry: Lasso can be understood geometrically as a constrained optimization problem. In two dimensions, the L1 constraint forms a diamond-shaped region; in three dimensions, it forms an octahedron. The corners encourage some coefficients to become exactly zero.

Key idea: Lasso helps simplify linear models by reducing overfitting and selecting features, especially when some predictors contribute little to the prediction.

---

## The California Housing Problem

### Three features, one of which contributes very little

The model uses three features to predict house prices: size, location, and wall color.

Ordinary linear regression learns a coefficient ($\theta$) for each feature. Even if wall color has little predictive value, its coefficient may still be small but nonzero.

Lasso Regression applies an L1 penalty controlled by the regularization parameter ($\alpha$). As \\($\alpha$\\) increases, the model is encouraged to reduce its coefficients, and some can become exactly zero.

- Size: A feature that strongly influences house prices.
- Location: A feature that strongly influences house prices.
- Wall color: A feature that contributes little to predicting house prices, so Lasso may reduce its coefficient to zero.
- ($\theta$) (coefficient): The weight assigned to a feature, representing how the prediction changes with that feature while the others are held constant.
- ($\alpha$) (regularization strength): Controls how strongly Lasso penalizes large coefficients. A larger value generally encourages more coefficients to become zero.

### How Lasso removes a feature

Initially, all three features may have nonzero coefficients. As ($\alpha$) increases, Lasso adjusts the coefficients to balance prediction accuracy against model complexity.

When ($\alpha = 0.15$), in this example, the coefficient for wall color reaches exactly zero. The model effectively excludes wall color from its predictions while retaining size and location.

Key idea: Lasso performs automatic feature selection by setting some coefficients exactly to zero. The value ($\alpha$ = 0.15) is specific to this example; the actual value depends on the dataset, feature scaling, and implementation.

---


## What is Lasso Regression?

Lasso Regression is a regularization technique for linear regression that adds a penalty to the model to prevent overfitting. It simplifies models by performing feature selection, meaning it drives the coefficients of less important variables exactly to zero.

### Least Absolute Shrinkage and Selection Operator

- Least Absolute (L1 Penalty): Lasso penalizes the absolute values of the coefficients, written as \(\vert{}\theta_j\vert{}\), instead of their squares. Mathematically, the absolute value function creates a sharp "corner" at zero. This geometric property is what allows coefficients to hit zero exactly.
- Shrinkage: Lasso shrinks the model's coefficients toward zero to reduce complexity and control variance. The strength of this effect is controlled by the hyperparameter α. A higher α increases the penalty pressure, shrinking the weights more aggressively.
- Selection: Because the L1 penalty can force coefficients to exactly zero, Lasso acts as an automated feature selection tool. It eliminates irrelevant or redundant variables from the model without requiring manual feature engineering.

### Three Cost Functions: The Key Differences

Linear models change behavior based on how they penalize model complexity.

### 1. Ordinary Least Squares (No Regularization)

$$J(\theta) = \text{MSE}(\theta)$$
Use code with caution.
- Penalty: None.
- Behavior: The model focuses entirely on minimizing training error. Coefficients can grow as large as necessary to fit the data, which often leads to overfitting by memorizing random noise.

### 2. Ridge Regression (L2 Regularization)


$$J(\theta) = \text{MSE}(\theta) + \alpha \sum_{j=1}^{p} \theta_j^2$$
Use code with caution.
- Penalty: Proportional to the square of the coefficients $((\alpha \theta_j^2))$.
- Behavior: Large coefficients face a heavy penalty, while small coefficients near zero cost almost nothing because the penalty curve (a parabola) becomes flat at the bottom. It shrinks all weights close to zero but rarely sets them exactly to zero.

### 3. Lasso Regression (L1 Regularization)

$$J(\theta) = \text{MSE}(\theta) + \alpha \sum_{j=1}^{p} |\theta_j|$$
Use code with caution.
- Penalty: Proportional to the absolute value of the coefficients $((\alpha \vert{}\theta_j\vert{}))$.
- Behavior: It applies a constant penalization pressure regardless of 


---

