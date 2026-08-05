# Linear Regression
The algorithm that learns a formula: from **features** to a **prediction** (and how the _training_ finds the **parameters 0** that minimize the **error.**)

- 0: parameters
- x:features
- ŷ: prediction
- MSE: error

## What is linear regression?
A machine learning algorithm that looks for a linear relationship between input variables and the value we want to predict.

## Simply put:
Learn a **formula** that transforms **features** into a **prediction**.

![alt text](../images/LinearRegressionPredictFormula.jpg)

---

### Example: Predicting a House Price

Suppose we want to estimate the price of a house.

#### Input (Features)

The model receives several characteristics of the house as input:

- **Size:** 150 m² → $(x_1)$
- **Bedrooms:** 3 → $(x_2)$
- **Age:** 10 years → $(x_3)$

These characteristics are called **features** because they describe the house.

---

#### Linear Regression Model

The model combines the features using the linear regression equation:

$\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3$

where:

- $(\theta_0)$ = bias (starting value)
- $(\theta_1, \theta_2, \theta_3)$ = learned weights
- $(x_1, x_2, x_3)$ = input features

---

#### Output (Prediction)

After applying the formula, the model predicts:

$\hat{y} = \$350,\!000$

This means the model **estimates** that the house is worth **$350,000**.

> **Important:** This is **not** the actual selling price. It is only the model's best prediction based on the information it has learned from the training data.

---

### Why is there a hat on $(\hat{y})$?

The model answers the question:

> **"Based on these features, how much is this house likely worth?"**

The output is written as **$(\hat{y})$** ("y hat") because it represents an **estimated value**, not the true value.

- $(y)$ → Actual house price (ground truth)
- $(\hat{y})$ → Predicted house price (model estimate)

---

### It's a Weighted Sum

Linear regression predicts a value by adding together the contribution of each feature.

For example:

$\text{Price} = 50,\!000 + 2,\!000 \times (\text{Size}) + 10,\!000 \times (\text{Rooms})$

or, using mathematical notation,

$\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2$

where:

- **$(\theta_0 = 50,\!000)$** → **Bias (Intercept):** the starting value of the prediction before considering any features.

- **$(\theta_1 = 2,\!000)$** → **Weight for Size:** each additional square meter increases the predicted price by **$2,000**.

- **$(\theta_2 = 10,\!000)$** → **Weight for Rooms:** each additional room increases the predicted price by **$10,000**.

### How the model thinks

The model starts with the **base price** (bias), then adjusts it according to the house's characteristics.

For a house with:

- Size = **150 m²**
- Rooms = **3**

the prediction becomes:

$50,\!000 + (2,\!000 \times 150) + (10,\!000 \times 3)$

Each feature contributes independently to the final prediction, and the model simply **adds all these contributions together**.

---

## Example: Predicting a House Price
Suppose we want to predict the price of a house using the following features:

* Size ($x_1$): $100\text{ m}^2$
* Rooms ($x_2$): $3$

The linear regression model is defined as:
$$\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2$$ 
Where the model weights (parameters) are:

* Bias ($\theta_0$): $50,000$
* Size weight ($\theta_1$): $2,000$
* Rooms weight ($\theta_2$): $10,000$

---
## 1. Calculate Feature Contributions
Multiply each feature value by its corresponding weight:

* Size contribution: $\theta_1x_1 = 2,000 \times 100 = 200,000$
* Rooms contribution: $\theta_2x_2 = 10,000 \times 3 = 30,000$

## 2. Substitute into the Model
Combine the base price and the calculated contributions into the main equation:
$$\hat{y} = 50,000 + 200,000 + 30,000$$ 
## 3. Compute Final Sum
Add all components together to get the final result:
$$\hat{y} = 280,000$$ 
------------------------------
## Final Prediction
The model predicts the total value of the house is:
$$\boxed{\hat{y} = \$280,000}$$ 

How it works: The model starts with a base price of $50,000 (the bias). It then adds $2,000 per square meter and $10,000 per room. The final prediction is the sum of all these individual components.