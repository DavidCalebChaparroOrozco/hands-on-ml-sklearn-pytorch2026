# Softmax Regression

**Multiclass Classification:** A machine learning task where a model assigns an input instance to one of three or more possible categories. For a given instance, the model calculates a raw score for each class and converts those scores into probabilities.

- **Scores (Logits):** The raw, unnormalized numerical values that the model computes for each class before converting them into probabilities.
- **Softmax:** A mathematical function that transforms a vector of raw scores into a probability distribution, ensuring every value is between $0$ and $1$ and all probabilities sum to $1$.
- **Argmax:** An operation that selects and returns the class with the highest probability (or highest score) as the model's final prediction.
- **Cross-Entropy:** A loss function that measures the difference between the predicted probability distribution and the actual target class, guiding the model to improve during training.

---

## What is Softmax Regression?
Softmax Regression (also known as **Multinomial Logistic Regression**) is a supervised machine learning algorithm used to solve multiclass classification problems. It is a direct generalization of Logistic Regression, extending it from handling only two classes (binary classification) to predicting multiple mutually exclusive classes simultaneously.

**Example:** 
Imagine a model designed to classify animal photos into three categories: **Cat**, **Dog**, or **Bird**. Given a photo, the model outputs a probability for each class such as $85\%$ Cat, $10\%$ Dog, and $5\%$ Bird—and predicts **Cat** because it has the highest probability.

---

## How does it work?
1. **Input:** The model receives a feature representation of an instance (for example, the pixel values of an image of a cat).
2. **Score Calculation:** Using the weights and biases learned during training, the model computes a linear score (logit) $s_k(\mathbf{x})$ for each possible class $k$.
3. **Probability Conversion:** It applies the **Softmax function** to normalize the scores into estimated probabilities $\hat{p}_k$:
   $$\hat{p}_k = \frac{\exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K} \exp(s_j(\mathbf{x}))}$$
4. **Final Prediction:** Using the **Argmax** operation, the model selects the class with the highest estimated probability as its final output (in this case, **"Cat"**).

---

## How does it learn those weights?

During training, the model iteratively adjusts its internal parameters (weights) to minimize prediction errors using two core components:

- **Cost Function (Cross-Entropy):** Measures how well the model's predicted probabilities match the actual target classes. It heavily penalizes the model whenever it assigns a low probability to the true class.
- **Optimization Algorithm (Gradient Descent):** An iterative optimization method that updates the model's weights in the direction that minimizes the Cross-Entropy cost function, gradually reducing prediction errors.

> **Key Takeaway:** Softmax Regression is a multiclass classifier that computes a raw score for each category, converts those scores into probabilities using the Softmax function, and predicts the class with the highest probability.

---

## Mathematical Definition

Given an input instance $\mathbf{x}$, the Softmax Regression model first computes a raw score $s_k(\mathbf{x})$ for each class $k$, and then estimates the probability of each class by applying the Softmax function to the scores.

### Vector Form
Each class $k$ has its own dedicated parameter vector $\boldsymbol{\theta}^{(k)}$. The score calculation mirrors linear regression:

$$s_k(\mathbf{x}) = \left(\boldsymbol{\theta}^{(k)}\right)^\top \mathbf{x}$$

### Expanded Form
Written out feature by feature, the score is a weighted sum of the input features (where $x_0 = 1$ represents the bias term):

$$s_k(\mathbf{x}) = \theta_0^{(k)}x_0 + \theta_1^{(k)}x_1 + \dots + \theta_n^{(k)}x_n = \sum_{i=0}^{n} \theta_i^{(k)} x_i$$