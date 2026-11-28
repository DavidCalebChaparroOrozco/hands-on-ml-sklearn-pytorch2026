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

---

## What is an instance?
An **instance** (also called an observation or sample) is a single data item that we want the model to classify, represented by its measurable features.

### Training Dataset

| Weight (kg) | Height (cm) | Animal |
| :---: | :---: | :--- |
| 5 | 25 | Cat |
| 30 | 60 | Dog |
| 0.2 | 10 | Bird |
| 7 | 30 | Cat |
| 25 | 55 | Dog |

Each row in the table represents a single training instance, where **Weight** and **Height** are the input features and **Animal** is the target class.

### Example Instance
$$\mathbf{x} = \begin{bmatrix} 8 \\ 32 \end{bmatrix}$$

Here, $\mathbf{x}$ is the feature vector representing a new, unlabeled animal with a weight of $8\text{ kg}$ and a height of $32\text{ cm}$.

---

## A Score for Each Class

Using the same instance $\mathbf{x} = \begin{bmatrix} 8 \\ 32 \end{bmatrix}$:

- $\mathbf{x}$: The input instance (feature vector) we want to classify.
- $k$: The index of the class being evaluated ($k \in \{1, 2, 3\}$).
- $s_k(\mathbf{x})$: The raw score (logit) the model assigns to instance $\mathbf{x}$ for class $k$.

For a single instance, the model computes three separate scores—one for each possible category:

- $s_1(\mathbf{x})$: Score indicating how strongly instance $\mathbf{x}$ corresponds to **Cat**.
- $s_2(\mathbf{x})$: Score indicating how strongly instance $\mathbf{x}$ corresponds to **Dog**.
- $s_3(\mathbf{x})$: Score indicating how strongly instance $\mathbf{x}$ corresponds to **Bird**.

---

## What happens next with those scores?

Raw scores (logits) are not probabilities yet—they can be negative, larger than $1$, and do not sum to $1$. The **Softmax** function takes these raw scores and converts them into a valid probability distribution:

| Class | Score ($s_k$) | Softmax | Probability ($\hat{p}_k$) |
| :--- | :---: | :---: | :---: |
| **Cat** | $4.2$ | $\rightarrow$ | $89.65\%$ |
| **Dog** | $1.8$ | $\rightarrow$ | $8.13\%$ |
| **Bird** | $0.5$ | $\rightarrow$ | $2.22\%$ |

> Now the results are easily interpretable: there is an **$89.65\%$** probability that the instance belongs to the **Cat** class, **$8.13\%$** to **Dog**, and **$2.22\%$** to **Bird**.

---

## Recalling the Linear Regression Equation

$$\hat{y} = \boldsymbol{\theta}^\top \mathbf{x}$$

- $\mathbf{x}$: The feature vector of the instance (including the bias term $x_0 = 1$).
- $\boldsymbol{\theta}$: The vector of parameters (weights) learned by the model.
- $\boldsymbol{\theta}^\top \mathbf{x}$: The dot product of the parameter vector and the feature vector.
- $\hat{y}$: The model's numerical prediction.

**Example:**
Given the vectors $\mathbf{x} = \begin{bmatrix} 1 \\ 8 \\ 32 \end{bmatrix}$ and $\boldsymbol{\theta} = \begin{bmatrix} 2 \\ 3 \\ 0.5 \end{bmatrix}$:

$$\hat{y} = 2(1) + 3(8) + 0.5(32) = 2 + 24 + 16 = 42$$

### Returning to Softmax Regression

Softmax Regression uses the exact same mathematical foundation to compute each class score:

- **Linear Regression:** $\hat{y} = \boldsymbol{\theta}^\top \mathbf{x}$ (one dot product for a single continuous output)
- **Softmax Regression:** $s_k(\mathbf{x}) = \left(\boldsymbol{\theta}^{(k)}\right)^\top \mathbf{x}$ (one dot product per class $k$ using its own parameter vector $\boldsymbol{\theta}^{(k)}$)

> The mathematical operation (dot product) is identical; the difference lies in what the output represents (a class score rather than a final value) and how each class uses its own dedicated set of parameters.

---
## What does the superscript $(k)$ mean?

In standard Linear or Logistic Regression, the model learns a single parameter vector $\boldsymbol{\theta}$. However, in Softmax Regression, the model learns a dedicated parameter vector for each individual class.

For a classification task with three classes—**Cat** (class $1$), **Dog** (class $2$), and **Bird** (class $3$)—the model maintains three separate parameter vectors:

- $\boldsymbol{\theta}^{(1)}$: Weights learned for **Cat**
- $\boldsymbol{\theta}^{(2)}$: Weights learned for **Dog**
- $\boldsymbol{\theta}^{(3)}$: Weights learned for **Bird**

> **Note:** The superscript $(k)$ is a class index label that identifies which category the parameters belong to—it is **not** a mathematical exponent.

---

## An Instance with Three Outcomes

Below is a complete numerical walkthrough showing how Softmax Regression turns raw class scores into probabilities for our three animal classes (**Cat**, **Dog**, and **Bird**).

### Step 1: Compute the Exponential of Each Score
First, we apply the exponential function $\exp(s_k) = e^{s_k}$ (where $e \approx 2.71828$) to each raw class score:

| Class | Score ($s_k$) | Exponential ($e^{s_k}$) |
| :--- | :---: | :---: |
| **Cat** | $4.2$ | $e^{4.2} \approx 66.69$ |
| **Dog** | $1.8$ | $e^{1.8} \approx 6.05$ |
| **Bird** | $0.5$ | $e^{0.5} \approx 1.65$ |

> **Why take the exponential?** The exponential function converts all scores (even negative ones) into strictly positive numbers while preserving their ranking—a higher score always yields a larger exponential value. However, these values are still unnormalized and do not sum to $1$.

### Step 2: Normalize the Results into Probabilities
Next, we divide each class's exponential value by the sum of all exponential values across all classes:

$$\text{Sum} = 66.69 + 6.05 + 1.65 = 74.39$$

$$\hat{p}_{\text{Cat}} = \frac{66.69}{74.39} \approx 0.8965 \quad (89.65\%)$$

$$\hat{p}_{\text{Dog}} = \frac{6.05}{74.39} \approx 0.0813 \quad (8.13\%)$$

$$\hat{p}_{\text{Bird}} = \frac{1.65}{74.39} \approx 0.0222 \quad (2.22\%)$$

---

## Multiclass vs. Multioutput Classification

- **Multiclass Classification:** Predicts **one single label** out of three or more mutually exclusive categories for each instance. (This is what **Softmax Regression** solves.)
- **Multioutput Classification:** Predicts **multiple separate labels simultaneously** for a single instance, where each output can have its own set of classes (including binary or multiclass choices).

### Quick Comparison

| Feature | Multiclass Classification | Multioutput Classification |
| :--- | :--- | :--- |
| **Outputs per instance** | $1$ target label | $2$ or more target labels |
| **Relationship** | Mutually exclusive (only one can win) | Independent predictions made together |
| **Softmax Regression?** | Yes | No (requires a multioutput model) |

### Example (Using Our Animal Dataset)

Suppose we give a model a photo of an animal:

- **Multiclass:** The model answers **one question**—*"What animal is this?"*—and picks a single category from `{Cat, Dog, Bird}` (e.g., **Cat**).
- **Multioutput:** The model answers **multiple questions at the same time** for that same photo:
  1. **Species:** `{Cat, Dog, Bird}` $\rightarrow$ **Cat**
  2. **Size:** `{Small, Medium, Large}` $\rightarrow$ **Small**
  3. **Indoor Pet?** `{Yes, No}` $\rightarrow$ **Yes**