# The Machine Learning Landscape (IT'S NOT THE MODEL)
The data is the real problem

## Introduction: The Real Problem

A model cannot learn what the data does not show.

1. Models learn from the examples they see.
2. They assume that the training data represents all of reality.
3. They don't "reason" or "understand" the context.
4. Incorrect data results in an incorrect model.

_📌 A model is only as good as its training data._

### Problem #1: Non-Representative Training Set

What does it mean?
The training data does NOT reflect the reality where the model will be used.

The model assumes that the training data represents the real world, but it doesn't.

_⚠ If the training world is different from the real world, the model will fail._

**Remember this:**
   - The model only sees examples
   - It look for patterns in those examples
   - It generalizes from what it sees.
   - It does NOT reason or understand the context

_A well-trained model in the wrong world = A useless model_.

### Practice example: E-commerce premium
Objective: To predict the probability of purchase

Variables:
  - Estimated income
  - Product price
  - Purchase history
  - Type of distribution channel

_⚠ Data only from premium physical stores:_
High-spending and loyal customers

**Consequences 📊**
  - Low variability in income
  - Price is NOT the deciding factor
  - The "non-buying" class is a minority

**Biased dataset**
_Does not represent the overall market_

### What does the model learn?

The model learns correctly... but ONLY in the premium context.
  - Income almost never limits the purchase.
  - Price has little importance.
  - The probability of purchase is high.

**Works well:**
_In premium stores_

Training data and usage have the same distribution.

**Doesn't work well:**
_In regular stores_

- Changes in income
- Changes in price sensitivity
- Changes in user behavior

_The model was trained for a different environment_

### Problem #2: Sampling Noise
**What is it?**
The dataset is representative, but it is VERY SMALL.

The sample exhibits random variations that do not reflect reality.

Even if the data is correct, if it is **small**, it cannot capture the true pattern.

- We see a sample, not the entire population.

- Small samples exaggerate or obscure characteristics.

- The model learns product patterns randomly.

**The problem:**
It's not bias, it's the size.

_Small samples:_
Random patterns.

_Large samples:_
True patterns.

**Few observations = Unreliable conclusions**

## Analogy: The Coin

Imagine flipping a coin:

5 times > 4 heads and 1 tail. Result: 80% heads 😐
10 times > Still unstable, variable results.

1,000 times > Approaches a true 50/50. ✅ Reliable result

_Results with few samples are NOT reliable_

**That's sampling noise**

_More data = More confidence in the patterns_

### Problem #3: Sampling Bias.
The way data is collected introduces bias: certain cases are more likely to be selected. It's not the quantity, it's how the sample is chosen.

| Concept                 	| Main Problem                  	|
|-------------------------	|-------------------------------	|
| Non-representative data 	| The data is out of this world 	|
| Sampling noise          	| There's very little data      	|
| Sampling bias           	| The sample is poorly selected 	|

**Example:**
Instagram-only survey
Only survey people who:
- Are active on social media
- Follow you (affinity)
- Use that platform

_Does it represent the entire population?_
**NO**

_Only a specific audience_

### Problem #4: Outliers
**What is an outlier?**
An observation that deviates SIGNIFICANTLY from the rest of the data.

It's not simply a 'weird' value.

It's a point that breaks the overall pattern of the dataset.

They can appear due to:
- Measurement errors
- Data loading errors
- Extremely rare cases
- Real but very infrequent events.

**Why are they a problem?**
Many models assume:
- The data follows general patterns
- Extreme values ​​have little influence.

**With Outliers:**
- They can DRAG the model
- They distort statistics (mean)
- They generate incorrect boundaries.

Outliers: Examples and Warnings

Example: Average Spending Prediction:
Normal Spending: $20 - $100
Outlier Spending: $50,000 (Corporate Purchase)

That single point can:
- Raise the average
- Bias the model
- Incorrect predictions

Analogy: Salaries
99 Employees: $1k
1 Director: $100k
_The 'average' is misleading_

Important: They are not always bad
- Errors: Eliminate
- Real Events: Analyze
- Anomalies: Are the target

_Do not automatically delete!_

## Solution: Regularization
Preventing the model from becoming overly complicated

**What is it?**

It's a technique to prevent the model from becoming 'too specific' with the training data.

**You impose LIMITS on learning**

_"Learn the general, don't memorize the details"_

### The problem: Overfitting
The model learns the training data **TOO WELL**, including noise and irrelevant details.

As a result, it fails to generalize with new, real-world data.

**It memorizes incorrect 'rules' such as:**
- 'If the house has 2-3 bathrooms, it sells'
- 'If the customer is from city X, they don't buy'

### The model memorizes instead of learning

## Analogy: The Student
### Without Tutoring 😐
_The student who memorizes:_
- Memorizes all the answers
- Learns useless details
- Gets a 10 on the practice exam
- _**Fails the real exam**_

---

**Overfitting**

### With Tutoring ✅
_The student who understands:_
- Learns the main ideas
- Doesn't get bogged down in details
- Doesn't get a 10 on the practice exam
- _**Passes the real exam**_

**Generalization**

---
Regularization sacrifices training accuracy for real-world performance gains.

Technically: It prevents model weights from becoming excessively large.

---

## Overfitting: Complete Definition
### What is overfitting?
The model fits the training data TOO tightly.

Instead of learning general patterns... it memorizes details, noise, and coincidences from the training data.

**Excellent performance in training, but FAILS with new data.**

**UNABLE TO GENERALIZE**

### Common Causes
Overfitting is facilitated by:
- Insufficient data (sampling noise)
- Data with outliers
- Noisy datasets
- Highly complex models

The model learns aspects that are NOT based on reality.

### The model gets confused... 😵‍💫
- Coincidences = Real patterns
- Noise = Useful information
- Specific cases - General rules

## Overfitting: Identification and Solutions
### How does it manifest in practice? 📊

- VERY HIGH performance on training data

- LOWER performance on validation/test data

**This difference indicates that the model does not generalize correctly**

### How to reduce overfitting?

- Collect MORE DATA.

- Apply REGULARIZATION
- Use cross-validation
- Simplify the model
- Remove noise from the data

_⚠️Overfitting does not mean the model is bad, but rather that it learned too much from the training data._

---

## Underfitting: The Opposite Problem
The model is TOO SIMPLE for the problem it's trying to solve.

_**It fails to learn the basic patterns in the data.**_

Poor performance **BOTH** in training and with new data.

This usually happens when:
- The model has very little capacity (it's too simple).
- The variables used are not predictive enough.
- The training is insufficient (too few epochs).

_The model isn't even able to understand the problem_

Characteristics:
- It doesn't capture the real relationship in the data.

- It ignores important information and patterns.
- Poor predictions (even in training).
- The problem is NOT in the data.
- It's in the model's capacity.

**Poor performance across the board**

## Underfitting: Example and Analogy

### Example: House Price Prediction 🏠

**You only use:**

Number of bedrooms

_But the price depends on:_
- Location
- Lot size
- Age
- Nearby amenities

**The model learns something very basic:**

_'More bedrooms = More expensive'_

**This pattern is INSUFFICIENT for reality. It fails in training and production.**

### Analogy: Simple Math 🎯
If you try to solve complex equations but only know how to add:
**YOU WILL FAIL.**

- Even if you have many exercises
- Even if you practice a lot

**The problem is NOT the amount of practice. It's that the method is TOO SIMPLE.**