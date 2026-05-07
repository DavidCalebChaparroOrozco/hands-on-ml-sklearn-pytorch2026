# Part I. The Fundamentals of
Machine Learning

## What is Machine Learning?

**Simple Definition:**
Instead of telling the computer 'if A happens, do B', we give it data + examples, and the model learns the logic behind it on its own.

**Technical Definition:**
Development of algorithms and statistical models that, based on historical data, adjust their parameters to minimize errors and improve their performance in specific tasks.

**Branches of AI:**
Systems capable of learning patterns from data and making decisions or predictions without being explicitly programmed for each case.

Why we need to use ML: Example with spam detection system.

### Traditional Programming Approach
DATA + HANDWRITTEN RULES = LEARNED RULES (MODEL)

Manual Process:
1. Manually analyze numerous emails to identify patterns.
2. Detect frequent words: 'free', 'offer', 'urgent'.
3. Identify patterns: all caps, suspicious links, etc.
4. Write an algorithm with rules and thresholds.
5. Repeat: review more emails, add new rules.
6. Adjust the logic until acceptable performance is achieved.

What is the problem?
All rules must be programmed manually. The code ends up being a giant if/else structure, difficult to maintain and not very scalable.

### Machine Learning Approach
DATA + RESULTS = LEARNED RULES (MODEL)

**How It Works**
We provide the system with the data (emails) along with the expected outcome (spam or not spam). The algorithm automatically learns the internal patterns that we previously defined manually.

_Advantages:_
- No need to define explicit rules
- Automatically learns complex patterns
- More flexible and scalable
- The trained model can generalize to new cases

The Fundamental Difference:
We no longer write rules. The model learns them from the data.

## When to use Machine Learning?
1. **Difficult or Impossible Rules to Define:** If you can't write clear if/else type rules, machine learning (ML) is a good option. 

    Examples:
    - Spam detection
    - Image recognition
    - Sentiment analysis
    - Recommendation systems

    _If you need hundreds or thousands of rules, you probably need ML._

2. **Complex or Non-Obvious Patterns:** ML is useful when patterns are not obvious, relationships are non-linear, or there are many signals combined.

    Examples:
    - Financial fraud detection
    - Churn prediction
    - Assisted medical diagnosis
    - Behavioral analysis
    
    _The model learns patterns that a human does not easily see._

3. **The Problem Changes Over Time:**
If the rules are constantly changing, maintaining them manually is
impractical.

    Examples
    - Spam (new techniques every day)
    - Fraud detection
    - Recommendation engines
    - Anomaly detection in systems

    _With ML you can retrain the model, not rewrite the rules_

4. **Plenty of Historical Data Available:** ML learns from data, not intuition. You need historical data,
labeled (or unlabeled) examples, and sufficient volume.

    Examples:
    - System logs
    - Financial transactions
    - User events
    - Images, text, audio

    _Without data, there is no ML_

5. **The Need to Generalize:**
ML can respond well to novel, previously unseen cases.

    Examples:
    - Spam emails with new words
    - Images with distinct variations
    - Users exhibiting atypical behavior
    - Text in a new context
    
    _Rules fail, the model generalizes_

6. **Probabilistic Problem**
If there isn't a 100% correct answer, machine learning (ML) is ideal. ML works with uncertainty.

    Examples:
    - Probability of a user making a purchase
    - Probability of fraud
    - Credit risk level
    - Estimated lifetime value

    _ML produces probabilities, not absolute certainties_

7. **The Cost of Error is Acceptable:**
ML makes mistakes, always. It's a good option when the error is tolerable,
you can measure the impact, and you can iterate.

    Examples:
    - Recommendation systems
    - Search engine ranking
    - Automatic content classification
    - Experience personalization

    _In production, continuous monitoring is essential_

## When should you NOT use Machine Learning?
   - If you can solve it with a simple rule
   - If an SQL query is sufficient
   - If an if/else statement solves the problem
   - If it's just a business validation

_Don't use machine learning for simple problems_

## Examples of ML applications
  - Product classification on production lines
  - Tumor detection in brain scans
  - Automatic classification of news articles
  - Detection of offensive comments
  - Automatic summarization of long documents
  - Genetic risk estimation by analyzing DNA
  - Creation of chatbots and personal assistants
  - Business revenue forecasting
  - Apps with voice commands
  - Card fraud detection
  - Customer segmentation
  - Recommendation systems

## Types of Machine Learning Systems
_ML systems are classified in several ways according to how they learn, what data they use, and how they are trained._

1. **By Type of Supervision:**
Supervised, Unsupervised, Semi-supervised, Reinforcement Learning
2. **By How They Learn Over Time:**
Batch Learning, Online Learning
3. **By Generalization Method:**
Instance-based, Model-based

### Supervised and Unsupervised Learning Supervised Learning

The model is trained with labeled data. Each input has a known output. The goal is to find a mathematical pattern that relates inputs to outputs, minimizing error.

_Use Cases:_
- Classification: fraud, spam, diagnostics
- Regression: pricing, sales, demand

**Advantages:**
Controllable results, easy to evaluate

**Challenges:**
Requires labeled data, labeling is costly

### Unsupervised Learning
The model works with unlabeled data. It looks for patterns, groupings, or hidden structures within the information.

_Use Cases:_
  - Customer segmentation
  - Anomaly detection
  - Data exploration and analysis

**Advantages:** Does not need labels, discovers unknown information

**Challenges:** Results difficult to interpret, less clear evaluation

### Semi-Supervised Learning
Combines a small amount of labeled data with a large amount of unlabeled data. Used when labeling is expensive.

Why is labeling expensive?

- Requires experts (e.g., doctors for X-rays)
- Significant human time
- Extensive manual labor

    **Example: Cats vs. Dogs:**
    - Labeled Data: 1,000 images (cat or dog)
    - Unlabeled Data: 100,000 images

_How It Works:_
  1. Model learns general structure from unlabeled images (visual patterns)
  2. Uses the 1,000 labeled images to give meaning to the patterns
  3. Result: Better generalization than with labeled data alone

_Leverages abundant unlabeled data + scarce labeled data = More powerful model_

### Reinforcement Learning
An agent learns by interacting with an environment, making decisions, and receiving rewards or punishments. The goal is to maximize the accumulated reward.

_Learning doesn't happen from correct responses, but from rewards._

**Example: Learning to Ride a Bicycle
State**
   - Current: Balance
   - Action: Turn the handlebars
   - Reward: Don't fall (+) / Fall (-)
   - Policy: How to react to maintain balance

**No one gives you the right answer; you learn by trying.**

__Common Use Cases__
   - Robotics and systems control
   - Games (AlphaGo, video games)
   - Process optimization
   - Autonomous vehicles
   - Algorithmic trading

_Key Concepts:_
   1. **Agent**
   2. **Environment**
   3. **State**
   4. **Action**
   5. **Reward**
   6. **Policy**

## Batch Learning vs. Online Learning
Based on how they learn over time

### Batch Learning
The model is trained using the entire dataset at once. Once trained, it does not learn from new data unless it is retrained.

**Characteristics:**
   - Complete training with all data
   - Static model after training
   - Requires retraining for new data

_Typical Uses:_
  - Offline analysis
  - Models that are stable over time
  - Complete historical data

### Online Learning
The model is continuously updated as new data arrives. It learns incrementally.

**Characteristics:**
  - Continuous model updates
  - Adapts to changes in real time
  - Learns from each new data point


_Typical Uses:_
  - Production systems
  - Streaming data
  - Scenarios with concept drift
  - Limited memory resources

## Instance-Based vs. Model-Based
According to the method of generalization

### Instance-Based
The model memorizes examples and compares new data with existing data. It does NOT learn a general formula; it stores the training data.

_How it Works:_
  1. Stores all training data
  2. When new data arrives, it looks for the most similar instances
  3. Decides based on nearest neighbors
  4. The heavy lifting occurs in inference

**Example:**
_K-Nearest Neighbors (KNN):_ 

(+) Simple, does not require complex training.

(-) Memory-intensive, slow inference.

_Generalizes by comparing, not by learning global rules._

### Model-Based
The system learns a general model from the data and then uses it to make predictions. It summarizes the data into a mathematical structure.

_How It Works_
  1. Analyzes all training data
  2. Creates a mathematical representation (model)
  3. The heavy lifting occurs during training
  4. Does NOT require original data to predict

**Examples:**
- Regressions, Decision Trees, Neural Networks.

(+) Fast inference, Does not require data storage

(-) Training can be expensive, Requires more design

_Learns general rules, does not memorize specific cases_

## Example KNN: Customer Data
We have this customer data. We want to predict whether a new customer will buy.

| Age 	| Income 	| Buys? 	|
|-----	|--------	|-------	|
| 25  	| 2M     	| Yes   	|
| 30  	| 3M     	| Yes   	|
| 45  	| 1M     	| No    	|
| 50  	| 1.5M   	| No    	|

**New client**
_Age: 28, Income: 2.5M_

**How KNN Works**
1. Find the K most similar (closest) data points
2. See what decision those neighboring data points made
3. Decide by majority (or average)

_The model doesn't learn rules like "if age < 30, buy." Instead, it learns by similarity: "if it's similar to previous customers,
it will probably do the same."_