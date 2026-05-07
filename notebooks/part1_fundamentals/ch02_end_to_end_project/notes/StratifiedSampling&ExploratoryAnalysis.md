# Stratified Sampling & Exploratory Analysis

## What is Stratified Sampling?

### Definition
Stratified sampling is a technique where you do NOT take the sample completely randomly, but rather you first divide the population into homogeneous groups called STRATA.

### The Process
1. Divide the population into homogeneous groups (strata)
2. Take samples from each group separately
3. Maintain the same proportion that exists in the actual population

> The key: You respect the actual structure of the population instead of relying solely on chance

---

## Limitations of Simple Random Sampling

Simple random sampling assumes that, purely by probability, the sample will have a distribution similar to the population. But this is ONLY reliable when the dataset is very large.

## The Assumption
With large datasets, the law of large numbers works in your favor.

But with small datasets or unbalanced classes, chance can work against you.

- Small dataset
- Highly unbalanced classes
- Significant minority groups

## Risks
- Proportions may vary significantly
- Minority groups may be concentrated in a single set
- The model learns from unrepresentative data
- The assessment loses validity

---

## Example 1: Unbalanced Dataset
### Medical Classification - Cancer Screening

**Dataset: 1,000 patients**

_95% healthy (950) | 5% with cancer (50)_

Objective: Create a test set of 20% (200 patients)

### Dataset Composition
1,000 records

Healthy: 950 (95%)

With Cancer: 50 (5%)

### Expected Split (20% test)
200 patients

Ideal Test Set:

190 Healthy (95%), 10 with Cancer (5%)

Ideal Training Set:

760 Healthy (95%), 40 with Cancer (5%)

> Random sampling does not guarantee the proportion: the minority class could be unevenly distributed.

---

## Comparison: Random vs Stratified
|            Method            	|    Test Set (200 Patients)    	|    Train Set (800 Patients)   	| Problem                            	|
|:----------------------------:	|:-----------------------------:	|:-----------------------------:	|------------------------------------	|
| Random Sampling - Scenario 1 	|  195 healthy, 5 cancer (2.5%) 	| 755 healthy, 45 cancer (5.6%) 	| Test underrepresents cancer cases  	|
| Random Sampling - Scenario 2 	|  198 healthy, 2 cancer (1.0%) 	| 752 healthy, 48 cancer (6.0%) 	| Test has almost no positive cases  	|
| Random Sampling - Scenario 3 	| 185 healthy, 15 cancer (7.5%) 	| 765 healthy, 35 cancer (4.4%) 	| Train underrepresents cancer cases 	|
| **Random Sampling - ALWAYS** 	| 190 healthy, 10 cancer (5.0%) 	| 760 healthy, 40 cancer (5.0%) 	|**Perfectly maintained proportions ✅**|

> ✅ With Stratified Sampling, the proportions are respected in EVERY split, no matter how many times you repeat the split.

---

## Example 2: California Housing Dataset
### Median Income and Housing Prices

> Domain experts indicate that median_income (average district income) is a very important variable for predicting median_house_value (average house price).

### The Problem with Random Sampling
If median_income is poorly represented in the train/test sets, the model will learn incorrect patterns.

- Test set with too many wealthy districts
- Too few low-income districts
- Skewed distribution of the key variable

> Result: Sampling Bias → Model does not generalize correctly


> ### Key Insight
> If income influences price, then the income distribution should be maintained in both the train and test models.
> **Solution: Stratified Sampling by median income**

## From Continuous to Categories: pd.cut()

median_income is a continuous variable. You can't stratify directly on continuous values ​​because each value would be almost unique.

## Continuous Values ​​- Problem
```
2.354
4.982
6.123
1.874
3.567
```
> ✖️ Each value is almost unique → Impossible to group → There are no clear 'strata'

## Conversion to Categories - Solution `pd.cut()`
```
bins = [0.0, 1.5, 3.0, 4.5, 6.0, ∞]
labels = [1, 2, 3, 4, 5]
```

Result:
- 4.982 → Category 4
- 2.354 → Category 2
- 6.123 → Category 5

> ✅Now you can count proportions and stratify

---

## Why These Specific Bins?

### The Selected Bins
```
1 0.0 - 1.5
2 1.5 - 3.0
3 3.0 - 4.5
4 4.5 - 6.0
5 6.0+
```
### Based on the Histogram
The distribution analysis showed:
Majority of income between 1.5 and 6.0
High-income outliers were scarce
Decision: Create 5 robust groups

### Important Rules
- Each stratum must be of sufficient size.

- Avoid an excessive number of categories.

- Small strata hinder model learning.

> Key Principle: Large strata for effective learning.

---
## StratifiedShuffleSplit by Scikit-Learn
### What is it?

A scikit-learn class that splits a dataset into train/test MULTIPLE TIMES, maintaining the proportion of a variable.

**Stratified + Shuffle + Split**

`train_test_split(stratify=...)`
_Performs ONE split._

`StratifiedShuffleSplit`
_Can perform MANY splits (e.g., n_splits=10)_

> Ideal for evaluating stability, reducing error variance, and simulating scenarios.

How it Works
1. Identifies the proportion of each category.

2. Randomly shuffles WITHIN each stratum.

3. Maintains the proportion in the resulting split.

4. Repeats the `n_splits` process n times.

> **Key Principle:** Random sampling WITHIN each stratum

--- 

## Exploratory Data Analysis (EDA)

> So far, you've only divided the data correctly. Now comes the part about UNDERSTANDING THE PROBLEM.

### What is EDA? 
is the process of:
- Visualizing patterns in the data
- Detecting relationships between variables
- Identifying outliers and anomalies
- Generating hypotheses about the problem
- Discovering hidden insights

>⚠️ **Important Rule:** DO NOT explore the test set

> _The test set should simulate 'never-before-seen' data. If you analyze it too much, you introduce mental data leakage._
> Work ONLY with the training set during EDA

**EDA Objectives:**
- Reduce uncertainty about the problem
- Detect biases in the data
- Identify hidden relationships between variables
- Generate ideas for feature engineering
- Guide future modeling decisions

---

## Geographic Visualization
**Using Latitude and Longitude**

> The dataset has latitude and longitude columns → We can visualize it spatially

### Visualization Process
1. Basic scatter plot: x=longitude, y=latitude
2. Result: The shape of California appears
3. Initial problem: Saturated points, difficult to see patterns

### Visible Regions
- Bay Area (San Francisco)
- Los Angeles
- San Diego
- Central Valley
> The data is NOT uniformly distributed → There are natural clusters

### 👁️First Insights:
- Geography matters in this problem
- There are clear population concentrations
- Coastal areas look different
- Non-uniform distribution suggests a need for spatial features
> We need to improve the visualization to see more details

---

## Improving Visualizations
**Advanced Techniques**

### Three techniques transform a basic visualization into a rich source of information:

### **1 Transparency (alpha)**
`alpha = 0.2`
_Controls point transparency_
> When many points overlap → They appear darker → You detect dense areas

### **2 Point Size**
`s = population / 100`
_Size represents population_
> Larger circles → Higher population density → Adds an extra dimension of information

### **3 Color Map**
`c = median_house_value cmap = 'jet'`
_Color represents price_
> Blue → Cheap | Red → Expensive → Geographically Visible Price Patterns

## 🎯 Combining the Three Techniques
You get a chart that simultaneously shows: Location + Population Density + Housing Price
_A single visualization reveals multiple complex patterns_

---

## Insights Revealed by Visualizations

> Enhanced visualization reveals patterns that confirm real-world intuitions.

- Proximity to the Ocean: Houses near the ocean are more expensive.
- Population Density: Dense areas have higher prices.

- Geographic Patterns: Location is a strong predictor of price.

- High-Value Clusters: Clear clusters of high geographic value exist.

## 💡Implication for Feature Engineering

Create variables to measure proximity, distance, and density.

- Value Clusters
- Distance to the Ocean
- Population Density

---

## Correct Mental Flow in Machine Learning
> Models work best when the human already understands the problem's structure.

## The Correct Process
1. **Correct Split:** Use stratified sampling for representative splitting

2. **Explore Training Set:** Analyze ONLY the training set

3. **Visualize Patterns:** Create graphs that reveal relationships and structures

4. **Feature Engineering:** Create new features based on insights

5. **Train Models:**
Now, train and evaluate your models

> ## ❌ Common Mistake
> Jumping straight to the model without understanding the data results in models that don't generalize well.

