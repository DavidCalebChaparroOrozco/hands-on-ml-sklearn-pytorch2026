## The Importance of Data

The most important and also the most expensive element in a Machine Learning project is **data**.

- Self-generated data = Unrealistic patterns
- Poor quality data = Hypotheses far removed from reality
- Improving the algorithm = Improving data ingestion

## Open Data Resources

### Solution: Open Datasets on the Internet
Fortunately, there are many resources available online with open datasets that we can use for our learning process.

_Let's look at some of the most important ones_

- Tabular Data
- Images
- Text and Series

## Dataset Repositories
|                  Repository                  	|                                           Description                                           	|              Data Type              	|
|:--------------------------------------------:	|:-----------------------------------------------------------------------------------------------:	|:-----------------------------------:	|
|                Kaggle Datasets               	| Popular platform with thousands of datasets across multiple topics, many with example notebooks 	|  Image, text, tabular, time series  	|
|        UCI Machine Learning Repository       	|      Classic repository with hundreds of well-documented datasets, ideal for traditional ML     	| Tabular, classification, regression 	|
|                    OpenML                    	|       Collaborative collection of datasets with tasks and metadata ready for benchmarking       	|         Tabular, image, text        	|
|          Papers with Code - Datasets         	|                Extensive list of datasets categorized by task (NLP, vision, etc.)               	|         Image, text, tabular        	|
|             Google Dataset Search            	|   Search engine for datasets from multiple sources (governments, universities, organizations)   	|        Various types of data        	|
| AWS Open Data / Google Cloud Public Datasets 	|                        Large datasets for advanced analytics and Big Data                       	|       Large scale, Cloud-ready      	|
|                   Data.gov                   	|                    US government data portal with hundreds of public datasets                   	| Economy, Health, Education, Climate 	|
|       Data.world / World Bank Open Data      	|    General open data, useful for sociological, economic, or applied machine learning analyses   	|            Socioeconomic            	|

## California Housing Prices Dataset
### Source
- 1990 US Census data
- Each row = one census block group
- Geographic area with 600–3,000 people

### Content
- 20,640 samples
- 8 numerical features
- Target variable: Average house value (USD)

### Use
Classic dataset for supervised machine learning, especially for regression problems

## What is a Block Group?

### A neighborhood or group of several city blocks
**In simple terms:**
- A cluster of nearby houses
- Typically has between 600 and 3000 people
- Used to calculate averages, not individual data points

**That's why in the dataset you see:**
- MedInc = Average income of the neighborhood
- AveRooms = Average number of rooms per house
- Population = Total population of that area

_**Each row in the dataset represents a neighborhood, not a specific house**_

## Dataset Characteristics
|   Feature  	|                Description               	|
|:----------:	|:----------------------------------------:	|
|   MedInc   	|        Average income in the block       	|
|  HouseAge  	|           Average age of homes           	|
|  AveRooms  	|   Average number of rooms per household  	|
|  AveBedrms 	|        Average number of bedrooms        	|
| Population 	|       Total population of the block      	|
|  AveOccup  	| Averge number of occupants per household 	|
|  Latitude  	|            Geographic latitude           	|
|  Longitude 	|           Geographic longitude           	|

_**Target variable: Median House Value (average price in USD)**_

---

## Practice California Dataset
[Code](PracticeCaliforniaDataset.ipynb.ipynb)

> Remember: 
> 
> **X: Represents the characteristics of the dataset**
> 
> **y: Target variable (average price)**

---

## What is Scikit-learn?

It's a Python library focused on classical machine learning, allowing us to work with tasks such as regression, classification, clustering, and dimensionality reduction.

- **Preprocessing:**
    _Scaling, normalization, and encoding of data_

- **Feature Selection:**
    _Identifying the most relevant variables_

- **Training:**
    _Ready-to-use algorithms (regression, classification, etc.)_

- **Evaluation:**
    _Metrics such as accuracy, RMSE, and F1 score_

---

## A checklist provided by the author to follow step by step:

[Checklist](ml-project-checklist.md)

This is a practical guide that will walk you through your entire Machine Learning project step by step.

**Why use it?**
- Avoid common mistakes
- Don't forget important steps
- Make better decisions
- From idea to production

_You can adapt this list to your needs. This is **YOUR ROLE AS A ML ENGINEER**_.

1. Frame the Problem
2. Get the Data
3. Explore the Data
4. Prepare the Data
5. Select & Train Models
6. Fine-Tune Models
7. Present Solution
8. Launch & Monitor

---

## Frame the Problem
Step 1: Define the Problem Correctly

Before thinking about models or algorithms, you must understand **WHAT** you want to achieve and **WHY**.

- What is the business objective?

- What solutions currently exist?

- Supervised or unsupervised?

- How will we measure performance?

**These questions prevent building a model blindly**

## Defining the Business Objective

### What does the company want to achieve?

_It is essential to correctly define the problem before considering models_

- **Understand**
what the company wants to achieve

- **Identify**
what you want to predict

- **Define**
the problem correctly

### Case Study
**Predicting the average price of homes**
The model's output will serve as input for another process

Along with other signals, the decision to invest or not in an area will be made

---

## Current Solutions / Workarounds

### What is being done **TODAY** to solve this problem?

Don't build a Machine Learning model blindly or reinvent the wheel

- Manual process
- Excel spreadsheet
- Heuristic system
- Existing software
- Human decisions

**Compare against something real** NOT against an abstract idea

**Define the value of ML** Does it really provide improvement?

## Problem Type: Supervised / Regression

### What do we have?

- California census data
- Multiple features
- Target variable: average price
- Fixed batch of data
- Predict continuous numerical value

### Supervised Learning
Labeled data with clear target variable

- **Regression:** Predict continuous numerical value (price)
- **Multiple Regression:** Multiple input features
- **Univariate Regression:** Single output prediction

## What is Regression in Machine Learning?

The goal of regression is to predict a **NUMERICAL VALUE**

- **Monetary:** 
  - Price
  - Cost
  - Revenue

- **Temporal:**
  - Time
  - Duration
  - Age

- **Physical:**
  - Distance
  - Temperature
  - Quantity

⚠️ **Don't confuse levels:**

- Regression = Problem type
- Linear Regression, XGBoost, etc. = Models that solve it

_First, classify the problem, then choose the algorithm_

---

## How to Measure Performance?

How do you know if the model is working correctly?
"Correctly" isn't a feeling, it's a **NUMBER**.

That number is the **METRIC**.

Use an objective metric that represents what matters for the problem.
_The selection depends on the type of problem_

### Regression
(predicting numbers)
- RMSE
- MAE
- R²

### Classification
(predicting classes)
- Accuracy
- Precision
- Recall
- F-score

---

## RMSE: Root Mean Squared Error. 
Measures how wrong a model is, on average. **The smaller the RMSE, the better the model.** 

Formula:

$$
\mathrm{RMSE}(X, y, h) \;=\; \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( h\left(x^{(i)}\right) - y^{(i)} \right)^2}
$$

- m: total number of data points
- h(x): model prediction
- y: actual value

**RMSE = average error, penalizing large errors more**

### Example Temperature Prediction

| Day 	| Real (y) 	| Prediction h(x) 	| Error 	|
|:---:	|:--------:	|:---------------:	|:-----:	|
|  1  	|    20    	|        18       	|   -2  	|
|  2  	|    25    	|        30       	|   5   	|
|  3  	|    30    	|        28       	|   -2  	|

**Step by step:**
1. Errors: -2, 5, -2
2. Squared errors: 4, 25, 4
3. Average: $\frac{(4 + 25 + 4)}{3}  = 11$
4. Square root: $\sqrt{11} = 3.3$

RMSE = 3.3
On average, the model is off by ~3 degrees

---

## MAE: Mean Absolute Error

Measures how wrong the model is, on average, without exaggerating large errors.

**The "honest" average error**

Formula:

$$
\mathrm{MAE}(X, y, h) \;=\; \frac{1}{m} \sum_{i=1}^{m} \left| h\left(x^{(i)}\right) - y^{(i)} \right|
$$

- m = total number of data points
- h(x) = model prediction
- y = actual value
- || = absolute value (ignore sign)

### Example: Same temperature data
| Day 	| Real (y) 	| Prediction h(x) 	| Error 	|
|:---:	|:--------:	|:---------------:	|:-----:	|
|  1  	|    20    	|        18       	|   -2  	|
|  2  	|    25    	|        30       	|   5   	|
|  3  	|    30    	|        28       	|   -2  	|

**Step by step:**
1. Absolute errors: |-2| = 2, |5| = 5, |-2| = 2
2. Average: $\frac{(2 + 5 + 2)}{3} = 3$

**MAE = 3 degrees**

_On average, the model is off by 3 units_

---

## **RMSE was 3.3 | MAE is 3.0**

---

## RMSE vs MAE: Which to Use?

### RMSE
Penalizes large errors
_When large errors are very costly_

**$(error)^2$**

### MAE
Treats all errors equally
_For an "honest" view of the average error_

**$|error|$**

### Decision Guide:
- Do large errors cause critical problems?
  - Use RMSE -> Consider MAE

- Are there many outliers in your data?
  - MAE is more robust -> Both work

- Do you need to emphasize extreme precision?
  - Use RMSE -> MAE is more interpretable

## Production Applications

Real-world models: Which metric to use?
| Typical Real-World Application | Metric | Reason |
|:-----------------------------:|:------:|:------:|
| Pricing (houses, cars)        | MAE    | Normal Outliers |
| ETA / Logistics               | MAE    | Average Error Matters |
| Sales Forecast                | MAE    | Noisy Data |
| Industrial Sensors            | RMSE   | Large Errors Dangerous |
| Energy                        | RMSE   | Costly Spikes |
| Finance                       | RMSE   | Extreme Risk |
| Medicine                      | RMSE   | Safety Critical |

**The choice of metric depends on the CONTEXT and the CONSEQUENCES of the errors**