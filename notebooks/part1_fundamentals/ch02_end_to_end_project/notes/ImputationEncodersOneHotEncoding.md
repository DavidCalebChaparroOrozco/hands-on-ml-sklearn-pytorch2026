# Data Cleaning, Scaling, and Transformation

## What is Imputation

### Imputation
The process of replacing missing (NaN/null) values ​​in a dataset with reasonable estimates

### Why is it necessary?

- **Distance:** Breaks KNN, Regression, and Clustering
- **Statistics:** Affects mean and variance
- **Loss:** Fails in supervised loss functions

---

## Types of Imputation `Simple / Univariate`
> Each column is treated independently, using only its own data.

- **Mean Imputation:** Replaces with the column mean
    > Symmetrical distributions (Numeric)
- **Median Imputation:** Replaces with the column median
    > Skew distributions / outliers (Numeric)
- **Mode Imputation:** Replaces with the most frequent value
    > Categorical variables (Categorical)

---

## Types of Imputation: Multivariate and Advanced
### Multivariate / Model-Based
Uses other columns as predictors to estimate missing values.

- > ## KNN Imputer
    > Finds nearest neighbors and averages their values.

- > ## Iterative Imputer
    > Models each feature with the others, iteratively.

> When the column is strongly correlated with other features.

### Advanced Imputation
Complex models that learn the underlying distribution of the data
- > ## Probabilistic Models
    > Assume distribution and show to replace NaN

- > ## Deep Learning
    > Autoencoders and GANs to reconstruct missing data

> Complex datasets with nonlinear patterns between features

---

## Imputation Selection `Factor 1: Variant Type`

| Variable Type        | Example           | Typical Strategy                                  |
|----------------------|------------------|---------------------------------------------------|
| Continuous Numeric   | median_income    | Mean - Median - KNN - Iterative                   |
| Discrete Numeric     | number_of_rooms  | Median - Mode - KNN                               |
| Categorical          | ocean_proximity  | Mode - Special value 'missing' - KNN              |
| Boolean              | is_available     | Mode - Most frequent value (True/False)           |

> General rule: The imputation must maintain the original data type and respect its distribution.

---

## Imputation Selection `Factor 2: Distribution`

- ### Symmetrical / Normal `Mean`
    The mean is an efficient estimator when there are no outliers.
    > $Mean \approx Median$

- ### Skew / With Outliers `Median`
    The median is robust against extreme values.

    > total_bedrooms (California Housing) → use median

- ### Unbalanced Categories `Mode or 'missing'`
    When one category dominates almost all the data, imputing with the **mode** can reinforce that imbalance and hide useful information. Creating a new `'missing'` category allows the model to capture that the data was missing.

    Let's consider the variable `has_garage`:
    ```
    Yes → 90 houses
    No → 5 houses
    Missing → 5 houses
    ```

    * **If we impute using the Mode (Yes):**
    Now there would be 95 "Yes" responses → we artificially inflate the dominant category.

    * **If we create a category `'missing'`:**

    ```
    Yes → 90
    No → 5
    Missing → 5
    ```

    We retain the information that these houses did not report the data.

    > In highly unbalanced categorical variables, using only the mode can further skew the distribution. Creating a `'missing'` category can be more informative.

---

## Selection Factors

### % of Missing Values
- **< 5%:** Simple imputation (mean/median) is usually sufficient
- **5% ~ 20%:** Univariate imputation works - check for bias
- **>20%:** Multivariate imputation or remove column

> If 50% is NaN, imputing with the mean distorts the information

### Correlation with Other Columns
- **High correlation with other features:** Use Multivariate Imputation (KNN, Iterative)
- **No relevant correlation:** Simple imputation is sufficient

> **Multivariate imputation** leverages the other columns to improve the estimate.
> Instead of filling missing values using only the variable itself (mean/median/mode), it uses relationships between features to predict more realistic values.
>
> Example: if `total_bedrooms` is missing, we can estimate it using `households`, `population`, and `median_income`.

--- 

## SimpleInput `Scikit-Learn`

### What is it?
A Scikit-Learn class that replaces NaN values ​​with univariate strategies.

- Each column is treated independently
- Works with numeric and categorical variables

### Available strategies
- **mean:** Mean of the column `(numeric without outliers)`
- **median:** Median of the column `(skewed numeric)`
- **most_frequent:** Most frequent value (mode) `(categorical)`
- **constant:** Fixed value (fill_value) `(custom value)`

---

## fit() and transform() `SimpleImputer`

1. `fit(X_train)`: Calculates and stores the statistics for each feature in X_train
2. `statistics_`: The imputer stores the values ​​in `imputer.statistics_`
3. `transform(X_test):` Applies the statistics from X_train - DOES NOT recalculate

> Example: `imputer.statistics_`
> 
> | Feature        | Stored Median (statistics_) |
> |----------------|-----------------------------|
> | total_rooms    | 500                         |
> | population     | 2000                        |
> | median_income  | 3.2                         |
>
> 
> Ensures consistency between training and testing - Prevents data leaks

```python
from sklearn.impute import SimpleImputer
# Impute missing values using the median strategy
imputer = SimpleImputer(strategy="median")

housing_num = housing.select_dtypes(include=[np.number])

imputer.fit(housing_num)

imputer.statistics_
array([ 3.54155000e+00,  2.90000000e+01,  5.23234164e+00,  1.04859934e+00,
        1.16400000e+03,  2.81766108e+00,  3.42600000e+01, -1.18510000e+02,
        1.79500000e+00])
```

---

## Categorical Attributes
> A variable that represents discrete categories or classes, rather than continuous numerical quantities.

| Attribute         | Type        | Categories                                      |
|-------------------|-------------|--------------------------------------------------|
| gender            | Categorical | Male - Female                                   |
| ocean_proximity   | Categorical | NEAR BAY - INLAND - NEAR OCEAN - ISLAND         |
| color             | Categorical | Red - Blue - Green                              |

> Each value is a discrete class - there is no numerical hierarchy between them.

---

## OrdinalEncoder `Scikit-Learn`

### What does it do?

- Converts categorical attributes into integer values.

- Each category receives a unique integer.
- The numbers represent arbitrary order - not a real mathematical relationship.

> Works well when the category has a natural order: bad < average < good < excellent

ocean_proximity → OrdinalEncoder
| Original Category | Transformed Value |
|-------------------|-------------------|
| NEAR BAY          | 0                 |
| INLAND            | 1                 |
| NEAR OCEAN        | 2                 |
| ISLAND            | 3                 |

---

## Object VS Category in Pandas '3 Problems'

**1. Higher memory usage:** With object: each string is a separate object, repeated for each row.

> category stores an integer code per row + a mapping → much lighter

**2. Slow operations:** `value_counts(), groupby(), merges` → slower with object.

> Category uses internal integer codes → much faster operations

**3. ML doesn't recognize it:** Scikit-Learn doesn't accept strings → requires manual conversion
> With category: `.category` already has the categorical structure defined `(dtype='category')`, which simplifies the transformation and avoids conceptual errors.

---
## The OrdinalEncoder Problem: False Order

The model sees these numbers as continuous values

| ocean_proximity | Assigned Code |
|-----------------|---------------|
| INLAND          | 0             |
| ISLAND          | 1             |
| NEAR BAY        | 2             |
| NEAR OCEAN      | 3             |

What the model interprets:
`NEAR BAY (2) - INLAND (0) = 2`

The model believes that NEAR BAY is "twice as" as INLAND.
> ✖️ This doesn't make sense! There's no real hierarchy between locations.

> ✅ OrdinalEncoder DOES work for: bad < average < good < excellent

---

## One-Hot Encoding: The Solution

### What does it do?

It converts each unique category into an independent binary (0/1) column.

- Models don't understand text
- `OrdinalEncoder` introduces a false order
- Each category is completely independent.

> A row has 1 in its category, 0 in the rest → "one-hot".

### OneHot Encoding — ocean_proximity

1 columna → 4 columnas binarias

| ocean_proximity | INLAND | ISLAND | NEAR BAY | NEAR OCEAN |
|-----------------|--------|--------|----------|------------|
| INLAND          | 1      | 0      | 0        | 0          |
| ISLAND          | 0      | 1      | 0        | 0          |
| NEAR BAY        | 0      | 0      | 1        | 0          |
| NEAR OCEAN      | 0      | 0      | 0        | 1          |