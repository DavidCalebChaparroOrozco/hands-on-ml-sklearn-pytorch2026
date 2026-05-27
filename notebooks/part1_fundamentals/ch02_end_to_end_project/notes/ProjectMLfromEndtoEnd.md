# Complete overview: from dataset to production model

## The Complete Project Journey
> 12 stages, one single goal: a robust and reproducible model

1. Loading & Exploration
2. Train/Test Split
3. Preparation
4. Feature Scaling
5. Transformers
6. Pipeline + Column Transformer
7. Models
8. Cross-Validation
9. Hyperparameters
10. Feature Importance
11. Final Evaluation
12. Save Model

---

## Dataset: California Housing (1990 US Census)
20,640 wards (rows)
- 10 attributes per ward
- 1 categorical column (ocean_proximity)
- NaN missing values ​​present

> Target: median_house_value

### 01. Loading and Exploration
**Initial Exploration Methods**
- `head()` First rows, quick view of the structure
- `info()` Data type and null values ​​by column
- `describe()` Statistics: mean, standard deviation, and percentiles
- `value_counts()` Frequencies in categorical variables
- `hist()` Visual distribution of each attribute
- `corr()` Correlations between numeric variables

---

