# Data loading, exploration, and preparation

## Data Download

Python function to download and load the housing dataset

- Automatic download > File extraction > Pandas and Polars support

``` Python
# Import necessary libraries
from pathlib import Path 
import pandas as pd 
import polars as pl
import tarfile 
import urllib.request

# URL
url = "https://github.com/ageron/data/raw/main/housing.tgz"

def load_housing_data(engine = 'pd'):
    datasets_dir = Path("datasets")
    tarball_path = datasets_dir / "housing.tgz"
    csv_path = datasets_dir / "housing" / "housing.csv"

    if not tarball_path.exists():
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        urllib.request.urlretrieve(url, tarball_path)

        with tarfile.open(tarball_path) as tar:
            tar.extractall(path=datasets_dir, filter="data")

    if engine == 'pl':
      return pl.read_csv(csv_path)
    elif engine == 'pd':
      return pd.read_csv(csv_path)
    else:
      raise ValueError("Invalid engine. Use 'pl' for Polars or 'pd' for Pandas.")
```

[Code](PracticeHousingDataset.ipynb)

---

## What is Pandas?

It's Python's classic library for data analysis.

**The de facto standard for over 10 years**

Its main structure is the **DataFrame**

### Main Capabilities

- Data Analysis
- Data Cleaning
- Transformation
- Dataset Exploration

### When to Use Pandas?

- Working with CSV, Excel, SQL
- Exploratory Analysis
- Notebooks and Scripts
- Machine Learning
- Data Science

---

## What is Polars?

The modern, ultra-fast library.

**For high-performance tabular data analysis**

_Write in **Rust**_

- **High performance**
Designed for extreme speed

- **Large datasets**
Designed to scale with large volumes of data

- **Modern DataFrames**
Same structure, different and more efficient approach

---
## Pandas VS Polars
| Feature              | Pandas        | Polars          |
|:----------------------:|:---------------:|:-----------------:|
| Base Language        | Python        | Rust            |
| Speed                | Medium        | Very High       |
| Memory Usage         | High          | Low             |
| Parallelism          | Limited       | Native          |
| Mutability           | Yes           | No              |
| Lazy Execution       | No            | Yes             |
| API                  | Very Flexible | More Strict     |
| Large Datasets       | Degrades      | Scales Better   |
| ML Integration       | Excellent     | Growing         |
| Learning Curve       | Low           | Medium          |
| Ecosystem Maturity   | Very Mature   | Expanding       |

---

## What are percentiles?

Also known as quartiles, percentiles describe how data is distributed.

A percentile is the value below which a specific percentage of observations lies.

**The 25th, 50th, and 75th percentiles are the most common.**

- **25% Q1 - First Quartile:**
25% of the data has values ​​less than or equal to this point.

- **50% Q2 - Median:**
Half of the data is below this central value.

- **75% Q3 - Third Quartile:**
75% of the data is below this value; only the top 25% is above it.

---
## Percentile Examples

Example Data Set:

2, 4, 6, 8, 10, 12, 14, 16

_8 values_

- 25% First Quartile (Q1): 6
25% of the data is below 6
- 50% Median (Q2): 9
50% is below (half)
- 75% Third Quartile (Q3): 14
75% of the data is below 14


---

## What are percentiles used for?

### Practical Use #1: Detecting "normal" vs. extreme values
Example: House prices
- 25th percentile: Affordable houses
- 50th percentile: Typical price (median)
- 75th percentile: Expensive houses

> **🌟 A house well above the 75th percentile is considered expensive for the market**

### Practical Use #2: Preventing outliers from misleading us
Example: Salaries with data of [1M, 1.2M, 1.3M, 1.4M, 20M]

✖️ Average ≈ 5M (misleading)

✅ Median (50th percentile) = 1.3M (representative)

> **The median and percentiles are robust to extreme values**

---
## Standard Deviation

**A statistical measure that indicates how dispersed the data is from the average**

_Measures how much the values ​​"deviate" from the average_

- # Small standard deviation
  - **Data concentrated near the average**
  - Less variability
  - The values ​​are similar to each other

- # Large standard deviation
  - **Data dispersed, far from the average**
  - Greater variability
  - The values ​​differ greatly from each other

### Example of Standard Deviation

Case 1: "Normal" Neighborhood

House Prices:
100k, 105k, 98k, 102k, 110k

Average ≈ 103k

**Homogeneous neighborhood, predictable prices**
- Price consistency
- Lower risk
- Easy to value

### Case 2: "Crazy" Neighborhood
House Prices:
50k, 80k, 100k, 200k, 500k

Average ≈ 186k

**Very unequal neighborhood, prices vary greatly**
- High variability
- Greater uncertainty
- Difficult to predict

---

## Standard Deviation in ML

- Problem: Variables with different scales
Size in ${m^2}$: 50, 80, 120, 300
Rooms: 1, 2, 3, 4

**Size "screams" more with larger numbers**

🌟 Solution: Standardization
$\frac{(value - average)}{standard .deviation}$

- Puts everything on the same scale
- Average becomes 0
- Most values ​​are between -1 and 1

**Models learn faster** gradient converges efficiently

**No bias from large numbers** all features have equal initial weight

**Better predictions** Improves model accuracy

---

## Histogram
Visualization tools for viewing the distribution of values
Groups data into intervals (bins) and shows how many observations fall into each interval

### 📊 How to read it:
X-axis (horizontal)
Possible values ​​of the variable
Example: Prices, rooms, income

Y-axis (vertical)
Number of records in that range
Example: Frequencies, count

_Each bar represents a range of values ​​and how many data points fall into it_

✅ Helps identify:
- Concentrations of values
- Dispersion and skewness
- Outliers
- Artificial limits

**🌟 Histograms are essential for exploratory data analysis (EDA)**