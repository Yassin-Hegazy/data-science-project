# Online Shoppers Intention Analysis

## What This Project Is About

This project studies one question:

**Can we understand which website sessions end with a purchase?**

The dataset is an e-commerce session dataset. Each row is one browsing session by one visitor. The target column is `Revenue`:

- `Revenue = True` means the session ended with a purchase.
- `Revenue = False` means the session did not end with a purchase.

So this is a **binary classification** problem. We are not predicting how much money was made. We are trying to understand and later predict whether a purchase happens or not.

## What The Dataset Contains

The file is `online_shoppers_intention.csv`.

Important facts from this project:

- Raw dataset size: `12,330` rows and `18` columns
- Duplicate rows found: `125`
- Cleaned dataset size: `12,205` rows and `18` columns
- Missing values found: `0`
- Rows removed by filtering `ProductRelated > 0`: `38`
- Filtered dataset used for preparation and insights: `12,167` rows

### Target Distribution

On the cleaned dataset:

- `Revenue = False`: `10,297` sessions (`84.37%`)
- `Revenue = True`: `1,908` sessions (`15.63%`)

This tells us the dataset is **imbalanced** because non-purchase sessions are much more common.

On the filtered dataset used later in the project:

- `Revenue = False`: `10,265`
- `Revenue = True`: `1,902`

After stratified sampling:

- `Revenue = False`: `1,902`
- `Revenue = True`: `1,902`

## What The Columns Mean

### Browsing behavior columns

| Column | Meaning |
| --- | --- |
| `Administrative` | Number of administrative pages visited |
| `Administrative_Duration` | Time spent on administrative pages |
| `Informational` | Number of informational pages visited |
| `Informational_Duration` | Time spent on informational pages |
| `ProductRelated` | Number of product-related pages visited |
| `ProductRelated_Duration` | Time spent on product-related pages |
| `BounceRates` | Bounce-related session signal |
| `ExitRates` | Exit-related session signal |
| `PageValues` | Estimated value collected before purchase or exit |
| `SpecialDay` | Closeness of the visit to a special day |

### Context and visitor columns

| Column | Meaning |
| --- | --- |
| `Month` | Month of the visit |
| `OperatingSystems` | Visitor operating system group |
| `Browser` | Visitor browser group |
| `Region` | Visitor region group |
| `TrafficType` | Traffic source/type group |
| `VisitorType` | Visitor category such as new or returning |
| `Weekend` | Whether the session happened on a weekend |

### Target column

| Column | Meaning |
| --- | --- |
| `Revenue` | Whether the session ended with a purchase |

## What Exploratory Data Analysis Means

Exploratory Data Analysis, or **EDA**, is the stage where we try to understand the data before building a model.

The goal of EDA is to answer questions like:

- What does each column mean?
- Is the data clean?
- Are there duplicates or missing values?
- Which columns seem related to the target?
- Are there strange patterns, outliers, or imbalances?
- What should be cleaned or transformed before modeling?

In short:

**EDA helps us understand the story of the dataset before we try to predict anything.**

## How EDA Is Done In This Project

This project performs EDA in a simple sequence.

### 1. Look at the raw structure

The script shows:

- the first 12 rows
- the last 12 rows
- the number of rows and columns
- column names and data types
- a full `info()` summary

This helps answer: *What kind of data am I working with?*

### 2. Study the target variable

The target is `Revenue`, so the script checks:

- how many `True` values there are
- how many `False` values there are
- the percentage of each class

This helps answer: *Is the problem balanced or imbalanced?*

### 3. Describe the features

The script examines:

- one categorical feature: `VisitorType`
- all true numerical features using mean, median, standard deviation, and percentiles

This helps answer: *What values are common, and what does a typical session look like?*

### 4. Check data quality

The script checks:

- missing values
- duplicate rows

In this dataset:

- missing values are `0`
- duplicates are `125`, so they were removed

This helps answer: *Can I trust the data as it is, or does it need cleaning first?*

### 5. Prepare the data

The preparation stage includes:

- filtering out rows where `ProductRelated = 0`
- encoding categorical columns
- scaling numerical columns with `StandardScaler`
- binning `ProductRelated` into 5 equal-width ranges
- feature selection using correlation and chi-square
- balancing the classes with stratified sampling

This helps answer: *How do I turn the raw data into a better dataset for analysis and modeling?*

## Why Each Preparation Step Exists

### Filtering

The script keeps only rows where `ProductRelated > 0`.

Reason:

If a visitor never looked at a product page, that session gives us very little information about shopping intention.

### Encoding

Some columns are words or categories, such as `Month` and `VisitorType`. Models work better with numbers, so the script converts them using:

- `0/1` encoding for boolean columns
- one-hot encoding for multi-category columns

### Scaling

Numerical columns can have very different ranges. For example, page counts and durations are not on the same scale. `StandardScaler` puts them on a common scale so one large-range feature does not dominate the others.

### Binning

Binning turns one continuous feature into ranges. In this project, `ProductRelated` is divided into 5 equal-width bins so we can see how sessions are distributed across low and high product-page counts.

### Feature selection

Two methods are used:

- **Correlation analysis** for numerical columns
- **Chi-square test** for categorical columns

The goal is to remove features that are redundant or weakly related to the target.

Features removed in this project:

- `ProductRelated_Duration`
- `ExitRates`
- `Region`

Why they were removed:

- `ProductRelated_Duration` was highly correlated with `ProductRelated`
- `ExitRates` was highly correlated with `BounceRates`
- `Region` had a weak chi-square relationship with `Revenue`

### Stratified sampling

The dataset is imbalanced, so the script uses **stratified sampling by `Revenue` class**.

That means:

- treat `Revenue=False` as one group
- treat `Revenue=True` as another group
- sample the same number from each group

This creates a balanced dataset for later classification experiments.

## What The Visualizations Are Trying To Show

The project creates four charts:

- `3a_histogram_pagevalues.png`
- `3b_boxplot_productrelated_duration.png`
- `3c_scatter_productrelated_vs_duration.png`
- `3d_correlation_heatmap.png`

Why these charts matter:

- The histogram shows that `PageValues` is heavily right-skewed.
- The boxplot shows strong outliers in `ProductRelated_Duration`.
- The scatterplot shows a positive relationship between `ProductRelated` and `ProductRelated_Duration`.
- The heatmap summarizes the strongest numerical relationships in one view.

## Main Findings In Simple Language

### 1. Strongest target-related insight

`PageValues` is much higher in sessions that ended with a purchase.

On the filtered dataset:

- average `PageValues` for `Revenue=True`: `27.35`
- average `PageValues` for `Revenue=False`: `2.01`
- Welch t-test p-value: `1.268e-173`

This means `PageValues` is one of the strongest signals related to purchase behavior.

### 2. Interaction insight

The strongest large segment was:

- `VisitorType = New_Visitor`
- `Month = Nov`

Its conversion rate was `30.46%`, which is much higher than the overall filtered-dataset conversion rate of `15.63%`.

### 3. Unexpected finding

`SpecialDay` was lower in purchasing sessions than in non-purchasing sessions.

That is surprising because many people would expect sessions near special days to convert more often.

### 4. Business recommendation

The project suggests:

- improving product-page clarity and offers
- making checkout intent easier to capture
- focusing more on high-intent new-visitor traffic during November-like periods

## Feature Engineering

The new feature created is:

- `Avg_Product_Time = ProductRelated_Duration / ProductRelated`

Why this is useful:

It measures **average time spent per product page**, not just the total number of pages or the total time.

Evidence from the filtered dataset:

- average `Avg_Product_Time` for `Revenue=True`: `42.13`
- average `Avg_Product_Time` for `Revenue=False`: `37.40`
- median `Avg_Product_Time` for `Revenue=True`: `35.83`
- median `Avg_Product_Time` for `Revenue=False`: `27.81`

This suggests that deeper engagement per product page may help future classification models.

## Simple Glossary

- **Target variable**: the column you want to explain or predict
- **EDA**: understanding the data before modeling
- **Categorical feature**: a column made of labels or groups
- **Numerical feature**: a column made of numbers
- **Encoding**: turning categories into numbers
- **Scaling**: putting numerical columns on a similar range
- **Correlation**: how strongly two numerical features move together
- **Chi-square test**: a test for whether a categorical feature is related to the target
- **Stratified sampling**: sampling separately from each class to keep balance
- **Feature engineering**: creating a new feature from existing ones

## Project Files

- `data.py`: main script
- `data_science_project.ipynb`: notebook version
- `online_shoppers_intention.csv`: dataset
- `visualizations/`: saved charts

## How To Run

Run the script with:

```bash
python data.py
```

If you prefer the notebook, open:

```text
data_science_project.ipynb
```

## Final Takeaway

This dataset is not just a list of random website visits. It is a collection of shopping sessions, and the whole project is about understanding what separates:

- sessions that buy
- sessions that do not buy

EDA is the process that makes that understanding possible. It helps us clean the data, study the target, find useful patterns, and prepare the dataset for future machine learning.
