# Online Shoppers Intention Analysis

## 1. Project Purpose

This project analyzes an e-commerce session dataset to answer one core question:

**Can we understand and eventually predict whether a visitor will generate revenue in a browsing session?**

Each row represents one user session on an online shopping site. The target variable is `Revenue`, which is a boolean field:

- `Revenue = True`: the session ended with a purchase
- `Revenue = False`: the session did not end with a purchase

So this is not a "how much money was made" regression problem. It is a **binary classification problem** where the goal is to separate:

- purchasing sessions
- non-purchasing sessions

### What this project is trying to reach

The script is building the foundation for a future predictive model. Before training a classifier, it tries to:

1. understand the dataset structure
2. clean the data
3. reduce redundancy
4. identify useful signals related to `Revenue`
5. visualize important patterns
6. engineer at least one stronger feature
7. prepare a cleaner dataset for later machine learning

In simple terms:

**The destination is a model or decision process that can tell which sessions look likely to convert into purchases.**

## 2. Dataset at a Glance

The dataset file is [online_shoppers_intention.csv](/c:/Users/Lenovo/Desktop/data%20science%20project/online_shoppers_intention.csv).

Key facts from the current project:

- Raw dataset size: `12,330` rows and `18` columns
- Duplicate rows found: `125`
- Rows after duplicate removal: `12,205`
- Missing values: `0`
- Rows removed by the `ProductRelated > 0` filter: `38`
- Rows remaining after that filter: `12,167`

### Target distribution

In the cleaned dataset used for the Part 1 summaries:

- `Revenue = False`: `10,297` sessions (`84.37%`)
- `Revenue = True`: `1,908` sessions (`15.63%`)

This means the dataset is **imbalanced**. Most sessions do not end in a purchase.

## 3. What Each Column Means

Below is the practical meaning of each feature.

| Column | Meaning | Type |
| --- | --- | --- |
| `Administrative` | Number of administrative pages visited | Numerical |
| `Administrative_Duration` | Time spent on administrative pages | Numerical |
| `Informational` | Number of informational pages visited | Numerical |
| `Informational_Duration` | Time spent on informational pages | Numerical |
| `ProductRelated` | Number of product-related pages visited | Numerical |
| `ProductRelated_Duration` | Time spent on product-related pages | Numerical |
| `BounceRates` | Bounce-rate related session signal | Numerical |
| `ExitRates` | Exit-rate related session signal | Numerical |
| `PageValues` | Estimated value accumulated before purchase or exit | Numerical |
| `SpecialDay` | Closeness of the session to a special day | Numerical |
| `Month` | Month of the visit | Categorical |
| `OperatingSystems` | Visitor operating system category | Categorical |
| `Browser` | Visitor browser category | Categorical |
| `Region` | Region category | Categorical |
| `TrafficType` | Traffic source/type category | Categorical |
| `VisitorType` | Type of visitor, such as returning or new | Categorical |
| `Weekend` | Whether the session happened on a weekend | Boolean / Binary |
| `Revenue` | Whether the session resulted in a purchase | Boolean / Target |

### Why `Revenue` matters so much

Every part of the project should connect back to the target.

If a feature helps explain or predict `Revenue`, it is useful.
If a feature is redundant, noisy, or unrelated to `Revenue`, it becomes less useful.

That is why you see:

- class distribution analysis
- chi-square tests against `Revenue`
- comparisons like `PageValues` when `Revenue=True` vs `Revenue=False`
- class balancing

## 4. Project Files

| File | Role |
| --- | --- |
| [data.py](/c:/Users/Lenovo/Desktop/data%20science%20project/data.py) | Main analysis script |
| [data_science_project.ipynb](/c:/Users/Lenovo/Desktop/data%20science%20project/data_science_project.ipynb) | Notebook version of the work |
| [online_shoppers_intention.csv](/c:/Users/Lenovo/Desktop/data%20science%20project/online_shoppers_intention.csv) | Dataset |
| [visualizations](/c:/Users/Lenovo/Desktop/data%20science%20project/visualizations) | Saved charts produced by the script |

## 5. How the Script Maps to the Assignment

The analysis in [data.py](/c:/Users/Lenovo/Desktop/data%20science%20project/data.py) is organized into five parts. Each part answers a different group of questions.

### Part 1: Understanding the Dataset

This part answers the "What is this data?" questions and applies the first basic cleaning step: duplicate removal.
The script and notebook remove duplicate rows first, then report the Part 1 summaries on the cleaned dataset.

#### 1a. Show the first and last 12 rows

Code idea:

```python
df.head(12)
df.tail(12)
```

Why it exists:

- gives a quick visual feel for the data
- checks whether the file loaded correctly
- helps you spot obvious issues early

#### 1b. Count rows and columns

Code idea:

```python
rows, cols = df.shape
```

Why it exists:

- tells you dataset size
- helps you understand the scale of the problem

If you know C++/Java, think of `.shape` as metadata about a 2D container.

#### 1c. Show column names and data types

Code idea:

```python
print(df.dtypes)
```

Why it exists:

- shows which columns are numeric, categorical, or boolean-like
- helps decide what preprocessing each column needs

#### 1d. Full dataset summary

Code idea:

```python
df.info()
```

Why it exists:

- shows non-null counts
- confirms storage types
- quickly exposes missing-value problems

#### 1e. Analyze the target variable

Code idea:

```python
df['Revenue'].value_counts()
df['Revenue'].value_counts(normalize=True) * 100
```

Why it exists:

- identifies the target classes
- reveals class imbalance
- shows that most sessions are non-purchases

This is one of the most important early checks because the whole project revolves around `Revenue`.

#### 1f. Examine one categorical feature

The script chooses `VisitorType`.

Why it exists:

- shows the distinct categories
- identifies the most frequent category
- gives a human-level understanding of session types

#### 1g. Numerical statistics

Code idea:

```python
numerical_features = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'PageValues',
    'SpecialDay',
]

numerical_stats = df[numerical_features].agg(['mean', 'median', 'std']).T
numerical_percentiles = df[numerical_features].quantile([0.20, 0.50, 0.75]).T
numerical_percentiles.columns = ['20%', '50%', '75%']
stats = numerical_stats.join(numerical_percentiles)
```

Why it exists:

- provides mean, median, standard deviation, and percentiles
- avoids treating coded categorical IDs like `Browser` or `Region` as continuous numeric features
- helps detect skewed distributions and outliers

#### 1h. Missing values

Code idea:

```python
df.isna().sum()
```

Why it exists:

- checks whether cleaning is required
- supports the later statement that no imputation was needed

#### 1i. Duplicate rows

Code idea:

```python
df_raw = pd.read_csv('online_shoppers_intention.csv')
duplicates_count = df_raw.duplicated().sum()
df = df_raw.drop_duplicates().reset_index(drop=True)
```

Why it exists:

- duplicate records can distort statistics
- repeated rows can bias later analysis

In this dataset, `125` duplicates were found and removed before the other Part 1 summaries were reported.

### Part 2: Data Preparation

This part transforms raw data into something more useful for analysis and future modeling.

#### 2a. Meaningful filtering

The script keeps only rows where:

```python
df['ProductRelated'] > 0
```

Reasoning:

- if a session never touched a product-related page, it likely carries weak purchase intent
- removing these rows reduces obvious noise

After duplicate removal, this filter removed `38` rows.

#### 2b. Encoding categorical variables

This is where non-numeric values are converted into numeric representations.

Two methods are used.

**Binary boolean columns converted to 0/1**

```python
df_prepared = df_filtered.copy()
df_prepared['Weekend'] = df_prepared['Weekend'].astype(int)
df_prepared['Revenue'] = df_prepared['Revenue'].astype(int)
```

Meaning:

- `False` and `True` become numbers such as `0` and `1`

Why:

- many algorithms and statistical functions expect numeric input

If you know Java/C++, think of this like replacing a boolean or string category with an integer code.

**One-hot encoding for multi-category variables**

```python
df_encoded = pd.get_dummies(df_prepared, columns=MULTI_CATEGORY_FEATURES, drop_first=True)
```

Meaning:

- one category becomes several binary columns

Example idea:

- instead of one `Month` column storing `"Feb"` or `"Nov"`
- you get columns like `Month_Nov`, `Month_May`, and so on
- each row contains `0` or `1` in those columns

Why:

- it avoids pretending that categories have numeric order
- `"Nov"` is not mathematically greater than `"Feb"`

#### 2c. Normalization

The script standardizes selected numerical columns using:

```python
scaler = StandardScaler()
df_encoded[NUMERICAL_FEATURES] = scaler.fit_transform(df_encoded[NUMERICAL_FEATURES])
```

Meaning:

- values are rescaled to roughly mean `0` and standard deviation `1`

Why:

- features measured on very different scales become easier to compare
- useful for many machine learning algorithms

Analogy:

This is like converting measurements into a shared unit system before comparing them.

#### 2d. Binning

The script places raw `ProductRelated` values into 5 equal-width intervals:

```python
product_related_bins = pd.cut(df_filtered['ProductRelated'], bins=5)
```

Why:

- turns a continuous variable into grouped ranges
- helps summarize the distribution more simply

#### 2e. Missing-value strategy

The code checks for missing values after filtering and prepares median/mode imputation before encoding and scaling.

The project states:

- use median for numerical columns
- use mode for categorical columns

But in this dataset, no imputation is actually needed because missing values are still zero before and after the check.

#### 2f. Feature selection

This step asks two different questions:

1. Are any **numerical** features so similar that one is redundant?
2. Are the chosen **categorical** features meaningfully related to the target?

##### Numerical feature redundancy with Pearson correlation

Code idea:

```python
correlation_matrix = df_filtered[NUMERICAL_FEATURES].corr(method='pearson').abs()
```

Meaning:

- compute pairwise Pearson correlation between numerical columns
- use absolute value so both strong positive and strong negative relationships count

Why Pearson here:

- the goal is to detect linear redundancy between numeric attributes

Threshold used:

- `0.85`

High-correlation pairs found:

- `ProductRelated` vs `ProductRelated_Duration`: `0.860`
- `BounceRates` vs `ExitRates`: `0.901`

Removed numerical features:

- `ProductRelated_Duration`
- `ExitRates`

Reason:

- each removed feature was strongly correlated with another feature already carrying almost the same information

##### Categorical feature relevance with chi-square

Code idea:

```python
for column in CATEGORICAL_FEATURES_FOR_SELECTION:
    contingency_table = pd.crosstab(df_filtered[column], df_filtered['Revenue'])
    chi_square_stat, p_value, _, _ = chi2_contingency(contingency_table)
```

What this is testing:

- whether each categorical feature has a statistically meaningful relationship with `Revenue`

Why the target variable is used:

- this is feature selection
- a feature is useful only if it helps explain or predict the target

Decision rule:

- if `p-value > 0.05`, the feature is considered weak and removed
- if `p-value <= 0.05`, the feature is kept

Results in this project:

- `Month`: significant
- `VisitorType`: significant
- `Weekend`: significant
- `OperatingSystems`: significant
- `Browser`: significant
- `TrafficType`: significant
- `Region`: not significant (`p = 0.252859`)
- removed categorical features: `Region`

This version uses `chi2_contingency`, which is the more standard approach for nominal categorical features.

#### 2g. Report removed features and why

This section prints a human-readable explanation for removed features rather than silently dropping them.

That is good practice because it makes the logic auditable.
In this project, the removed features are `ProductRelated_Duration`, `ExitRates`, and `Region`.

#### 2h. Handle class imbalance

Because `Revenue=False` is much more common than `Revenue=True`, the script uses **stratified sampling by class** on the filtered dataset:

```python
minority_class_size = int(class_distribution_before.min())

for _, class_subset in df_filtered.groupby('Revenue'):
    stratified_samples.append(class_subset.sample(n=minority_class_size, random_state=42))
```

Meaning:

- treat each `Revenue` class as its own stratum
- sample the same number of rows from each class

Why:

- it keeps the sampling step tied to the filtered Part 2 dataset
- it creates a balanced working sample without mixing class proportions accidentally

#### 2i. Verify balancing

Before sampling on the filtered dataset:

- `False`: `10,265`
- `True`: `1,902`

After stratified sampling:

- `False`: `1,902`
- `True`: `1,902`

So the dataset becomes balanced for later classification experiments.

### Part 3: Data Visualization

This part answers the assignment requirement to create and interpret charts.

The plots are saved in [visualizations](/c:/Users/Lenovo/Desktop/data%20science%20project/visualizations).

#### 3a. Histogram of `PageValues`

Why this chart:

- shows the distribution of a meaningful numeric feature
- reveals strong right skew

Interpretation:

- most sessions have low page value
- a smaller number of sessions have very high page value

#### 3b. Boxplot of `ProductRelated_Duration`

Why this chart:

- shows spread and outliers clearly
- useful for identifying sessions with unusually high engagement

Interpretation:

- there are many high-value outliers

#### 3c. Scatterplot of `ProductRelated` vs `ProductRelated_Duration`

Why this chart:

- compares browsing depth against browsing time
- helps visualize whether more product pages usually means more time spent

Interpretation:

- there is a positive relationship

#### 3d. Correlation heatmap

Why this chart:

- gives a compact visual summary of all numerical linear relationships
- supports the earlier feature-selection step

Interpretation:

- `ProductRelated` and `ProductRelated_Duration` are strongly positively correlated
- `BounceRates` and `ExitRates` are also strongly positively correlated

### Part 4: Insight Discovery

This part goes beyond cleaning and tries to say something useful about shopper behavior.
The Part 4 results are reported on the filtered dataset from Part 2 so the analysis stays consistent.

#### 4a. Strong target-related insight

The script compares `PageValues` between purchasing and non-purchasing sessions using a Welch t-test.

Results:

- sessions with `Revenue=True`: `1,902`
- sessions with `Revenue=False`: `10,265`
- average `PageValues` when `Revenue=True`: `27.35`
- average `PageValues` when `Revenue=False`: `2.01`
- median `PageValues` when `Revenue=True`: `16.85`
- median `PageValues` when `Revenue=False`: `0.00`
- Welch t-test p-value: `1.268e-173`

Meaning:

- sessions that purchase have dramatically higher page values
- `PageValues` is strongly related to the target

Supporting visualization:

- [4a_pagevalues_by_revenue.png](/c:/Users/Lenovo/Desktop/data%20science%20project/visualizations/4a_pagevalues_by_revenue.png)

#### 4b. Interaction-based insight

The script groups by `VisitorType` and `Month`, then keeps segments with at least `100` sessions to avoid noisy tiny groups:

```python
visitor_month_summary = (
    insight_df.groupby(['VisitorType', 'Month'])['Revenue']
    .agg(['mean', 'count', 'sum'])
)
```

Top segment found:

- `VisitorType = New_Visitor`
- `Month = Nov`
- segment sessions: `417`
- segment conversions: `127`
- segment conversion rate: `30.46%`

Comparison baselines on the filtered dataset:

- `New_Visitor` overall conversion rate: `24.91%`
- `Nov` overall conversion rate: `25.49%`
- overall conversion rate: `15.63%`
- segment vs rest chi-square p-value: `3.995e-17`

Meaning:

- new visitors in November convert better than either the visitor-type baseline or the month baseline
- the combination of season and visitor type matters in a statistically meaningful way

#### 4c. Counter-intuitive finding

The script compares `SpecialDay` between target classes.

Results:

- average `SpecialDay` when `Revenue=True`: `0.0232`
- average `SpecialDay` when `Revenue=False`: `0.0692`
- share with `SpecialDay > 0` when `Revenue=True`: `4.05%`
- share with `SpecialDay > 0` when `Revenue=False`: `11.40%`
- Welch t-test p-value: `2.526e-38`

Meaning:

- in this dataset, purchases happen less near special-day periods than non-purchases
- this is not what many people would expect

#### 4d. Business recommendation

The script combines:

- the strong effect of `PageValues`
- the strong `New_Visitor` + `Nov` interaction
- a chi-square test comparing that segment against the rest of the dataset

Result:

- average `PageValues` when `Revenue=True`: `27.35`
- average `PageValues` when `Revenue=False`: `2.01`
- `New_Visitor` in `Nov` conversion rate: `30.46%`
- overall conversion rate: `15.63%`
- segment vs rest chi-square p-value: `3.995e-17`

Recommendation produced by the project:

- prioritize acquisition campaigns for new visitors during high-conversion November-like periods
- improve product-page clarity, offers, and checkout cues that are associated with higher `PageValues`

### Part 5: Feature Engineering

This part creates a new feature instead of only using the original dataset as-is.

#### 5a. Create `Avg_Product_Time`

Code:

```python
df_filtered['Avg_Product_Time'] = (
    df_filtered['ProductRelated_Duration'] / df_filtered['ProductRelated']
)
```

Meaning:

- average time spent per product-related page
- safe to compute because Part 2 already keeps only rows where `ProductRelated > 0`
- added to `df_filtered`, `df_prepared`, `df_encoded`, and `df_sampled` so later models can use it directly

Why this can help:

- two users may open the same number of product pages
- but the one spending more time per page may show stronger buying intent

Evidence from the filtered dataset:

- average `Avg_Product_Time` when `Revenue=True`: `42.13`
- average `Avg_Product_Time` when `Revenue=False`: `37.40`
- median `Avg_Product_Time` when `Revenue=True`: `35.83`
- median `Avg_Product_Time` when `Revenue=False`: `27.81`
- Welch t-test p-value: `6.751e-09`

This is a good example of combining two raw features into a more behavior-focused signal, and the project now adds it into the model-ready data structures instead of leaving it isolated at the end.

## 6. Python Guide for Someone Coming from C++ or Java

If Python syntax feels unclear, this section maps the main ideas to concepts you already know.

### `df = pd.read_csv(...)`

Think of `df` as an object similar to a table container.

- `pd` is the pandas library alias
- `read_csv` is like a factory function that returns a `DataFrame`

### `DataFrame`

A `DataFrame` is the central pandas data structure.

Rough analogy:

- like a 2D table object
- closer to a spreadsheet plus database-style operations

### `df['Revenue']`

This accesses one column.

Think of it like:

- getting one field collection from a table
- or selecting one vector by key

### Boolean filtering

```python
df[df['ProductRelated'] > 0]
```

This means:

1. create a true/false mask from `df['ProductRelated'] > 0`
2. use that mask to keep only matching rows

If you know C++/Java, conceptually it is like:

- iterating through rows
- keeping only rows where the condition is true

but pandas lets you write it in one expression.

### Method calls

Python frequently chains operations:

```python
df['Revenue'].value_counts(normalize=True)
```

Read it left to right:

1. get the `Revenue` column
2. count values
3. normalize the counts into proportions

### `for column in categorical_features:`

This is a standard loop over a list.

Equivalent idea in Java/C++:

```text
for each column in CATEGORICAL_FEATURES_FOR_SELECTION
```

### Lists

Example:

```python
CATEGORICAL_FEATURES_FOR_SELECTION = MULTI_CATEGORY_FEATURES + ['Weekend']
```

This is Python's built-in list type.

### List comprehension

Example:

```python
selected_features = [
    column for column in df_filtered.columns
    if column not in removed_numerical_features and column not in removed_categorical_features
]
```

This is a compact way to build a list.

Equivalent mental model:

- loop over all columns
- keep only the columns that satisfy the condition
- store the kept ones in a new list

### Tuples

Example:

```python
chi_square_results.append((column, chi_square_stat, p_value))
```

This adds a tuple containing three values.

Think of a tuple as:

- a small fixed-size group of values
- similar to a lightweight record without a named class

### Indentation matters

In Python, indentation is part of the syntax.

There are no braces like `{}` to define blocks.

So this:

```python
for column in CATEGORICAL_FEATURES_FOR_SELECTION:
    contingency_table = pd.crosstab(df_filtered[column], df_filtered['Revenue'])
```

means the indented line belongs inside the loop.

### `print(f"... {value} ...")`

This is an f-string.

It is similar to formatted output in other languages, but cleaner to read.

Example:

```python
print(f"Total rows: {rows}")
```

### `.copy()`

This appears in:

```python
df_filtered = df[df['ProductRelated'] > 0].copy()
```

It means:

- make an explicit copy of the filtered data
- avoid accidental issues from referencing the same underlying object

## 7. Why the Current Workflow Makes Sense

The script follows a sensible data-science order:

1. inspect the data
2. clean the data
3. encode and scale where needed
4. remove weak or redundant features
5. deal with class imbalance
6. visualize important patterns
7. extract insights
8. engineer a new feature

That ordering matters because later steps depend on earlier ones being correct.

Example:

- you should not discuss feature usefulness before checking duplicates and basic structure
- you should not trust a classifier later if class imbalance is ignored

## 8. Important Limitations and Notes

- This project performs analysis and preparation, but it does **not** yet train a final predictive model.
- The target is binary purchase outcome, not revenue amount.
- The chi-square section now uses contingency tables, which is a better fit for nominal categorical features.
- The balancing method is stratified sampling with equal class sizes, which is clear and assignment-friendly, but it still throws away some majority-class data.
- The feature-selection summary prints selected columns from the original dataset structure, while modeling preparation also creates an encoded dataset with extra dummy columns.

## 9. How to Run the Script

From the project directory:

```powershell
python data.py
```

This will:

- print the analysis to the console
- generate image files in the `visualizations` folder

## 10. Final Takeaway

This project is about understanding **purchase intention from browsing behavior**.

The target `Revenue` answers a simple question:

**Did this session lead to a purchase or not?**

Everything else in the script supports that goal:

- cleaning rows that add noise
- encoding data into usable numeric form
- reducing redundant features
- testing relationships with the target
- balancing the classes
- visualizing behavior patterns
- creating a stronger engagement feature

If you continue this project, the natural next step is to train a classification model such as:

- logistic regression
- decision tree
- random forest
- gradient boosting

using the cleaned and engineered data prepared here.
