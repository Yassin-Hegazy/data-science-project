# Online Shoppers Intention Analysis

## 1. Objective and Target

This project studies online shopping sessions to answer one main question:

**Which session patterns are associated with a purchase?**

The target variable is **Revenue**:

- `Revenue = True`: session ended with a purchase
- `Revenue = False`: session did not end with a purchase

So this is a **binary classification context** (purchase vs non-purchase), not a regression problem.

## 2. EDA Goal vs Preparation Goal

To keep the assignment logic consistent, the workflow is split into two purposes:

### EDA purpose (understanding data)

- Understand structure, types, and quality of data
- Describe class distribution of `Revenue`
- Explore feature behavior and relationships
- Identify signals that may explain conversion behavior

### Data-preparation purpose (getting data ready for later modeling)

- Filter less-informative sessions
- Encode categorical features
- Scale numerical features
- Check feature redundancy/significance
- Build a balanced sampled dataset for model experiments

Important: class balancing is used for modeling readiness, not for describing real-world prevalence.

## 3. Dataset Snapshot

File: `online_shoppers_intention.csv`

Current project figures:

- Raw size: 12,330 rows, 18 columns
- Duplicate rows removed: 125
- Cleaned size: 12,205 rows
- Missing values: 0
- Rows removed by `ProductRelated > 0` filter: 38
- Filtered size: 12,167 rows

Target distribution on cleaned data:

- `Revenue=False`: 10,297 (84.37%)
- `Revenue=True`: 1,908 (15.63%)

This indicates class imbalance in the real dataset.

## 4. Workflow Logic (Assignment Mapping)

## Part 1: Data Exploration

Purpose: answer What data do we have, and is it reliable?

1. Display head/tail rows
2. Check shape (rows, columns)
3. Check dtypes and `info()`
4. Inspect target distribution (`Revenue` counts and percentages)
5. Summarize numerical features (mean/median/std/quantiles)
6. Check missing values and duplicates

Why this is logically correct:
- EDA begins with quality + understanding before transformation.

## Part 2: Data Preparation

Purpose: prepare a cleaner analytical dataset.

1. Filter to `ProductRelated > 0`
2. Encode booleans and categorical columns
3. Standardize numerical columns (`StandardScaler`)
4. Bin `ProductRelated` for distribution view
5. Handle missing values if present (median/mode strategy)
6. Feature selection checks:
   - Pearson correlation for numerical redundancy
   - Chi-square tests for categorical relation to `Revenue`
7. Stratified class balancing for a model-ready sampled dataset

Why this is logically correct:
- You first clean and transform, then reduce noise/redundancy, then prepare balanced data for downstream models.

## Part 3: Visualization

Purpose: support EDA claims with visual evidence.

Generated charts:

- Histogram of `PageValues`
- Boxplot of `ProductRelated_Duration`
- Scatterplot of `ProductRelated` vs `ProductRelated_Duration`
- Correlation heatmap of numerical features

Why this is logically correct:
- Visuals validate distribution, outliers, and feature relationships found in numerical analysis.

## Part 4: Insight Discovery

Purpose: convert statistical outputs into interpretable findings.

Included insight types:

- Target-linked statistical difference (`PageValues` by `Revenue`)
- Interaction insight (`VisitorType` x `Month`)
- Counter-intuitive finding (`SpecialDay` behavior)
- Business recommendation tied to observed evidence

Why this is logically correct:
- EDA should end with interpretable conclusions, not just raw statistics.

## Part 5: Feature Engineering

Purpose: create a behavior-based signal for future classification.

Engineered feature:

- `Avg_Product_Time = ProductRelated_Duration / ProductRelated`

Why this is logically correct:
- It captures engagement intensity per product page, not only total volume.

## 5. Consistency Statements (Use These in Report)

To keep your submission internally consistent, state these explicitly:

1. **Scope statement**:
   Descriptive EDA is performed on cleaned data; some analyses are repeated on filtered active-shopping sessions (`ProductRelated > 0`).

2. **Imbalance statement**:
   Original class imbalance is reported from cleaned data; stratified balancing is only for later model experimentation.

3. **Feature-selection statement**:
   Correlation/chi-square outputs are used to justify feature relevance and redundancy before modeling.

These three lines make your academic logic clear and defensible.

## 6. Main Files

- `data.py`: full scripted workflow
- `data_science_project.ipynb`: notebook workflow
- `online_shoppers_intention.csv`: source dataset
- `visualizations/`: saved chart outputs

## 7. Final Academic Verdict

The project logic is **correct and consistent for an academic EDA assignment**, provided you clearly separate:

- EDA interpretation on original/cleaned data
- model-preparation operations (especially class balancing)

With that framing, your methodology is coherent and aligned with standard data analysis practice.
