import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


df = pd.read_csv('online_shoppers_intention.csv')

# Part 1 a) Display the first and last 12 rows 
print("--- First 12 Rows ---")
print(df.head(12))

print("\n--- Last 12 Rows ---")
print(df.tail(12))
# b) Identify and print the total number of rows and columns 
print("\n--- 1b) Total Rows and Columns ---")
rows, cols = df.shape
print(f"Total rows: {rows}")
print(f"Total columns: {cols}")

# c) List all column names and their corresponding data types 
print("\n--- 1c) Column Names and Data Types ---")
print(df.dtypes)

# d) Generate a full dataset summary (non-null counts and data types) 
print("\n--- 1d) Full Dataset Summary ---")
df.info() # Note: info() automatically prints to the console

# e) Identify the target variable and analyze its class distribution 
print("\n--- 1e) Target Variable Class Distribution ('Revenue') ---")
print(df['Revenue'].value_counts(dropna=False))
print("\nPercentage distribution:")
print(df['Revenue'].value_counts(normalize=True) * 100)

# f) Select one categorical feature and display distinct/most frequent values 
print("\n--- 1f) Categorical Feature Analysis ('VisitorType') ---")
print("Distinct values:")
print(df['VisitorType'].unique())
print(f"Most frequent value: {df['VisitorType'].mode()[0]}")

# g) Compute mean, median, standard dev, and percentiles (20%, 50%, 75%) 
print("\n--- 1g) Numerical Features Statistics ---")
# The describe method calculates mean, std, and we can specify exact percentiles (50% is the median)
stats = df.describe(percentiles=[.20, .50, .75])
print(stats)

# h) Detect missing values and report their counts per column 
print("\n--- 1h) Missing Values per Column ---")
print(df.isnull().sum())

# i) Identify duplicate records and remove them if found 
print("\n--- 1i) Duplicate Records ---")
duplicates_count = df.duplicated().sum()
print(f"Number of duplicate records found: {duplicates_count}")

if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New dataset shape: {df.shape}")
else:
    print("No duplicates to remove.")

# Part 2: Data Preparation
print("\n=== Part 2: Data Preparation ===")

# 2a) Meaningful filtering condition
# We filter out sessions where the user visited 0 product-related pages.
# Justification: If a user doesn't look at a single product, their purchasing intention is fundamentally zero. 
# Removing these "instant bounce" or purely administrative visits helps our future models focus on actual shopper behavior.
initial_shape = df.shape
df_filtered = df[df['ProductRelated'] > 0].copy()
print("\n--- 2a) Filtering ---")
print(f"Removed {initial_shape[0] - df_filtered.shape[0]} rows where ProductRelated was 0.")
print(f"New dataset shape: {df_filtered.shape}")

# 2b) Encode categorical variables using an appropriate technique
print("\n--- 2b) Encoding Categorical Variables ---")
# We use Label Encoding for binary (True/False) targets so they become 1 and 0.
le = LabelEncoder()
df_filtered['Weekend'] = le.fit_transform(df_filtered['Weekend'])
df_filtered['Revenue'] = le.fit_transform(df_filtered['Revenue'])

# We use One-Hot Encoding (pd.get_dummies) for multi-class categoricals to prevent the model 
# from assuming false mathematical hierarchies (e.g., assuming Month 12 is "greater" than Month 1).
# We drop the first column to avoid the dummy variable trap (perfect multicollinearity).
categorical_cols = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
df_encoded = pd.get_dummies(df_filtered, columns=categorical_cols, drop_first=True)
print(f"Data shape after One-Hot Encoding: {df_encoded.shape}")

# 2c) Normalize numerical features using StandardScaler
print("\n--- 2c) Normalization ---")
numerical_cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
                  'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

scaler = StandardScaler()
# StandardScaler shifts the distribution to have a mean of 0 and a standard deviation of 1.
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
print("StandardScaler successfully applied to numerical features.")

# 2d) Divide one numerical feature into 5 equal-width bins and report bin distribution
print("\n--- 2d) Bin Distribution ---")
# Using pd.cut to create 5 equal-width bins on the scaled 'ProductRelated' feature.
df_encoded['ProductRelated_Binned'] = pd.cut(df_encoded['ProductRelated'], bins=5)
print("Bin distribution for ProductRelated (Scaled):")
print(df_encoded['ProductRelated_Binned'].value_counts().sort_index())

# 2e) Identify and handle missing values
print("\n--- 2e) Missing Values Handling ---")
missing_values_after_preparation = df_filtered.isnull().sum()
print("Missing values per column after filtering:")
print(missing_values_after_preparation)
print(
    "\nMedian would be used for numerical columns and mode for categorical columns "
    "because median is robust to outliers while mode preserves the most common category."
)
print("This dataset has no missing values, so no imputation step was required.")

# 2f) Use correlation analysis to detect numerical features that carry nearly the same information.
# When two numerical columns are very strongly correlated, keeping both can add redundancy without improving the analysis.
print("\n--- 2f) Correlation Analysis (Numerical Features) ---")
numerical_cols_for_selection = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
correlation_matrix = df[numerical_cols_for_selection].corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

correlation_threshold = 0.85
high_correlation_pairs = []
removed_numerical_features = []

for column in upper_triangle.columns:
    for row_name, corr_value in upper_triangle[column].dropna().items():
        if corr_value > correlation_threshold:
            high_correlation_pairs.append((row_name, column, corr_value))
            removed_numerical_features.append(column)

if high_correlation_pairs:
    print(f"Highly correlated pairs found (threshold > {correlation_threshold}):")
    for feature_a, feature_b, corr_value in high_correlation_pairs:
        print(f"- {feature_a} vs {feature_b}: {corr_value:.3f}")
else:
    print("No highly correlated numerical features found.")

removed_numerical_features = sorted(set(removed_numerical_features))

# Still in 2f, use the chi-square test to measure whether each categorical feature has a meaningful relationship with the target.
# Features with weak statistical association can be removed because they contribute little predictive value.
print("\n--- 2f) Chi-square Test (Categorical Features) ---")
categorical_features = ['Month', 'VisitorType', 'Weekend']
chi_square_results = []
removed_categorical_features = []

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(df['Revenue'])

for column in categorical_features:
    feature_encoder = LabelEncoder()
    encoded_feature = feature_encoder.fit_transform(df[column]).reshape(-1, 1)
    chi_square_stat, p_value = chi2(encoded_feature, y_encoded)
    chi_square_stat = float(chi_square_stat[0])
    p_value = float(p_value[0])

    chi_square_results.append((column, chi_square_stat, p_value))

    if p_value > 0.05:
        removed_categorical_features.append(column)

for feature, statistic, p_value in chi_square_results:
    print(f"- {feature}: chi2 = {statistic:.3f}, p-value = {p_value:.6f}")

selected_features = [
    column for column in df.columns
    if column not in removed_numerical_features and column not in removed_categorical_features
]

# 2g) Report exactly which features were removed and include the statistical evidence behind each removal.
# This keeps the feature-selection result explicit and avoids hiding the important answer behind generic messages.
print("\n--- 2g) Removed Features and Justification ---")
if removed_numerical_features:
    print("Removed numerical features:")
    for removed_feature in removed_numerical_features:
        matching_pair = next(
            pair for pair in high_correlation_pairs if pair[1] == removed_feature
        )
        kept_feature, _, corr_value = matching_pair
        print(
            f"- {removed_feature}: removed because it is highly correlated with "
            f"{kept_feature} (correlation = {corr_value:.3f})"
        )
else:
    print("Removed numerical features: None")

if removed_categorical_features:
    print("\nRemoved categorical features:")
    for removed_feature in removed_categorical_features:
        matching_result = next(
            result for result in chi_square_results if result[0] == removed_feature
        )
        _, chi_square_stat, p_value = matching_result
        print(
            f"- {removed_feature}: removed because chi2 = {chi_square_stat:.3f} "
            f"and p-value = {p_value:.6f} (> 0.05)"
        )
else:
    print("\nRemoved categorical features: None")

print("\nSelected features after feature selection:")
print(selected_features)

# 2h) Measure class imbalance in the target and then apply simple random undersampling.
# The goal is to reduce the dominance of the majority class so both classes are represented equally.
print("\n--- 2h) Class Imbalance Analysis and Sampling ---")
class_distribution_before = df['Revenue'].value_counts()
class_percentage_before = df['Revenue'].value_counts(normalize=True) * 100

print("Class distribution before sampling:")
print(class_distribution_before)
print("\nClass percentage before sampling:")
print(class_percentage_before.round(2))

majority_class = df[df['Revenue'] == False]
minority_class = df[df['Revenue'] == True]

balanced_majority = resample(
    majority_class,
    replace=False,
    n_samples=len(minority_class),
    random_state=42,
)

df_sampled = pd.concat([balanced_majority, minority_class]).sample(
    frac=1,
    random_state=42,
).reset_index(drop=True)

# 2i) Compare the class distribution before and after sampling to confirm that the balancing step worked.
print("\n--- 2i) Class Distribution Before vs After Sampling ---")
class_distribution_after = df_sampled['Revenue'].value_counts()
class_percentage_after = df_sampled['Revenue'].value_counts(normalize=True) * 100

print("Before sampling:")
print(class_distribution_before)
print("\nAfter simple random undersampling:")
print(class_distribution_after)

print("\nPercentage before sampling:")
print(class_percentage_before.round(2))
print("\nPercentage after sampling:")
print(class_percentage_after.round(2))






# Part 3: Data Visualization
print("\n=== Part 3: Data Visualization ===")

# Create an output folder for the generated charts so they can be reviewed after the script finishes.
visualization_dir = "visualizations"
os.makedirs(visualization_dir, exist_ok=True)

# 3a) Histogram for a relevant numerical feature
# PageValues is a useful feature because it reflects how much value a session generated before purchase or exit.
print("\n--- 3a) Histogram: PageValues ---")
plt.figure(figsize=(8, 5))
plt.hist(df['PageValues'], bins=30, color='steelblue', edgecolor='black')
plt.title('Histogram of PageValues')
plt.xlabel('PageValues')
plt.ylabel('Frequency')
histogram_path = os.path.join(visualization_dir, '3a_histogram_pagevalues.png')
plt.tight_layout()
plt.savefig(histogram_path)
plt.close()

print(f"Histogram saved to: {histogram_path}")
print(
    "Interpretation: The distribution is heavily right-skewed, which means most sessions "
    "have low PageValues while a smaller number of sessions have very high values."
)

# 3b) Boxplot for a significant feature
# ProductRelated_Duration is significant because time spent on product pages is closely tied to shopping intent.
print("\n--- 3b) Boxplot: ProductRelated_Duration ---")
plt.figure(figsize=(8, 5))
plt.boxplot(df['ProductRelated_Duration'], vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Boxplot of ProductRelated_Duration')
plt.ylabel('ProductRelated_Duration')
boxplot_path = os.path.join(visualization_dir, '3b_boxplot_productrelated_duration.png')
plt.tight_layout()
plt.savefig(boxplot_path)
plt.close()

print(f"Boxplot saved to: {boxplot_path}")
print(
    "Interpretation: This feature shows many high-value outliers, which justifies its choice "
    "because shopping-session duration varies strongly between casual visitors and engaged buyers."
)

# 3c) Scatterplot between two meaningful features
# ProductRelated and ProductRelated_Duration are paired behavioral features that describe browsing depth and time spent.
print("\n--- 3c) Scatterplot: ProductRelated vs ProductRelated_Duration ---")
plt.figure(figsize=(8, 5))
plt.scatter(df['ProductRelated'], df['ProductRelated_Duration'], alpha=0.4, color='darkorange')
plt.title('ProductRelated vs ProductRelated_Duration')
plt.xlabel('ProductRelated')
plt.ylabel('ProductRelated_Duration')
scatterplot_path = os.path.join(visualization_dir, '3c_scatter_productrelated_vs_duration.png')
plt.tight_layout()
plt.savefig(scatterplot_path)
plt.close()

print(f"Scatterplot saved to: {scatterplot_path}")
print(
    "Interpretation: The scatterplot shows a clear positive relationship. As the number of "
    "product-related pages increases, the total time spent on those pages also tends to increase."
)

# 3d) Correlation heatmap for the numerical features
# A heatmap summarizes the strongest linear relationships and supports the feature-selection results from Part 2.
print("\n--- 3d) Correlation Heatmap ---")
heatmap_corr = df[numerical_cols_for_selection].corr()
plt.figure(figsize=(12, 8))
heatmap = plt.imshow(heatmap_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(heatmap, fraction=0.046, pad=0.04)
plt.xticks(range(len(numerical_cols_for_selection)), numerical_cols_for_selection, rotation=90)
plt.yticks(range(len(numerical_cols_for_selection)), numerical_cols_for_selection)
plt.title('Correlation Heatmap of Numerical Features')
heatmap_path = os.path.join(visualization_dir, '3d_correlation_heatmap.png')
plt.tight_layout()
plt.savefig(heatmap_path)
plt.close()

print(f"Heatmap saved to: {heatmap_path}")
print(
    "Interpretation 1: ProductRelated and ProductRelated_Duration have a strong positive correlation, "
    "showing that users who open more product pages usually spend more time on them."
)
print(
    "Interpretation 2: BounceRates and ExitRates also have a strong positive correlation, "
    "which indicates that sessions with early bounces often also end with high exit behavior."
)

##################################################
# Part 4: Insight Discovery
##################################################

print("\n=== Part 4: Insight Discovery ===")

# 4a) Sessions that ended in a purchase generated far higher PageValues than non-purchasing sessions.
# This is the strongest target-related insight in the dataset because the average PageValues for
# Revenue=True sessions is 27.26 versus only 2.00 for Revenue=False sessions, and the difference
# is statistically significant with a Welch t-test p-value of 1.88e-173. The result suggests that
# PageValues captures meaningful purchase intent rather than random browsing noise. In practical
# terms, users who accumulate page value during the session are much more likely to convert.
print("\n--- 4a) Statistically Supported Insight About the Target ---")
pagevalues_true = df[df['Revenue'] == True]['PageValues']
pagevalues_false = df[df['Revenue'] == False]['PageValues']
pagevalues_ttest = ttest_ind(pagevalues_true, pagevalues_false, equal_var=False)

print(f"Average PageValues when Revenue = True: {pagevalues_true.mean():.2f}")
print(f"Average PageValues when Revenue = False: {pagevalues_false.mean():.2f}")
print(f"Welch t-test p-value: {pagevalues_ttest.pvalue:.3e}")
print(
    "Insight: Purchasing sessions have dramatically higher PageValues, so PageValues is strongly "
    "associated with the target variable."
)

plt.figure(figsize=(8, 5))
plt.boxplot(
    [pagevalues_false, pagevalues_true],
    tick_labels=['Revenue=False', 'Revenue=True'],
    patch_artist=True,
    boxprops=dict(facecolor='lightblue'),
)
plt.title('PageValues by Revenue Outcome')
plt.ylabel('PageValues')
insight_plot_path = os.path.join(visualization_dir, '4a_pagevalues_by_revenue.png')
plt.tight_layout()
plt.savefig(insight_plot_path)
plt.close()
print(f"Supporting visualization saved to: {insight_plot_path}")

# 4b) The interaction between VisitorType and Month reveals that new visitors convert especially well in November.
# The highest conversion segment is New_Visitor in November with a conversion rate of 30.55%, while the
# overall conversion rate in the cleaned dataset is only 15.63%. This matters because the effect is not
# explained by month alone or visitor type alone; it appears when both features are considered together.
# The pattern suggests that seasonal shopping periods attract high-intent first-time visitors.
print("\n--- 4b) Interaction-Based Insight ---")
visitor_month_conversion = (
    df.groupby(['VisitorType', 'Month'])['Revenue']
    .mean()
    .sort_values(ascending=False)
)
top_segment = visitor_month_conversion.index[0]
top_conversion_rate = visitor_month_conversion.iloc[0]
overall_conversion_rate = df['Revenue'].mean()

print(
    f"Top interaction segment: VisitorType = {top_segment[0]}, Month = {top_segment[1]}"
)
print(f"Conversion rate for this segment: {top_conversion_rate:.2%}")
print(f"Overall conversion rate: {overall_conversion_rate:.2%}")
print(
    "Insight: First-time visitors in November convert at a much higher rate than the dataset average, "
    "which shows a meaningful interaction between seasonality and visitor type."
)

# 4c) A counter-intuitive result is that SpecialDay is lower for purchasing sessions than for non-purchasing sessions.
# One might expect buying behavior to rise near special shopping days, yet the average SpecialDay score is
# 0.023 for Revenue=True sessions and 0.069 for Revenue=False sessions. The Welch t-test p-value is 1.47e-38,
# so the difference is statistically strong. This suggests that regular browsing behavior outside special-day
# periods may produce better conversions than sessions clustered near special-event dates.
print("\n--- 4c) Counter-Intuitive Finding ---")
specialday_true = df[df['Revenue'] == True]['SpecialDay']
specialday_false = df[df['Revenue'] == False]['SpecialDay']
specialday_ttest = ttest_ind(specialday_true, specialday_false, equal_var=False)

print(f"Average SpecialDay when Revenue = True: {specialday_true.mean():.4f}")
print(f"Average SpecialDay when Revenue = False: {specialday_false.mean():.4f}")
print(f"Welch t-test p-value: {specialday_ttest.pvalue:.3e}")
print(
    "Insight: Purchases are less associated with SpecialDay periods than non-purchases, which is an "
    "unexpected pattern in this dataset."
)

# 4d) A practical business recommendation is to prioritize campaigns that raise PageValues and target high-intent seasonal segments.
# The data shows that PageValues is much higher in purchasing sessions, and New_Visitor traffic in November
# converts at 30.55%, nearly double the overall conversion rate of 15.63%. A business should therefore
# emphasize stronger product-page value, clearer offers, and acquisition campaigns for new visitors during
# high-conversion months. This recommendation is evidence-based because it combines target strength and
# feature interaction into one actionable strategy.
print("\n--- 4d) Business Recommendation ---")
visitortype_revenue_table = pd.crosstab(df['VisitorType'], df['Revenue'])
visitortype_chi2 = chi2_contingency(visitortype_revenue_table)

print(f"New_Visitor conversion rate in November: {top_conversion_rate:.2%}")
print(f"Overall conversion rate: {overall_conversion_rate:.2%}")
print(f"VisitorType vs Revenue chi-square p-value: {visitortype_chi2[1]:.3e}")
print(
    "Recommendation: Focus acquisition and remarketing efforts on new visitors during November-like "
    "high-conversion periods, and optimize product pages to increase PageValues before checkout."
)

##################################################
# Part 5: Feature Engineering
##################################################

print("\n=== Part 5: Feature Engineering ===")

# 5a) Create a new behavioral feature that measures the average time spent per product-related page.
# This summarizes browsing depth and browsing time into one variable, which can be more informative
# than using only raw counts or only total duration.
print("\n--- 5a) New Feature Creation ---")
df_filtered['Avg_Product_Time'] = (
    df_filtered['ProductRelated_Duration'] / df_filtered['ProductRelated']
)

print("New feature created: Avg_Product_Time")
print("Definition: ProductRelated_Duration / ProductRelated")
print("First 10 values:")
print(df_filtered['Avg_Product_Time'].head(10))

# 5b) Explain why the engineered feature may help future classification models.
# Users may open a similar number of product pages but behave very differently in terms of time spent.
# This feature captures engagement quality, which can improve a classifier's ability to separate casual
# browsing from genuine purchase intent.
print("\n--- 5b) Why This Feature May Improve Classification ---")
print(
    "Avg_Product_Time may improve future classification models because it captures how much time a user "
    "spends on each product page on average. Two visitors can view the same number of product pages, "
    "but the visitor who spends longer on each page may show stronger purchase intent. This makes the "
    "feature useful for distinguishing shallow browsing from engaged shopping behavior."
)
