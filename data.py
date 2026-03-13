import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = 'Revenue'
CATEGORICAL_FEATURE = 'VisitorType'
ROWS_TO_DISPLAY = 12
FILTER_COLUMN = 'ProductRelated'
CORRELATION_THRESHOLD = 0.85
MINIMUM_SEGMENT_SIZE = 100
RANDOM_SEED = 42
NUMERICAL_FEATURES = [
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
MULTI_CATEGORY_FEATURES = [
    'Month',
    'VisitorType',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
]
CATEGORICAL_FEATURES_FOR_SELECTION = MULTI_CATEGORY_FEATURES + ['Weekend']
ENGINEERED_FEATURE = 'Avg_Product_Time'

df_raw = pd.read_csv('online_shoppers_intention.csv')
duplicates_count = df_raw.duplicated().sum()
df = df_raw.drop_duplicates().reset_index(drop=True)

print("--- First 12 Rows (Cleaned Dataset) ---")
print(df.head(ROWS_TO_DISPLAY))

print("\n--- Last 12 Rows (Cleaned Dataset) ---")
print(df.tail(ROWS_TO_DISPLAY))

print("\n--- 1b) Total Rows and Columns (Cleaned Dataset) ---")
rows, cols = df.shape
print(f"Total rows: {rows}")
print(f"Total columns: {cols}")

print("\n--- 1c) Column Names and Data Types ---")
print(df.dtypes)

print("\n--- 1d) Full Dataset Summary (Cleaned Dataset) ---")
df.info()

print(f"\n--- 1e) Target Variable Class Distribution ('{TARGET_COLUMN}') ---")
target_distribution = df[TARGET_COLUMN].value_counts(dropna=False)
target_total = len(df)
target_percentage = target_distribution / target_total * 100
print(target_distribution)
print("\nPercentage distribution:")
print(target_percentage)

print(f"\n--- 1f) Categorical Feature Analysis ('{CATEGORICAL_FEATURE}') ---")
distinct_values = df[CATEGORICAL_FEATURE].dropna().unique().tolist()
distinct_values.sort()
most_frequent_value = df[CATEGORICAL_FEATURE].mode(dropna=True)[0]
print("Distinct values:")
print(distinct_values)
print(f"Most frequent value: {most_frequent_value}")

print("\n--- 1g) Numerical Features Statistics ---")
mean_values = df[NUMERICAL_FEATURES].mean()
median_values = df[NUMERICAL_FEATURES].median()
std_values = df[NUMERICAL_FEATURES].std()
percentile_20 = df[NUMERICAL_FEATURES].quantile(0.20)
percentile_50 = df[NUMERICAL_FEATURES].quantile(0.50)
percentile_75 = df[NUMERICAL_FEATURES].quantile(0.75)

stats = pd.DataFrame()
stats['mean'] = mean_values
stats['median'] = median_values
stats['std'] = std_values
stats['20%'] = percentile_20
stats['50%'] = percentile_50
stats['75%'] = percentile_75
print(stats)

print("\n--- 1h) Missing Values per Column ---")
print(df.isna().sum())

print("\n--- 1i) Duplicate Records ---")
print(f"Duplicate records found in raw dataset: {duplicates_count}")
print(f"Dataset shape after duplicate removal: {df.shape}")

if duplicates_count > 0:
    print("Duplicates were removed before the Part 1 summaries above.")
else:
    print("No duplicates were found, so no rows were removed.")

print("\n=== Part 2: Data Preparation ===")

# Keep only sessions where the visitor actually viewed product pages.
initial_shape = df.shape
df_filtered = df[df[FILTER_COLUMN] > 0].copy()
missing_values_before_imputation = df_filtered.isna().sum()

# If missing values ever appear, use median for numerical columns and mode for categorical columns.
if missing_values_before_imputation[NUMERICAL_FEATURES].sum() > 0:
    df_filtered[NUMERICAL_FEATURES] = df_filtered[NUMERICAL_FEATURES].fillna(
        df_filtered[NUMERICAL_FEATURES].median()
    )

columns_that_use_mode = MULTI_CATEGORY_FEATURES.copy()
columns_that_use_mode.append('Weekend')
columns_that_use_mode.append(TARGET_COLUMN)

for column in columns_that_use_mode:
    if df_filtered[column].isna().any():
        df_filtered[column] = df_filtered[column].fillna(
            df_filtered[column].mode(dropna=True)[0]
        )

missing_values_after_imputation = df_filtered.isna().sum()

print("\n--- 2a) Filtering ---")
print(f"Removed {initial_shape[0] - df_filtered.shape[0]} rows where ProductRelated was 0.")
print(f"New dataset shape: {df_filtered.shape}")

print("\n--- 2b) Encoding Categorical Variables ---")
df_prepared = df_filtered.copy()

# Convert booleans to 0/1, then one-hot encode the multi-category columns.
df_prepared['Weekend'] = df_prepared['Weekend'].astype(int)
df_prepared[TARGET_COLUMN] = df_prepared[TARGET_COLUMN].astype(int)

df_encoded = pd.get_dummies(
    df_prepared,
    columns=MULTI_CATEGORY_FEATURES,
    drop_first=True,
)
print(f"Data shape after One-Hot Encoding: {df_encoded.shape}")

print("\n--- 2c) Normalization ---")
scaler = StandardScaler()
df_encoded[NUMERICAL_FEATURES] = scaler.fit_transform(df_encoded[NUMERICAL_FEATURES])
print("StandardScaler successfully applied to numerical features.")

print("\n--- 2d) Bin Distribution ---")
product_related_bins = pd.cut(df_filtered['ProductRelated'], bins=5)
print("Bin distribution for ProductRelated:")
print(product_related_bins.value_counts().sort_index())

print("\n--- 2e) Missing Values Handling ---")
print("Missing values per column before imputation:")
print(missing_values_before_imputation)
print(
    "\nMedian would be used for numerical columns and mode for categorical columns "
    "because median is robust to outliers while mode preserves the most common category."
)
if missing_values_before_imputation.sum() > 0:
    print("\nMissing values per column after imputation:")
    print(missing_values_after_imputation)
    print("Median/mode imputation was applied before encoding and scaling.")
else:
    print("This dataset has no missing values, so no imputation step was required.")

# Remove redundant numerical features and weak categorical features.
print("\n--- 2f) Correlation Analysis (Numerical Features) ---")
numerical_cols_for_selection = NUMERICAL_FEATURES.copy()
correlation_matrix = df_filtered[NUMERICAL_FEATURES].corr(method='pearson').abs()
high_correlation_pairs = []
removed_numerical_features = []

for left_index in range(len(numerical_cols_for_selection)):
    left_column = numerical_cols_for_selection[left_index]

    for right_index in range(left_index + 1, len(numerical_cols_for_selection)):
        right_column = numerical_cols_for_selection[right_index]
        correlation_value = correlation_matrix.loc[left_column, right_column]

        if correlation_value > CORRELATION_THRESHOLD:
            pair_details = {
                'kept_feature': left_column,
                'removed_feature': right_column,
                'correlation': correlation_value,
            }
            high_correlation_pairs.append(pair_details)

            if right_column not in removed_numerical_features:
                removed_numerical_features.append(right_column)

if high_correlation_pairs:
    print(f"Highly correlated pairs found (threshold > {CORRELATION_THRESHOLD}):")
    for pair_details in high_correlation_pairs:
        kept_feature = pair_details['kept_feature']
        removed_feature = pair_details['removed_feature']
        correlation_value = pair_details['correlation']
        print(f"- {kept_feature} vs {removed_feature}: {correlation_value:.3f}")
else:
    print("No highly correlated numerical features found.")

print("\n--- 2f) Chi-square Test (Categorical Features) ---")
chi_square_results = []
removed_categorical_features = []

for column in CATEGORICAL_FEATURES_FOR_SELECTION:
    contingency_table = pd.crosstab(df_filtered[column], df_filtered[TARGET_COLUMN])
    chi_square_stat, p_value, _, _ = chi2_contingency(contingency_table)
    result_details = {
        'feature': column,
        'chi_square': chi_square_stat,
        'p_value': p_value,
    }
    chi_square_results.append(result_details)

    if p_value > 0.05:
        removed_categorical_features.append(column)

for result_details in chi_square_results:
    feature_name = result_details['feature']
    chi_square_stat = result_details['chi_square']
    p_value = result_details['p_value']
    print(f"- {feature_name}: chi2 = {chi_square_stat:.3f}, p-value = {p_value:.6f}")

selected_features = []
for column in df_filtered.columns:
    if column not in removed_numerical_features and column not in removed_categorical_features:
        selected_features.append(column)

print("\n--- 2g) Removed Features and Justification ---")
if removed_numerical_features:
    print("Removed numerical features:")
    for removed_feature in removed_numerical_features:
        kept_feature = ""
        correlation_value = 0

        for pair_details in high_correlation_pairs:
            if pair_details['removed_feature'] == removed_feature:
                kept_feature = pair_details['kept_feature']
                correlation_value = pair_details['correlation']
                break

        print(
            f"- {removed_feature}: removed because it is highly correlated with "
            f"{kept_feature} (correlation = {correlation_value:.3f})"
        )
else:
    print("Removed numerical features: None")

if removed_categorical_features:
    print("\nRemoved categorical features:")
    for removed_feature in removed_categorical_features:
        chi_square_stat = 0
        p_value = 0

        for result_details in chi_square_results:
            if result_details['feature'] == removed_feature:
                chi_square_stat = result_details['chi_square']
                p_value = result_details['p_value']
                break

        print(
            f"- {removed_feature}: removed because chi2 = {chi_square_stat:.3f} "
            f"and p-value = {p_value:.6f} (> 0.05)"
        )
else:
    print("\nRemoved categorical features: None")

print("\nSelected features after feature selection:")
print(selected_features)

# Balance the target classes with stratified sampling.
print("\n--- 2h) Class Imbalance Analysis and Stratified Sampling ---")
class_order = [False, True]
class_distribution_before = pd.Series(index=class_order, dtype='int64')
for class_value in class_order:
    class_count = int(len(df_filtered[df_filtered[TARGET_COLUMN] == class_value]))
    class_distribution_before.loc[class_value] = class_count

class_distribution_before = class_distribution_before.astype(int)

class_percentage_before = class_distribution_before / len(df_filtered) * 100
imbalance_ratio = class_distribution_before[False] / class_distribution_before[True]

print("Class distribution before sampling:")
print(class_distribution_before)
print("\nClass percentage before sampling:")
print(class_percentage_before.round(2))
print(f"\nImbalance ratio (False:True): {imbalance_ratio:.2f}:1")

minority_class_size = int(class_distribution_before.min())
stratified_samples = []

for class_value in class_order:
    class_subset = df_filtered[df_filtered[TARGET_COLUMN] == class_value]
    sampled_subset = class_subset.sample(n=minority_class_size, random_state=RANDOM_SEED)
    stratified_samples.append(sampled_subset)

df_sampled = pd.concat(stratified_samples).sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(drop=True)

print("\nSampling method used: Stratified sampling by Revenue class.")
print(f"Rows sampled from each class: {minority_class_size}")

print("\n--- 2i) Class Distribution Before vs After Sampling ---")
class_distribution_after = pd.Series(index=class_order, dtype='int64')
for class_value in class_order:
    class_count = int(len(df_sampled[df_sampled[TARGET_COLUMN] == class_value]))
    class_distribution_after.loc[class_value] = class_count

class_distribution_after = class_distribution_after.astype(int)

class_percentage_after = class_distribution_after / len(df_sampled) * 100

print("Before sampling:")
print(class_distribution_before)
print("\nAfter stratified sampling:")
print(class_distribution_after)

print("\nPercentage before sampling:")
print(class_percentage_before.round(2))
print("\nPercentage after sampling:")
print(class_percentage_after.round(2))
print("\n=== Part 3: Data Visualization ===")

# Save all charts to one folder.
visualization_dir = "visualizations"
os.makedirs(visualization_dir, exist_ok=True)

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

print("\n--- 3d) Correlation Heatmap ---")
heatmap_corr = df_filtered[numerical_cols_for_selection].corr(method='pearson')
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

print("\n=== Part 4: Insight Discovery ===")

insight_df = df_filtered

# PageValues is much higher in purchasing sessions, so it is the clearest target-related signal.
print("\n--- 4a) Statistically Supported Insight About the Target ---")
pagevalues_true = insight_df[insight_df[TARGET_COLUMN] == True]['PageValues']
pagevalues_false = insight_df[insight_df[TARGET_COLUMN] == False]['PageValues']
pagevalues_ttest = ttest_ind(pagevalues_true, pagevalues_false, equal_var=False)

print(f"Sessions with Revenue = True: {len(pagevalues_true)}")
print(f"Sessions with Revenue = False: {len(pagevalues_false)}")
print(f"Average PageValues when Revenue = True: {pagevalues_true.mean():.2f}")
print(f"Average PageValues when Revenue = False: {pagevalues_false.mean():.2f}")
print(f"Median PageValues when Revenue = True: {pagevalues_true.median():.2f}")
print(f"Median PageValues when Revenue = False: {pagevalues_false.median():.2f}")
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

# Check interaction patterns only on segments that are large enough to be trustworthy.
print("\n--- 4b) Interaction-Based Insight ---")
visitor_month_summary = insight_df.groupby(['VisitorType', 'Month'])[TARGET_COLUMN].agg(
    ['mean', 'count', 'sum']
)
visitor_month_summary = visitor_month_summary.reset_index()
visitor_month_summary = visitor_month_summary.sort_values(
    ['mean', 'count'],
    ascending=[False, False],
)
meaningful_segments = visitor_month_summary[visitor_month_summary['count'] >= MINIMUM_SEGMENT_SIZE]
top_segment_row = meaningful_segments.iloc[0]
top_visitor_type = top_segment_row['VisitorType']
top_month = top_segment_row['Month']
top_conversion_rate = float(top_segment_row['mean'])
top_segment_count = int(top_segment_row['count'])
top_segment_conversions = int(top_segment_row['sum'])
overall_conversion_rate = insight_df[TARGET_COLUMN].mean()
visitor_type_rate = insight_df[insight_df['VisitorType'] == top_visitor_type][TARGET_COLUMN].mean()
month_rate = insight_df[insight_df['Month'] == top_month][TARGET_COLUMN].mean()
top_segment_mask = (
    (insight_df['VisitorType'] == top_visitor_type) &
    (insight_df['Month'] == top_month)
)
top_segment_table = pd.crosstab(top_segment_mask, insight_df[TARGET_COLUMN])
top_segment_chi2 = chi2_contingency(top_segment_table)

print(
    f"Top interaction segment: VisitorType = {top_visitor_type}, Month = {top_month}"
)
print(f"Segment sessions: {top_segment_count}")
print(f"Segment conversions: {top_segment_conversions}")
print(f"Segment conversion rate: {top_conversion_rate:.2%}")
print(f"{top_visitor_type} overall conversion rate: {visitor_type_rate:.2%}")
print(f"{top_month} overall conversion rate: {month_rate:.2%}")
print(f"Overall conversion rate: {overall_conversion_rate:.2%}")
print(f"Segment vs rest chi-square p-value: {top_segment_chi2[1]:.3e}")
print(
    "Insight: New visitors in November convert better than either the visitor-type baseline or the "
    "month baseline, which makes this a meaningful interaction pattern."
)

# SpecialDay is lower in purchasing sessions, which makes it an unexpected result.
print("\n--- 4c) Counter-Intuitive Finding ---")
specialday_true = insight_df[insight_df[TARGET_COLUMN] == True]['SpecialDay']
specialday_false = insight_df[insight_df[TARGET_COLUMN] == False]['SpecialDay']
specialday_ttest = ttest_ind(specialday_true, specialday_false, equal_var=False)

print(f"Average SpecialDay when Revenue = True: {specialday_true.mean():.4f}")
print(f"Average SpecialDay when Revenue = False: {specialday_false.mean():.4f}")
print(f"Share of Revenue = True sessions with SpecialDay > 0: {specialday_true.gt(0).mean():.2%}")
print(f"Share of Revenue = False sessions with SpecialDay > 0: {specialday_false.gt(0).mean():.2%}")
print(f"Welch t-test p-value: {specialday_ttest.pvalue:.3e}")
print(
    "Insight: Purchases are less associated with SpecialDay periods than non-purchases, which is an "
    "unexpected pattern in this dataset."
)

# Turn the strongest patterns into one practical recommendation.
print("\n--- 4d) Business Recommendation ---")
print(f"Average PageValues when Revenue = True: {pagevalues_true.mean():.2f}")
print(f"Average PageValues when Revenue = False: {pagevalues_false.mean():.2f}")
print(f"{top_visitor_type} conversion rate in {top_month}: {top_conversion_rate:.2%}")
print(f"Overall conversion rate: {overall_conversion_rate:.2%}")
print(f"Segment vs rest chi-square p-value: {top_segment_chi2[1]:.3e}")
print(
    "Recommendation: Improve product-page clarity, offers, and checkout cues that are associated with "
    "higher PageValues, and prioritize acquisition campaigns for new visitors during November-like "
    "high-conversion periods."
)

print("\n=== Part 5: Feature Engineering ===")

# Measure average time spent per product page.
print("\n--- 5a) New Feature Creation ---")
df_filtered[ENGINEERED_FEATURE] = (
    df_filtered['ProductRelated_Duration'] / df_filtered['ProductRelated']
)
df_prepared[ENGINEERED_FEATURE] = df_filtered[ENGINEERED_FEATURE]

engineered_scaler = StandardScaler()
df_encoded[ENGINEERED_FEATURE] = engineered_scaler.fit_transform(
    df_prepared[[ENGINEERED_FEATURE]]
)
df_sampled[ENGINEERED_FEATURE] = (
    df_sampled['ProductRelated_Duration'] / df_sampled['ProductRelated']
)

if ENGINEERED_FEATURE not in selected_features:
    selected_features.append(ENGINEERED_FEATURE)

print(f"New feature created: {ENGINEERED_FEATURE}")
print("Definition: ProductRelated_Duration / ProductRelated")
print("Added to datasets: df_filtered, df_prepared, df_encoded, and df_sampled")

# This feature adds depth of engagement, not just visit count.
print("\n--- 5b) Why This Feature May Improve Classification ---")
avg_time_true = df_filtered[df_filtered[TARGET_COLUMN] == True][ENGINEERED_FEATURE]
avg_time_false = df_filtered[df_filtered[TARGET_COLUMN] == False][ENGINEERED_FEATURE]
avg_time_ttest = ttest_ind(avg_time_true, avg_time_false, equal_var=False)

print(f"Average {ENGINEERED_FEATURE} when Revenue = True: {avg_time_true.mean():.2f}")
print(f"Average {ENGINEERED_FEATURE} when Revenue = False: {avg_time_false.mean():.2f}")
print(f"Median {ENGINEERED_FEATURE} when Revenue = True: {avg_time_true.median():.2f}")
print(f"Median {ENGINEERED_FEATURE} when Revenue = False: {avg_time_false.median():.2f}")
print(f"Welch t-test p-value: {avg_time_ttest.pvalue:.3e}")
print(
    "Avg_Product_Time may improve future classification models because it captures engagement quality per "
    "product page rather than only total browsing volume. That helps distinguish shallow browsing from "
    "more deliberate shopping behavior."
)
