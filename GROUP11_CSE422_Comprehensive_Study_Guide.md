# CSE422 Group 11 — Complete Project Guide
### Hotel Booking Cancellation Prediction Using Machine Learning
**Members:** Afnan Jahan Prithila (21201675) | B.M. Muktadir Wafi (22301737)

---

# PART 1: FILES IN THE FOLDER — WHAT EACH ONE IS

## File 1: `11.csv` — Your Dataset
**Size:** 119,392 rows × 32 columns | **Size on disk:** ~16.8 MB

This is the **Hotel Booking Demand** dataset. Every row represents one hotel booking. It was collected from a Portuguese hotel management company and contains real booking records from 2015–2017 spanning two property types: a **Resort Hotel** and a **City Hotel**.

**Why this is your dataset:** The spreadsheet screenshot shows Group 11 → `11.csv` (confirmed by your group info: Afnan Jahan Prithila 21201675 and B.M. Muktadir Wafi 22301737).

---

## File 2: `422 Project Report Template.pdf`
**Purpose:** The official guideline from your faculty defining exactly what must be in the project.

**What it contains:**
- Cover page, Table of Contents, and Page number requirements
- 9 numbered sections that define both the code AND the report structure
- Specification that Neural Network is mandatory + at least 2 more models
- Report format: IEEE journal template (Overleaf preferred)
- Link: `https://www.overleaf.com/latex/templates/labreport-ieee-journal-paper-template/ypbfnkbvsyhw`

---

## File 3: `eda_cse422_lab.py`
**Purpose:** Lab reference file for Exploratory Data Analysis (EDA).

**What it contains:** 472 lines of annotated EDA code demonstrating:
- `dataset.info()`, `dataset.head()`, `dataset.shape` — basic exploration
- `select_dtypes(include='number')` — separating numeric features
- `numerical_data.hist()` — histograms of distributions
- `sns.boxplot()` — outlier visualization
- `numerical_data.corr()` with `sns.heatmap()` — correlation analysis
- Pearson, Spearman, and Kendall correlation methods
- Class imbalance checking with `groupby("OUTCOME").size()`
- Density plots for feature distributions

**Original dataset in this lab:** Vehicle insurance dataset (`modified_train.csv`). NOT your dataset — this is just a teaching example.

---

## File 4: `introduction_to_ml_&_preprocessing.py`
**Purpose:** Lab reference for the ML pipeline and preprocessing steps.

**What it contains:** 423 lines covering:
- Statistical Learning Framework theory (domain set, label set, hypothesis)
- Traditional programming vs Machine Learning comparison
- ML Pipeline steps (collect → preprocess → split → train → evaluate)
- Null value handling: `dropna()`, `fillna()`, `SimpleImputer(strategy='mean')`
- Categorical encoding: `LabelEncoder()`, `.map()`, `pd.get_dummies()` (one-hot)
- Feature selection: dropping highly correlated columns using heatmap
- Feature scaling: `MinMaxScaler`, `StandardScaler`, `RobustScaler`
- KNN example showing impact of scaling

**Key equation shown:** StandardScaler: `z = (X - mean) / σ` | MinMax: `(X - X_min) / (X_max - X_min)`

---

## File 5: `model_evaluation.py`
**Purpose:** Lab reference for evaluating ML models.

**What it contains:** 332 lines covering:
- ROC (Receiver Operating Characteristic) Curve theory
- AUC (Area Under Curve): AUC=1 is perfect, AUC=0.5 is random guessing
- Binary confusion matrix plotting with `ConfusionMatrixDisplay`
- Multiclass confusion matrix (10-class digits dataset)
- Precision, Recall, F1-Score formulas and interpretation
- `classification_report()` for detailed per-class metrics
- Linear Regression on California housing data (for regression context)

---

## File 6: `cse422_neuralnetworklab.py`
**Purpose:** Lab reference for Neural Networks.

**What it contains:** 336 lines covering:
- Forward propagation math: `Z^[l] = W^[l]·A^[l-1] + B^[l]`, `A^[l] = f(Z^[l])`
- Backpropagation math: gradient descent weight updates
- Manual implementation of: `ReLU` class, `MeanSquareError` loss, `GradientDescent` optimizer, `NeuronLayer`, `Model` class
- Boston Housing dataset regression example (manual NN)
- TensorFlow/Keras Sequential model (MNIST digit classification)
- `Adam` optimizer, `SparseCategoricalCrossentropy` loss

---

## File 7 & 8: `cse_422_lab_5_part_1_[k_means_clustering].py` (both copies identical)
**Purpose:** Lab reference for K-Means unsupervised learning.

**What it contains:** 290 lines covering:
- Supervised vs Unsupervised Learning analogy (ABCs with teacher vs sorting LEGO blocks)
- K-Means algorithm theory: assigns data to nearest centroid, recomputes centroids
- Elbow method to find optimal k
- Cluster visualization
- WCSS (Within-Cluster Sum of Squares) = inertia

---

## File 9: `modified_train.csv`
**Purpose:** Dataset used in the EDA lab (`eda_cse422_lab.py`). NOT your dataset.

**What it contains:** Vehicle insurance data with features like CREDIT_SCORE, VEHICLE_OWNERSHIP, ANNUAL_MILEAGE, DUIS, PAST_ACCIDENTS, OUTCOME. About 105,000 rows.

---

# PART 1.5: LAB CODE MAPPING — HOW LAB CODES WERE ADAPTED INTO YOUR PROJECT NOTEBOOK

To ensure full compliance with the course requirements and CSE422 lab curriculum, all standard code paradigms, function calls, and analytical structures taught in the 5 lab sessions were adapted directly to your dataset (`11.csv`). Below is the complete verification and mapping of how every lab file's code was used:

### 1. Code from `eda_cse422_lab.py` → Used in Sections 2 & 3
- **Data Inspection:** `df.info()`, `df.describe().T`, and `df.shape` were used directly in Section 2 and Section 3.
- **Separating Column Types:** `select_dtypes(include=['int64', 'float64'])` and `select_dtypes(include=['object'])` from the lab were used to split numerical and categorical features in Section 2.
- **Class Imbalance Check:** The lab's `groupby("OUTCOME").size()` logic was implemented using `df['is_canceled'].value_counts()` and visualized with a dual-colored bar chart with exact percentage annotations in Section 2.
- **Distribution Analysis:** The lab's `numerical_data.hist(bins=..., figsize=...)` was adapted into a clean 3x3 subplot grid of histograms with custom binning and formatting in Section 3.
- **Outlier Detection:** `sns.boxplot()` with custom outlier marker styling (`flierprops`) from the EDA lab was implemented in Section 3 across all key numerical columns.
- **Correlation Heatmap:** The Pearson correlation analysis with upper-triangle masking (`np.triu`) and `coolwarm` colormap was adapted directly from the EDA lab in Section 3.
- **Multi-Method Correlation:** Pearson, Spearman, and Kendall correlation methods demonstrated in the EDA lab were applied side-by-side to assess linear and non-linear relationships with `is_canceled` in Section 3.

---

### 2. Code from `introduction_to_ml_&_preprocessing.py` → Used in Sections 4 & 5
- **Null Value Checking:** `df.isnull().sum()` and missing value filtering from the preprocessing lab were implemented in Sections 3 & 4.
- **Dropping Irrelevant / Leakage Columns:** The lab's column dropping pattern `df.drop(['BIN', 'BBL', ...], axis=1)` was used to drop `reservation_status`, `reservation_status_date` (data leakage), and `country` (high cardinality) in Section 4.
- **Imputation Strategy:** The lab's `fillna()` / `SimpleImputer` methodology was used to impute `agent` (0), `company` (0), and `children` (0) in Section 4.
- **Categorical Encoding:** `LabelEncoder().fit_transform()` from the preprocessing lab was used across all nominal categorical columns in Section 4.
- **Duplicate Removal:** `df.drop_duplicates()` was applied in Section 4.
- **Train/Test Splitting:** `train_test_split(..., test_size=0.20, stratify=y, random_state=42)` was implemented in Section 5, maintaining the stratified class ratios taught in the lab.
- **Feature Scaling:** `StandardScaler().fit_transform(X_train)` and `.transform(X_test)` was implemented in Section 5 exactly as taught in the lab to prevent test data leakage.

---

### 3. Code from `cse422_neuralnetworklab.py` → Used in Section 6.5
- **Sequential Model Architecture:** The Keras `Sequential` API with stacked `Dense` layers was implemented in Section 6.5 (`Dense(128)` → `Dense(64)` → `Dense(32)` → `Dense(1)`).
- **Activation Functions:** `ReLU` activation for hidden layers and `Sigmoid` activation for the final binary output unit were used as demonstrated in the Neural Network lab.
- **Compilation:** `model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])` was implemented matching the lab structure.
- **Training & Mini-batching:** `model.fit(X_train_scaled, y_train, epochs=50, batch_size=512, validation_split=0.1, ...)` was executed with mini-batch updates.
- **Training History & Learning Curves:** Loss and accuracy history curves were extracted from `history.history` and plotted in Figure 9 as shown in the lab.

---

### 4. Code from `model_evaluation.py` → Used in Sections 6 & 7
- **Classification Report:** `classification_report(y_true, y_pred)` providing per-class Precision, Recall, and F1-score was embedded into the `evaluate_model()` function in Section 6.
- **Confusion Matrix:** `confusion_matrix()` and `ConfusionMatrixDisplay()` were used to plot individual confusion matrices for all 5 models in Section 7.
- **ROC Curve & AUC Score:** `roc_curve(y_true, y_prob)` and `auc(fpr, tpr)` were implemented in Section 7 to calculate AUC and plot all 5 model ROC curves on a single comparison chart with the random diagonal baseline.
- **Model Comparison Table:** Summary metrics table and comparative bar charts were constructed in Section 7 to benchmark all models against each other.

---

### 5. Code from `cse_422_lab_5_part_1_[k_means_clustering].py` → Used in Section 8
- **Unsupervised Transformation:** Target variable `is_canceled` was stripped from the feature matrix to treat the problem in an unsupervised manner in Section 8.
- **K-Means Model:** `KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, random_state=42)` was implemented in Section 8.
- **Elbow Method:** WCSS / `inertia_` was recorded across $k=1 \dots 10$ and plotted in Figure 14 to mathematically identify the optimal elbow at $k=2$.
- **Cluster Visualization via PCA:** `PCA(n_components=2)` was used to project the 27-dimensional feature space into 2D scatter plots, comparing unsupervised K-Means clusters against ground truth labels in Section 8.
- **Cross-Tabulation Analysis:** `pd.crosstab(cluster_labels, y_train)` was used to profile cluster composition and quantify alignment with actual cancellation rates.

---

# PART 2: YOUR DATASET — `11.csv` EXPLAINED IN DETAIL

## The 32 Columns

| # | Column | Type | Explanation |
|---|--------|------|-------------|
| 1 | `hotel` | Categorical | "Resort Hotel" or "City Hotel" — two property types |
| 2 | **`is_canceled`** | **Binary (0/1)** | **TARGET** — 0 = booking kept, 1 = booking canceled |
| 3 | `lead_time` | Numerical | Days between booking date and arrival date |
| 4 | `arrival_date_year` | Numerical | Year of arrival (2015, 2016, 2017) |
| 5 | `arrival_date_month` | Categorical | Month name ("July", "August", etc.) |
| 6 | `arrival_date_week_number` | Numerical | Week number (1–53) |
| 7 | `arrival_date_day_of_month` | Numerical | Day of month (1–31) |
| 8 | `stays_in_weekend_nights` | Numerical | Weekend nights booked |
| 9 | `stays_in_week_nights` | Numerical | Weekday nights booked |
| 10 | `adults` | Numerical | Number of adult guests |
| 11 | `children` | Numerical | Number of children (has 4 NULLs) |
| 12 | `babies` | Numerical | Number of babies |
| 13 | `meal` | Categorical | Meal plan: BB=Bed&Breakfast, FB=Full Board, HB=Half Board, SC=Self-catering |
| 14 | `country` | Categorical | Guest country code (178 unique values — DROPPED) |
| 15 | `market_segment` | Categorical | Booking source: Online TA, Offline TA/TO, Direct, Corporate, Groups, Aviation |
| 16 | `distribution_channel` | Categorical | How booking was distributed: TA/TO, Direct, Corporate, GDS |
| 17 | `is_repeated_guest` | Binary | 1 if returning guest, 0 if first-time |
| 18 | `previous_cancellations` | Numerical | Number of previous cancellations |
| 19 | `previous_bookings_not_canceled` | Numerical | Previous successful bookings |
| 20 | `reserved_room_type` | Categorical | Room type originally reserved (A, B, C, ...) |
| 21 | `assigned_room_type` | Categorical | Actual room type assigned (may differ if upgraded/downgraded) |
| 22 | `booking_changes` | Numerical | Number of amendments made to booking |
| 23 | `deposit_type` | Categorical | No Deposit / Non Refund / Refundable |
| 24 | `agent` | Numerical | Travel agent ID (NULL = no agent, 16,340 NULLs) |
| 25 | `company` | Numerical | Company ID (NULL = no company, 112,593 NULLs) |
| 26 | `days_in_waiting_list` | Numerical | Days booking was on waitlist |
| 27 | `customer_type` | Categorical | Transient, Transient-Party, Contract, Group |
| 28 | `adr` | Numerical | Average Daily Rate (price per night in €) |
| 29 | `required_car_parking_spaces` | Numerical | Parking spaces requested |
| 30 | `total_of_special_requests` | Numerical | Special requests count (e.g., high floor, twin beds) |
| 31 | **`reservation_status`** | **Categorical** | **DATA LEAKAGE — DROPPED** — Check-Out/Canceled/No-Show |
| 32 | **`reservation_status_date`** | **Date** | **DATA LEAKAGE — DROPPED** — Date of status update |

## Why Classification (Not Regression)?
`is_canceled` takes only two values: **0** or **1** — discrete binary labels. We are predicting a **category**, not a continuous number. Regression would predict something like 0.67 which has no meaningful interpretation. Classification is the correct framing.

## Class Distribution
- Class 0 (Not Canceled): ~75,166 (~62.96%)
- Class 1 (Canceled): ~44,224 (~37.04%)
- **Slight imbalance** — requires stratified train/test split

---

# PART 3: LINE-BY-LINE CODE EXPLANATION

> **How to read this section:** Each heading shows the notebook **Cell number** (as it appears in Colab) and how many lines it contains. Every single line is then explained individually with its logic, purpose, and any relevant math.

---

## CELL 2 — Global Imports & Global Settings (30 lines)

**What this cell does:** Loads every library the project needs in one place. This is placed at the top so all sections below can use these tools without re-importing.

```
Line 1:  import pandas as pd
```
`pandas` is Python's primary data table library. It gives us the `DataFrame` object — essentially a 2D labeled spreadsheet in memory. `pd` is the universal alias used across the entire ML community.

```
Line 2:  import numpy as np
```
`numpy` provides N-dimensional arrays and mathematical operations (matrix math, random numbers, statistical functions). Every other ML library internally uses NumPy arrays. `np` is the universal alias.

```
Line 3:  import matplotlib.pyplot as plt
```
`matplotlib.pyplot` is the foundational plotting engine. `plt` is the standard alias. All 15 figures in this project ultimately go through matplotlib.

```
Line 4:  import matplotlib.ticker as mticker
```
`mticker` provides axis formatters. Used in this project to:
- Format y-axis as percentages: `mticker.PercentFormatter(xmax=1.0)` → shows `0.85` as `85%`
- Format large numbers with commas: `mticker.FuncFormatter(lambda x, _: f'{int(x):,}')` → shows `75166` as `75,166`

```
Line 5:  import seaborn as sns
```
`seaborn` is a statistical visualization library built on top of matplotlib. Much simpler API for heatmaps, boxplots, and distribution plots. `sns` is the standard alias.

```
Line 6:  import warnings
Line 7:  warnings.filterwarnings('ignore')
```
Suppresses non-critical Python warnings (e.g., convergence hints, deprecation notices from sklearn). Keeps Colab output clean. Real errors (tracebacks) are NOT suppressed — only informational warnings.

```
Line 8:  from sklearn.preprocessing import LabelEncoder, StandardScaler
```
- `LabelEncoder`: Converts text categories → integers. Example: `{'City Hotel':0, 'Resort Hotel':1}`.
- `StandardScaler`: Normalizes features using z-score formula: `z = (x − μ) / σ`. After scaling, each feature has mean≈0, std≈1.

```
Line 9:  from sklearn.model_selection import train_test_split
```
Shuffles the dataset and splits it into training and testing portions in one call. Supports stratified splitting to maintain class ratios.

```
Line 10: from sklearn.neighbors import KNeighborsClassifier
Line 11: from sklearn.tree import DecisionTreeClassifier
Line 12: from sklearn.linear_model import LogisticRegression
Line 13: from sklearn.naive_bayes import GaussianNB
```
The four classical classifiers. All share the same scikit-learn API: `.fit()` to train, `.predict()` to get class labels, `.predict_proba()` to get class probabilities.

```
Lines 14–19: from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_score, recall_score, f1_score
)
```
All evaluation tools imported together:
- `accuracy_score` → (TP+TN)/Total
- `classification_report` → Full table of precision/recall/F1 per class
- `confusion_matrix` → 2×2 matrix of TN, FP, FN, TP
- `ConfusionMatrixDisplay` → Renders the confusion matrix as a color heatmap
- `roc_curve` → Returns FPR, TPR arrays across all thresholds
- `auc` → Computes area under any x/y curve using the trapezoidal rule
- `precision_score` → TP / (TP + FP)
- `recall_score` → TP / (TP + FN)
- `f1_score` → 2 × Precision × Recall / (Precision + Recall)

```
Line 20: from sklearn.cluster import KMeans
Line 21: from sklearn.decomposition import PCA
```
- `KMeans`: Unsupervised clustering for Section 8.
- `PCA`: Principal Component Analysis — used to reduce 27 features down to 2 for visualization in Section 8.

```
Line 22: import tensorflow as tf
Line 23: from tensorflow.keras.models import Sequential
Line 24: from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
Line 25: from tensorflow.keras.callbacks import EarlyStopping
```
TensorFlow/Keras deep learning framework:
- `Sequential`: A linear stack of layers.
- `Dense`: Fully connected neural network layer.
- `Dropout`: Regularization technique — randomly drops neurons during training to prevent overfitting.
- `BatchNormalization`: Normalizes layer outputs per mini-batch for stable, fast training.
- `EarlyStopping`: Monitors validation loss and halts training when it stops improving.

```
Line 26: RANDOM_STATE = 42
Line 27: np.random.seed(RANDOM_STATE)
Line 28: tf.random.set_seed(RANDOM_STATE)
```
`RANDOM_STATE = 42` sets a global constant for reproducibility. The value 42 is arbitrary (ML community convention from "The Hitchhiker's Guide to the Galaxy"). Setting seeds means: every run produces identical train/test splits, identical model weight initializations, and identical results. `np.random.seed` seeds NumPy's RNG. `tf.random.set_seed` seeds TensorFlow's separate RNG engine.

```
Line 29: sns.set_style('whitegrid')
Line 30: plt.rcParams['figure.dpi'] = 100
```
- `sns.set_style('whitegrid')`: Sets global plot theme — white background with grey grid lines. Professional and clean.
- `plt.rcParams['figure.dpi'] = 100`: Sets figure resolution to 100 DPI globally. Good quality for Colab display and saving.

---

## CELL 5 — Load Dataset (3 lines)

**What this cell does:** Reads the CSV file into a pandas DataFrame and shows its shape and first 5 rows.

```
Line 1: df = pd.read_csv('11.csv')
```
`pd.read_csv()` reads the file line by line, parses values by comma delimiter, infers data types automatically, and creates a DataFrame with the first row as column names. The entire 119,392-row, 32-column dataset is loaded into RAM as `df`. Note: In the Colab-downloaded version the path is `'11.csv'` (local, since the file is in the same directory). In Google Drive mode it would be the full Drive path.

```
Line 2: print(f'Dataset Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns')
```
`df.shape` is a tuple `(119392, 32)`. `.shape[0]` = row count, `.shape[1]` = column count. `:,` formats the number with comma separators (e.g., 119,392). The `×` symbol is a Unicode multiplication sign for readability.

```
Line 3: df.head()
```
Returns the first 5 rows as a formatted HTML table in Colab. Essential sanity check: confirms the file loaded correctly, shows all 32 column names in context, and reveals value ranges at a glance.

---

## CELL 6 — Data Info (2 lines)

```
Line 1: # Inspect column data types and non-null value counts
Line 2: df.info()
```
`df.info()` prints: column name, non-null count, and dtype for every column. This immediately reveals:
- `agent`: 103,052 non-null out of 119,392 → 16,340 nulls
- `company`: 6,799 non-null → 112,593 nulls
- `children`: 119,388 non-null → 4 nulls
- `reservation_status_date`: stored as `object` (string) despite being a date

---

## CELL 7 — Separate Column Types (6 lines)

```
Line 1: # Separate quantitative (numerical) and qualitative (categorical) columns
Line 2: numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
Line 3: categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
```
`select_dtypes(include=['int64', 'float64'])` filters only integer and float columns. `select_dtypes(include=['object'])` filters only string columns. `.columns` returns a pandas Index. `.tolist()` converts it to a regular Python list. This separation is used throughout: EDA plots apply different techniques to each type, encoding only applies to categoricals, and correlation only works on numericals.

```
Line 4: (empty line for readability)
Line 5: print(f'Numerical features ({len(numerical_cols)}):', numerical_cols)
Line 6: print(f'\nCategorical features ({len(categorical_cols)}):', categorical_cols)
```
Prints both lists so you can confirm which columns are in each group. `\n` adds a blank line between the two prints. `len()` counts the number of features in each group.

---

## CELL 8 — Target Class Distribution + Figure 1 (34 lines)

**What this cell does:** Counts how many bookings were canceled vs not canceled, prints the percentages, and draws the first figure.

```
Line 1: # Calculate frequency and percentage distribution of the target variable (is_canceled)
Line 2: class_counts = df['is_canceled'].value_counts()
Line 3: class_percentages = df['is_canceled'].value_counts(normalize=True) * 100
```
`df['is_canceled']` selects just the target column as a pandas Series. `.value_counts()` counts each unique value: `{0: 75166, 1: 44224}`. `normalize=True` divides by total to give proportions (0 to 1). `* 100` converts to percentage. Result: `{0: 62.96, 1: 37.04}`.

```
Lines 5–6: print statements
```
Print the absolute count and percentage for each class so the class imbalance is immediately visible in the output.

```
Line 8: # Figure 1: Plot Target Class Balance Bar Chart
Line 9: fig, ax = plt.subplots(figsize=(7, 5))
```
`plt.subplots()` creates a Figure (canvas) and one Axes (plot area). `figsize=(7, 5)` = 7 inches wide, 5 inches tall.

```
Lines 10–16: bars = ax.bar(...)
```
`ax.bar()` draws vertical bars:
- Arg 1: x-axis labels — `['Not Canceled (0)', 'Canceled (1)']`
- Arg 2: bar heights — `class_counts.values` (the raw count numbers)
- `color=['#2ecc71', '#e74c3c']`: green for class 0 (kept), red for class 1 (canceled)
- `edgecolor='black'`: thin black border around each bar for contrast
- `width=0.5`: bars are half the slot width, leaving visual spacing

```
Lines 18–25: Annotation loop
Line 19: for bar, count, pct in zip(bars, class_counts.values, class_percentages.values):
```
`zip()` loops over three sequences simultaneously — the bar objects, the raw counts, and the percentages. For each bar, we place a text label above it.

```
Line 20:     ax.text(
Line 21:         bar.get_x() + bar.get_width() / 2,
Line 22:         bar.get_height() + 500,
Line 23:         f'{count:,}\n({pct:.1f}%)',
Line 24:         ha='center', va='bottom', fontsize=11, fontweight='bold'
Line 25:     )
```
- `bar.get_x() + bar.get_width() / 2`: x-position at horizontal center of bar
- `bar.get_height() + 500`: y-position slightly above bar top (500 units of padding)
- `f'{count:,}\n({pct:.1f}%)'`: count with commas, newline, then percentage with 1 decimal
- `ha='center'`: center-align text horizontally; `va='bottom'`: anchor text bottom at y-position

```
Lines 27–31: Axis formatting
Line 27: ax.set_title('Figure 1: Class Distribution...', fontsize=13, fontweight='bold')
Line 28: ax.set_xlabel('Booking Status', fontsize=12)
Line 29: ax.set_ylabel('Number of Bookings', fontsize=12)
Line 30: ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
Line 31: ax.set_ylim(0, max(class_counts.values) * 1.15)
```
Line 30 formats y-axis tick labels with comma separators (75,000 instead of 75000). Line 31 sets y-axis upper limit to 115% of the tallest bar so labels have space above bars.

```
Line 32: plt.tight_layout()
Line 33: plt.savefig('class_distribution.png', bbox_inches='tight')
Line 34: plt.show()
```
`tight_layout()` automatically adjusts subplot padding to prevent label overlap. `savefig()` saves as PNG to Colab's working directory for inclusion in the Overleaf report. `bbox_inches='tight'` crops extra whitespace from the saved image. `plt.show()` renders the plot inline in Colab.

---

## CELL 10 — Descriptive Statistics (2 lines)

```
Line 1: # Descriptive statistics (mean, std, min, quartiles, max) for numerical features
Line 2: df[numerical_cols].describe().T.round(2)
```
`df[numerical_cols]` selects only numerical columns. `.describe()` computes: count, mean, std, min, 25%, 50%, 75%, max for each column. `.T` transposes — features become rows, statistics become columns. Easier to read with many features. `.round(2)` rounds to 2 decimal places.

**What to look for:** Features where min=0 and 25%=0 (e.g., `previous_cancellations`) are heavily concentrated at zero. High std relative to mean (e.g., `lead_time`, `adr`) indicates wide spread and potential outliers.

---

## CELL 11 — Missing Values Analysis + Figure 2 (13 lines)

```
Line 1: # Identify columns containing null/missing values
Line 2: missing = df.isnull().sum()
Line 3: missing = missing[missing > 0].sort_values(ascending=False)
```
`df.isnull()` creates a boolean DataFrame (True where NaN). `.sum()` adds up True values per column (True=1). `missing[missing > 0]` keeps only columns with at least one null. `.sort_values(ascending=False)` puts the worst offenders first: `company` (112,593), `agent` (16,340), `children` (4).

```
Line 5: # Figure 2: Plot Missing Values Bar Chart
Line 6: fig, ax = plt.subplots(figsize=(8, 4))
Line 7: ax.bar(missing.index, missing.values, color='#e67e22', edgecolor='black')
```
Orange (#e67e22) signals a warning/issue. `missing.index` = column names, `missing.values` = null counts.

```
Lines 8–13: Title, labels, layout, save, show
Line 10: plt.xticks(rotation=20, ha='right')
```
Rotates x-axis column name labels 20° so they don't overlap. `ha='right'` right-aligns the rotated labels.

---

## CELL 12 — Histograms + Figure 3 (22 lines)

**What this cell does:** Shows the distribution shape of 7 key numerical features.

```
Line 1: # Select key numerical features to inspect distribution shape
Lines 2–4: key_num_cols = ['lead_time', 'adr', 'stays_in_weekend_nights',
                'stays_in_week_nights', 'total_of_special_requests',
                'previous_cancellations', 'days_in_waiting_list']
```
These 7 features were selected because they are most relevant to cancellation prediction (based on domain knowledge and correlation analysis).

```
Line 6: # Figure 3: Histograms of Key Numerical Features
Line 7: fig, axes = plt.subplots(3, 3, figsize=(15, 11))
Line 8: axes = axes.flatten()
```
`plt.subplots(3, 3)` creates a 3×3 grid of 9 subplots. `axes` shape is `(3, 3)`. `.flatten()` converts it to a 1D array of shape `(9,)` so we can index as `axes[0]`, `axes[1]`, etc.

```
Lines 10–13: for i, col in enumerate(key_num_cols):
    axes[i].hist(df[col].dropna(), bins=40, color='#3498db', edgecolor='white', linewidth=0.5)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Frequency')
```
`enumerate()` gives `(index, value)` pairs. `df[col].dropna()` removes NaN values before plotting. `bins=40` splits data into 40 equal-width buckets. `edgecolor='white'` adds thin white lines between bars for readability.

```
Line 15: # Hide unused subplot slots in 3x3 grid
Lines 16–17: for j in range(len(key_num_cols), len(axes)):
    axes[j].set_visible(False)
```
We have 7 features but 9 subplot slots. This hides the 8th and 9th empty slots.

```
Lines 19–22: suptitle, tight_layout, savefig, show
Line 19: fig.suptitle(..., y=1.01)
```
`y=1.01` places the super-title slightly above the top of the figure to prevent overlap with subplot titles.

**What each histogram tells you:**
- `lead_time`: Right-skewed — most bookings made close to arrival; long tail means some plan months ahead
- `adr` (Average Daily Rate): Right-skewed with extreme outliers — most rooms €50–150/night, some luxury outliers over €5000
- `previous_cancellations`: Nearly all zero — most guests have never canceled before, making non-zero values strong predictors

---

## CELL 13 — Box Plots + Figure 4 (14 lines)

**What this cell does:** Visualizes the spread and outliers of 5 numerical features using box plots.

```
Lines 1–3: box_cols defined
Line 5: fig, axes = plt.subplots(1, len(box_cols), figsize=(18, 5))
```
`1, len(box_cols)` = 1 row, 5 columns → 5 side-by-side box plots.

```
Lines 6–9: for ax, col in zip(axes, box_cols):
    sns.boxplot(y=df[col], ax=ax, color='#9b59b6',
                flierprops=dict(marker='o', markerfacecolor='red', markersize=3))
    ax.set_title(col, ...)
```
`sns.boxplot()` draws: Q1 (bottom of box), median (line in box), Q3 (top of box), whiskers (up to 1.5×IQR beyond Q1/Q3), and outlier dots beyond whiskers. `y=df[col]` = vertical orientation. `flierprops=dict(...)` styles outlier dots: red circles of size 3.

**Reading box plots:** For `adr` — whiskers will extend to ~400, but there are extreme red dots up to 5000+. These are genuine data points (luxury hotel rooms), not errors. For `days_in_waiting_list` — most values are 0 (box is near zero) with extreme outliers for bookings that waited months on the waiting list.

---

## CELL 14 — Categorical Analysis + Figure 5 (20 lines)

**What this cell does:** Shows what percentage of each categorical group (hotel type, meal plan, etc.) ended up canceling.

```
Line 2: key_cat_cols = ['hotel', 'meal', 'market_segment', 'deposit_type', 'customer_type']
Line 4: fig, axes = plt.subplots(2, 3, figsize=(18, 10))
Line 5: axes = axes.flatten()
```
2×3 grid = 6 subplot slots for 5 features (1 slot hidden at end).

```
Lines 7–14: for i, col in enumerate(key_cat_cols):
    # Normalized crosstab to show percentage cancellation per category
    ct = pd.crosstab(df[col], df['is_canceled'], normalize='index') * 100
    ct.columns = ['Not Canceled (%)', 'Canceled (%)']
    ct.plot(kind='bar', ax=axes[i], color=['#2ecc71', '#e74c3c'], edgecolor='black', rot=25)
    axes[i].set_title(f'{col} vs is_canceled', ...)
    axes[i].set_ylabel('Percentage (%)')
    axes[i].legend(fontsize=9)
```
`pd.crosstab(df[col], df['is_canceled'], normalize='index')` creates a contingency table where each **row sums to 100%**. So for `deposit_type = 'Non Refund'`, we see that ~99.4% of those bookings were canceled. `* 100` converts to percentages. `ct.plot(kind='bar')` draws the grouped bar chart via pandas.

**Key finding:** `deposit_type = 'Non Refund'` cancels at ~99% — the strongest single predictor. Guests who paid a non-refundable deposit tend to cancel when plans change since they can't recover the money either way.

```
Line 16: axes[5].set_visible(False)
```
Hides the 6th (empty) subplot.

---

## CELL 15 — Pearson Correlation Heatmap + Figure 6 (24 lines)

```
Line 2: num_df = df[numerical_cols].copy()
Line 3: pearson_corr = num_df.corr(method='pearson')
```
`.corr(method='pearson')` computes the n×n Pearson correlation matrix where n = number of numerical features. **Pearson formula:** `r = Σ[(xᵢ−x̄)(yᵢ−ȳ)] / [√Σ(xᵢ−x̄)² · √Σ(yᵢ−ȳ)²]`. Range: [−1, +1]. Measures linear relationship strength. Result is a symmetric matrix (r(A,B) = r(B,A)).

```
Line 5: # Mask upper triangle to avoid redundant symmetric information
Line 6: mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
```
`np.ones_like(pearson_corr, dtype=bool)` creates a matrix of True values the same shape as the correlation matrix. `np.triu()` keeps only the upper triangle as True (including diagonal). When passed to `sns.heatmap(mask=...)`, cells where mask=True are hidden. This shows only the lower triangle, eliminating duplicate information since r(A,B) = r(B,A).

```
Lines 8–19: sns.heatmap(...)
    pearson_corr, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', linewidths=0.5, ax=ax, annot_kws={'size': 8}, vmin=-1, vmax=1
```
- `annot=True`: writes the correlation number inside each cell
- `fmt='.2f'`: 2 decimal place format
- `cmap='coolwarm'`: blue = strong negative, white = near zero, red = strong positive
- `vmin=-1, vmax=1`: fixes color scale to the full correlation range — otherwise 0.3 might look intensely red if the actual max is 0.35

---

## CELL 16 — Target Correlation (3 Methods) + Figure 7 (14 lines)

```
Line 2: fig, axes = plt.subplots(1, 3, figsize=(18, 7))
Line 3: methods = ['pearson', 'spearman', 'kendall']
```
Three correlation methods shown side by side for comparison.

```
Lines 5–9: for ax, method in zip(axes, methods):
    # Compute correlation with target, drop target self-correlation, sort values
    corr_target = num_df.corr(method=method)[['is_canceled']].drop('is_canceled').sort_values('is_canceled', ascending=False)
    sns.heatmap(corr_target, annot=True, fmt='.3f', cmap='RdYlGn', vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(...)
```
`num_df.corr(method=method)` computes the full correlation matrix using the specified method. `[['is_canceled']]` — double brackets keep it as a DataFrame (single bracket returns a Series). `.drop('is_canceled')` removes the "feature vs itself = 1.0" row. `.sort_values('is_canceled', ascending=False)` sorts so most positively correlated features are at the top.

**Three correlation methods explained:**
- **Pearson:** Linear relationships only. Sensitive to outliers. Best for normally distributed continuous data.
- **Spearman:** Applies Pearson on the **ranks** of values. Detects monotonic (consistently increasing/decreasing) relationships even if nonlinear. More robust to outliers.
- **Kendall Tau:** `τ = (concordant pairs − discordant pairs) / C(n,2)`. More conservative than Spearman. Better when there are many tied values.

**Key findings from these plots:**
- `previous_cancellations` → High positive correlation → Past cancelers cancel again
- `total_of_special_requests` → Negative correlation → Engaged guests (who request things) are committed to staying
- `required_car_parking_spaces` → Negative → Guests needing parking are more likely to arrive

---

## CELL 18 — Data Preprocessing (36 lines)

**What this cell does:** Cleans and transforms the raw data into model-ready format in 7 explicit steps.

```
Line 1: # Create an independent working copy
Line 2: data = df.copy()
```
`.copy()` creates a **deep copy** — completely independent from `df` in memory. Changes to `data` do NOT affect `df`. This is essential so: (1) if preprocessing corrupts something, you don't need to re-read the 16 MB CSV; (2) you can compare `data` vs `df` at any point.

```
Line 4: # Step 1: Drop Data Leakage features (reservation_status directly reveals cancellation outcome)
Line 5: data.drop(columns=['reservation_status', 'reservation_status_date'], inplace=True)
```
**Why these are leakage:** `reservation_status` contains the string `"Canceled"` for canceled bookings — it directly encodes the answer we're predicting. At booking time in production, this value doesn't exist yet. `reservation_status_date` is the date the cancellation was recorded — same problem. `inplace=True` modifies `data` directly instead of returning a new DataFrame.

```
Line 7: # Step 2: Drop high-cardinality nominal feature (country has 178 categories -> curse of dimensionality)
Line 8: data.drop(columns=['country'], inplace=True)
```
`country` has 178 unique values. Label encoding implies false ordinal relationships (e.g., Albania=0 < Austria=3 is meaningless). One-hot encoding would create 178 new binary columns — massive memory use and sparse matrices. Decision: drop it.

```
Line 10: # Step 3: Handle missing values (0 denotes no agent, no company, 0 children)
Line 11: data['agent'].fillna(0, inplace=True)
Line 12: data['company'].fillna(0, inplace=True)
Line 13: data['children'].fillna(0, inplace=True)
```
In the original dataset documentation, NULL in `agent` means "no travel agent was used." NULL in `company` means "individual booking, no company." Both are semantically 0 (not a real agent/company ID). `children` NULLs (only 4) most likely mean 0 children. `fillna(0)` is semantically correct for all three.

```
Line 15: # Step 4: Remove exact duplicate rows
Line 16: data.drop_duplicates(inplace=True)
```
Removes rows that are completely identical across all remaining columns. Prevents the model from seeing the same booking twice, which would artificially inflate confidence.

```
Lines 18–24: # Step 5: Encode nominal categorical features
categorical_features = data.select_dtypes(include='object').columns.tolist()
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
```
After dropping leakage and country, remaining string columns are: `hotel`, `arrival_date_month`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `assigned_room_type`, `deposit_type`, `customer_type`. For each: `LabelEncoder()` creates a new encoder instance. `.astype(str)` converts all values to string first (prevents errors if NaN slipped through). `.fit_transform()` maps each unique string to an integer (sorted alphabetically). `label_encoders[col] = le` stores each encoder — needed if you want to decode predictions later using `le.inverse_transform()`.

```
Lines 26–29: # Step 6: Convert float IDs to integer
data['agent'] = data['agent'].astype(int)
data['company'] = data['company'].astype(int)
data['children'] = data['children'].astype(int)
```
After `fillna(0)`, these columns became float64 (pandas stores NaN as float, so the whole column becomes float). After filling, we convert back to int64 — cleaner, more memory-efficient, and semantically correct (IDs and counts should be integers).

```
Lines 31–33: # Step 7: Separate feature matrix (X) and target label vector (y)
X = data.drop(columns=['is_canceled'])
y = data['is_canceled']
```
`X` is the feature matrix — all 27 remaining columns after preprocessing. `y` is the target vector — just the `is_canceled` column. Shape after preprocessing: `X` ≈ (118,898 × 27), `y` ≈ (118,898,).

```
Lines 35–36: print statements for shape confirmation
```
Confirms the expected dimensions after all preprocessing steps.

---

## CELL 20 — Train-Test Split + Scaling (15 lines)

```
Line 1: # Perform 80/20 Stratified Train-Test Split to preserve target class proportions
Lines 2–7: X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
```
`train_test_split` shuffles data (using `random_state=42`) then assigns 80% to training and 20% to testing. **Why `stratify=y`?** Without stratification, the random split might give 40% canceled in train but 34% canceled in test. With `stratify=y`, both splits maintain the same ~37% canceled ratio as the full dataset — crucial for fair evaluation.

Returns 4 arrays: `X_train` (~95,100 rows), `X_test` (~23,800 rows), `y_train`, `y_test`.

```
Line 9: # Apply StandardScaler: Fit on training data only, transform both train and test to prevent leakage
Line 10: scaler = StandardScaler()
Line 11: X_train_scaled = scaler.fit_transform(X_train)
Line 12: X_test_scaled = scaler.transform(X_test)
```
**StandardScaler formula:** `z = (x − μ_train) / σ_train`

`scaler.fit_transform(X_train)`: (1) `fit` computes μ (mean) and σ (std) for each of 27 features using **only training data**, (2) `transform` applies the formula to every value in X_train. Result: each feature column has mean≈0, std≈1.

`scaler.transform(X_test)`: Uses the **same μ and σ from training** — does NOT recompute from test data. This is critical: computing statistics from test data would be **data leakage** (the model would know about test distribution during preprocessing).

**Who needs scaling and why:**
- **KNN (needs it):** Distance-based. Without scaling, `adr` (range 0–5400) would completely dominate `is_repeated_guest` (range 0–1).
- **Logistic Regression (needs it):** Gradient descent converges much faster when all features are on similar scales.
- **Neural Network (needs it):** Same as LR. Mismatched scales cause vanishing/exploding gradients.
- **Decision Tree (does NOT need it):** Splits on thresholds — `adr < 80` works the same whether adr is raw or scaled. Scale-invariant.
- **Naive Bayes (optional):** Estimates Gaussians per feature per class — not strictly distance-based, but we use scaled data for consistency.

```
Lines 14–15: print statements
```
Confirm training and testing sample counts.

---

## CELL 22 — evaluate_model() Helper Function (31 lines)

**What this cell does:** Defines a reusable function that computes all metrics for any model and stores results centrally.

```
Line 1: # Dictionary to store metrics and predictions for all models
Line 2: results = {}
```
An empty dictionary that will accumulate results from all 5 models. Used in Section 7 for the comparison plots.

```
Line 4: def evaluate_model(name, y_true, y_pred, y_prob):
Line 5:     """Compute and store classification metrics (Accuracy, Precision, Recall, F1, AUC)."""
```
Defines the function. Takes model name string, true test labels, predicted labels, and predicted probabilities.

```
Line 6:  acc  = accuracy_score(y_true, y_pred)
Line 7:  prec = precision_score(y_true, y_pred, zero_division=0)
Line 8:  rec  = recall_score(y_true, y_pred, zero_division=0)
Line 9:  f1   = f1_score(y_true, y_pred, zero_division=0)
```
All four primary classification metrics:
- **Accuracy** = (TP+TN) / (TP+TN+FP+FN) — overall correct rate
- **Precision** = TP / (TP+FP) — of predicted cancellations, how many were real
- **Recall** = TP / (TP+FN) — of all real cancellations, how many did we catch
- **F1** = 2 × (Prec × Rec) / (Prec + Rec) — harmonic mean; best single metric for imbalanced data
- `zero_division=0` prevents division-by-zero if a model predicts only one class.

```
Line 10: fpr, tpr, _ = roc_curve(y_true, y_prob)
Line 11: auc_score = auc(fpr, tpr)
```
`roc_curve()` varies the classification threshold from 0 to 1 and at each threshold computes: FPR = FP/(FP+TN) and TPR = TP/(TP+FN) = Recall. Returns arrays of (FPR, TPR) pairs. The `_` discards the threshold array (Python convention for "don't need this"). `auc(fpr, tpr)` computes area under the ROC curve using the trapezoidal rule. AUC=1.0 is perfect; AUC=0.5 is random guessing.

```
Lines 13–24: results[name] = { ... }
```
Stores all computed values plus the raw prediction arrays (`y_pred`, `y_prob`, `fpr`, `tpr`) keyed by model name. This central dictionary powers all comparison plots in Section 7.

```
Lines 26–31: print statements
```
Immediately prints all metrics after each model runs, so you can monitor results as the notebook executes.

---

## CELL 24 — KNN Model (10 lines)

```
Line 1: # Model 1: K-Nearest Neighbors Classifier (trained on scaled features)
Line 2: knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, n_jobs=-1)
```
Creates KNN model. **Parameters:**
- `n_neighbors=5`: Look at the 5 nearest training points and take majority vote. Odd number avoids ties.
- `metric='minkowski', p=2`: Minkowski distance with p=2 = Euclidean distance: `d = √(Σ(xᵢ−yᵢ)²)`.
- `n_jobs=-1`: Use all available CPU cores for parallel distance computation.

```
Line 3: knn.fit(X_train_scaled, y_train)
```
KNN is a "lazy learner" — `.fit()` simply stores all training data in memory. No actual model parameters are learned. Distance computation happens only at prediction time.

```
Line 5: # Predict test set classes and probabilities
Line 6: y_pred_knn = knn.predict(X_test_scaled)
Line 7: y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
```
`predict()`: For each test point, finds 5 nearest training points → majority class label. `predict_proba()`: Returns matrix of shape (n_test, 2). `[:, 1]` selects column 1 = P(class 1 = canceled). For KNN, probability = fraction of k neighbors that are class 1 (e.g., 3 out of 5 → probability = 0.6).
🏨 Example: A New Guest Names "Alex" Books a Room
We want our KNN model to answer two questions about Alex:

1. predict() 
→
→ "Give me a Final Decision (Yes or No)"
What it does: KNN finds the 5 past guests who are most similar to Alex.
The Vote:
Suppose 3 of those past guests cancelled.
And 2 of those past guests showed up.
Result: Since 3 is the majority (3>2), predict() outputs: 1 (Canceled).
Output format: A single label: 0 or 1.

```
Line 9: # Evaluate KNN performance
Line 10: evaluate_model('KNN', y_test, y_pred_knn, y_prob_knn)
```
Calls the helper function — computes and stores all metrics, prints them.

---

## CELL 26 — Decision Tree Model + Figure 8 (21 lines)
 Decision Tree: The "Receptionist's Checklist"
```
Line 1: # Model 2: Decision Tree Classifier (max_depth=10 to prevent overfitting)
Line 2: dt = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=20, random_state=RANDOM_STATE)
```
**Parameters:**
- `criterion='gini'`: Splits using Gini Impurity. Formula: `Gini = 1 − (p₀² + p₁²)`. A pure node (all one class) = 0. Fully mixed = 0.5. The tree picks the split that maximizes purity gain.
- `max_depth=10`: Limits tree to 10 levels deep. Without this, tree grows until every leaf has 1 sample → perfect training accuracy, terrible generalization (memorization).
- `min_samples_split=20`: A node must have at least 20 samples to be split further. Prevents overfitting on tiny groups.
- Note: `X_train` (unscaled) is used because Decision Trees are scale-invariant.

```
Line 3: dt.fit(X_train, y_train)
```
Actually builds the tree — finds the best feature+threshold at each node to maximize Gini purity gain.

```
Lines 5–10: Predict and evaluate
```
Same pattern as KNN: predict labels, extract class 1 probabilities, call evaluate_model.

```
Line 12: # Figure 8: Plot Top 15 Feature Importances computed by the Decision Tree
Line 13: feat_imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
```
`dt.feature_importances_` — After training, each feature gets an importance score = total weighted reduction in Gini impurity it contributed across all splits, normalized so all scores sum to 1.0. `pd.Series(values, index=X.columns)` attaches feature names to scores. `.sort_values(ascending=False).head(15)` takes the top 15.

```
Lines 14–21: Plot horizontal bar chart
Line 18: ax.invert_yaxis()
```
`invert_yaxis()` puts the most important feature at the top of the chart (bar charts default to bottom-first).

---

## CELL 28 — Logistic Regression Model (10 lines)

```
Line 2: lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_STATE, n_jobs=-1)
```
**Parameters:**
- `max_iter=1000`: Maximum gradient descent iterations. Default of 100 often fails to converge on large datasets. 1000 ensures convergence.
- `solver='lbfgs'`: Limited-memory BFGS — a quasi-Newton method that uses curvature information to take smarter steps than plain gradient descent. Recommended for L2-penalized logistic regression on medium datasets.

**How Logistic Regression works:**
1. Computes: `z = w·x + b` (linear combination of features and learned weights)
2. Applies sigmoid: `ŷ = 1 / (1 + e^(−z))` → probability of class 1
3. Minimizes binary cross-entropy loss: `L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]`
4. Updates weights via gradient descent until convergence

Scaled data is critical here: mismatched feature scales cause slow, oscillating gradient descent.

Think of Logistic Regression like a Credit Score or Risk Meter:

When a guest books a room, the model calculates a Cancellation Score by adding and subtracting points for each feature:
---

## CELL 30 — Naive Bayes Model (10 lines)

```
Line 2: nb = GaussianNB()
Line 3: nb.fit(X_train_scaled, y_train)
```
GaussianNB has no hyperparameters. `.fit()` estimates μₖᵢ and σ²ₖᵢ (mean and variance) for each feature i in each class k from training data.

**How it works:** `P(y=k | X) ∝ P(y=k) × Π P(xᵢ | y=k)` where each `P(xᵢ | y=k)` is a Gaussian density with the estimated mean and variance. The "Naive" assumption is that all features are **conditionally independent** given the class — almost never true in reality, but the model still often gives reasonable results.

**Why it likely performs worst here:** "Sir, Gaussian Naive Bayes amader dataset e 41.8% accuracy diyeche karon eitar duita main assumption amader data te break hoyeche:

Amader features gulo independent na (price, children, room type correlated).
Amader continuous data gulo Gaussian (Normal) distribution na, borong heavily skewed."
---

## CELL 32 — Neural Network Model + Figure 9 (53 lines)

```
Line 1: # Model 5: Deep Neural Network Architecture with BatchNormalization and Dropout
Lines 2–11: nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
Layer-by-layer explanation:
- **Dense(128, relu, input_shape=(27,))**: First hidden layer. 128 neurons. Each computes `output = ReLU(W·input + b)`. Weight matrix W: shape (27, 128) = 3,456 weights. ReLU: `f(x) = max(0, x)`. Chosen over sigmoid/tanh for hidden layers because it doesn't suffer from vanishing gradients.
- **BatchNormalization()**: Normalizes the 128 outputs per mini-batch: `y = γ × (x − μ_batch)/σ_batch + β`. Stabilizes training, allows higher learning rates, acts as mild regularizer.
- **Dropout(0.3)**: During training, randomly sets 30% of the 128 outputs to 0 each forward pass. Forces neurons to not co-adapt — each must learn independently. During inference, all neurons active (outputs scaled by 0.7).
- **Dense(64, relu)**: Second hidden layer. Compresses 128 → 64.
- **BatchNormalization() / Dropout(0.2)**: Same purpose as before but smaller dropout rate since layer is smaller.
- **Dense(32, relu)**: Third hidden layer. Further compresses 64 → 32.
- **Dense(1, sigmoid)**: Output layer. Single neuron with sigmoid activation → output in [0, 1] = P(is_canceled=1). If output ≥ 0.5, predict class 1.

```
Line 13: # Compile model using Adam optimizer and binary cross-entropy loss
Line 14: nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- **Adam optimizer**: Adaptive Moment Estimation. Maintains exponential moving averages of gradients (momentum) and squared gradients (adaptive scale). Adjusts learning rate per parameter. Default learning rate = 0.001. Combines benefits of Momentum and RMSprop. Much faster and more stable than vanilla SGD.
- **binary_crossentropy loss**: `L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]`. Penalizes confident wrong predictions heavily (e.g., predicting 0.01 when true label is 1 gives large loss).
- **metrics=['accuracy']**: Tracked and reported each epoch for monitoring progress.

```
Line 16: # Early stopping callback to prevent overfitting and save best weights
Line 17: early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
```
- `monitor='val_loss'`: Watches the validation loss after every epoch.
- `patience=5`: If val_loss doesn't improve for 5 consecutive epochs, stop training.
- `restore_best_weights=True`: After stopping, roll back model weights to the epoch with the lowest val_loss — not the last epoch.
- `verbose=1`: Prints a message when early stopping triggers.

```
Lines 19–27: nn_history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=50, batch_size=512, validation_split=0.1,
    callbacks=[early_stop], verbose=1
)
```
- `epochs=50`: Maximum 50 passes through the training data. Early stopping usually halts it at 15–30.
- `batch_size=512`: Process 512 samples, compute gradients, update weights. One epoch = `ceil(n_train/512)` ≈ 168 weight updates.
- `validation_split=0.1`: Keras holds out the **last 10%** of training data for validation. Used to monitor val_loss for early stopping.
- `callbacks=[early_stop]`: After each epoch, Keras calls the callback which checks stopping conditions.

```
Lines 29–46: Figure 9 — Learning Curves
Line 30: fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
```
Unpacks two Axes directly into `ax1` and `ax2`.

```
Lines 31–35: ax1 (Loss curve)
ax1.plot(nn_history.history['loss'], label='Train Loss', color='#e74c3c')
ax1.plot(nn_history.history['val_loss'], label='Val Loss', color='#3498db', linestyle='--')
```
`nn_history.history` is a dictionary: `{'loss': [...], 'val_loss': [...], 'accuracy': [...], 'val_accuracy': [...]}` — one value per epoch.

Healthy loss curves: both decrease and plateau. If val_loss starts rising while train_loss keeps falling → overfitting (early stopping catches this).

```
Lines 48–50: Prediction
Line 49: y_prob_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()
Line 50: y_pred_nn = (y_prob_nn >= 0.5).astype(int)
```
`nn_model.predict()` returns shape (n_test, 1). `.flatten()` converts to shape (n_test,). `(y_prob_nn >= 0.5)` creates a boolean array. `.astype(int)` converts True→1, False→0. The 0.5 threshold can be adjusted — lower threshold means higher recall (catch more cancellations) at cost of lower precision.

---

## CELL 34 — Comparison Table + Figure 10: Accuracy Bar Chart (28 lines)

```
Lines 1–9: comparison_df creation
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    ...
}).set_index('Model').round(4).sort_values('F1-Score', ascending=False)
```
Builds a summary DataFrame from the `results` dictionary. `list(results.keys())` = model names. List comprehensions extract each metric for each model. `.set_index('Model')` makes model names the row index. `.round(4)` rounds to 4 decimal places. `.sort_values('F1-Score', ascending=False)` sorts with best F1 model first. `display(comparison_df)` renders it as a formatted HTML table in Colab.

```
Lines 14–28: Figure 10 — Accuracy Bar Chart
Line 15: model_names = list(results.keys())
Line 16: accuracies = [results[m]['accuracy'] for m in model_names]
Line 17: colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
```
One distinct color per model — consistent across all Section 7 plots.

```
Lines 21–22: Annotation loop
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc*100:.2f}%', ...)
```
Labels each bar with its percentage value floating above.

```
Line 25: ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
```
Formats y-axis ticks: `0.85` → `85.00%`. `xmax=1.0` specifies that 1.0 represents 100%.

---

## CELL 35 — Figure 11: Precision/Recall/F1 Grouped Bar (14 lines)

```
Lines 2–6: metrics_df = pd.DataFrame({
    'Precision': [...], 'Recall': [...], 'F1-Score': [...]
}, index=model_names)
```
Creates a DataFrame with one row per model and three metric columns.

```
Line 9: metrics_df.plot(kind='bar', ax=ax, color=[...], edgecolor='black', rot=15, width=0.7)
```
`metrics_df.plot(kind='bar')` creates a **grouped bar chart** — 3 bars side by side for each model. `rot=15` rotates x-axis model name labels 15° to prevent overlap.

**How to read this chart:** A model with high F1 but low recall is missing too many real cancellations. A model with high recall but low precision is raising too many false alarms. The ideal model balances both.

---

## CELL 36 — Figure 12: Confusion Matrices Grid (15 lines)

```
Line 2: fig, axes = plt.subplots(2, 3, figsize=(18, 10))
Line 3: axes = axes.flatten()
Line 4: cmaps = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']
```
2×3 grid for 5 models + 1 hidden slot. Each model gets a distinct colormap.

```
Lines 6–9: for i, (m, c) in enumerate(zip(model_names, cmaps)):
    cm = confusion_matrix(y_test, results[m]['y_pred'])
    ConfusionMatrixDisplay(cm, display_labels=['Not Canceled', 'Canceled']).plot(ax=axes[i], cmap=c, colorbar=False)
    axes[i].set_title(f'{m} Confusion Matrix', fontweight='bold')
```
`confusion_matrix(y_test, results[m]['y_pred'])` returns a 2×2 array:
```
[[TN  FP]
 [FN  TP]]
```
- **TN**: Correctly predicted "will not cancel" — good
- **TP**: Correctly predicted "will cancel" — good
- **FP**: Predicted "cancel" but they showed up — wasted resources (false alarm)
- **FN**: Predicted "will come" but they canceled — empty room, lost revenue (most costly)

`ConfusionMatrixDisplay(...).plot()` renders the matrix as a color-coded grid with cell values. `colorbar=False` removes the color scale bar (cleaner in a grid).

```
Line 11: axes[5].set_visible(False)
```
Hides the empty 6th subplot.

---

## CELL 37 — Figure 13: ROC Curves Comparison (14 lines)

```
Lines 2–4: fig, ax creation and loop
for m, c in zip(model_names, colors):
    ax.plot(results[m]['fpr'], results[m]['tpr'], color=c, linewidth=2, label=f"{m} (AUC = {results[m]['auc']:.4f})")
```
Plots one ROC curve per model using pre-computed FPR and TPR arrays from the `results` dictionary. Each curve shows how well a model separates classes as the decision threshold varies.

```
Line 6: # Plot random chance baseline (AUC = 0.5)
Line 7: ax.plot([0, 1], [0, 1], 'k--', label='Random Baseline (AUC = 0.5)')
```
The diagonal dashed line represents a random classifier (50/50 coin flip). Any useful model must curve significantly above this line. The area between a model's curve and this line represents the model's discriminative power.

**AUC interpretation:**
- AUC = 1.0 → Perfect classifier
- AUC = 0.5 → Random guessing
- AUC = 0.85 → The model ranks a random positive sample higher than a random negative sample 85% of the time

---

## CELL 39 — K-Means Clustering + Figures 14 & 15 (57 lines)

```
Line 1: # Step 1: Remove target variable to create purely unsupervised feature set
Line 2: X_unsup = X_train_scaled.copy()
```
The target `is_canceled` was already removed when we separated X and y in preprocessing. `X_train_scaled` contains only the 27 features. `.copy()` creates an independent copy to avoid any accidental modification of the supervised learning data.

```
Line 4: # Step 2: Compute WCSS (Inertia) over k=1..10 using Elbow Method (subsample for efficiency)
Line 5: np.random.seed(RANDOM_STATE)
Line 6: sample_idx = np.random.choice(len(X_unsup), size=min(10000, len(X_unsup)), replace=False)
Line 7: X_sample = X_unsup[sample_idx]
```
`np.random.choice(n, size=10000, replace=False)`: Randomly selects 10,000 row indices without replacement. `X_unsup[sample_idx]`: Selects those rows. Subsampling is used because running KMeans for k=1 to 10 on all 95,000 training samples would take several minutes. 10,000 samples gives reliable elbow detection in seconds.

```
Lines 9–14: Elbow Method loop
inertia_values = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=RANDOM_STATE)
    km.fit(X_sample)
    inertia_values.append(km.inertia_)
```
`km.inertia_` = WCSS (Within-Cluster Sum of Squares): `Σᵢ Σ(x∈Cᵢ) ||x − μᵢ||²`. Always decreases as k increases. The elbow is where the decrease rate sharply slows — marginal benefit of adding another cluster drops. `n_init=5`: Run 5 times from different random starts, keep the best (lowest inertia).

```
Lines 16–26: Figure 14 — Elbow Curve
Line 19: ax.axvline(x=2, color='red', linestyle='--', label='Optimal k=2')
```
`axvline()` draws a vertical line at x=2, marking the chosen optimal number of clusters. The elbow should be visibly at k=2 (or nearby), confirming the binary structure matches the two-class problem.

```
Line 28: # Step 3: Train final K-Means model with optimal k=2 on full training features
Line 29: kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, random_state=RANDOM_STATE)
Line 30: kmeans.fit(X_unsup)
Line 31: cluster_labels = kmeans.labels_
```
**K-Means++ initialization:** Spreads initial centroids far apart to avoid poor starting positions. **Algorithm:** (1) Assign each point to nearest centroid, (2) Recompute centroids as means of their points, (3) Repeat until convergence or max_iter=300 reached. `n_init=10`: Run 10 times from different starts, keep best run. `cluster_labels`: Array of shape (n_train,), values 0 or 1 — the assigned cluster for each training point.

```
Line 33: # Step 4: Reduce features to 2D via PCA for 2D visual cluster analysis
Line 34: pca = PCA(n_components=2, random_state=RANDOM_STATE)
Line 35: X_pca = pca.fit_transform(X_unsup[:5000])
Line 36: cluster_sample = cluster_labels[:5000]
Line 37: actual_sample = y_train.values[:5000]
```
**PCA algorithm:** (1) Centers data, (2) Computes covariance matrix, (3) Finds eigenvectors (principal components), (4) Projects data onto top 2 eigenvectors. Result: 27D → 2D. We use only 5,000 points for fast rendering of a readable scatter plot. `y_train.values[:5000]` gets the actual class labels (0/1) for the same 5,000 points — used for the "ground truth" side of the comparison.

```
Lines 39–52: Figure 15 — Side-by-side PCA scatter plots
Line 41: s1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_sample, cmap='RdYlGn', alpha=0.5, s=15)
Line 43: ax1.add_artist(ax1.legend(*s1.legend_elements(), labels=['Cluster 0', 'Cluster 1']))
```
Left plot: colored by K-Means cluster assignment. Right plot: colored by actual `is_canceled` label. Comparing the two plots shows how closely the unsupervised clusters match the real class structure. `alpha=0.5`: semi-transparent points so overlapping points are visible. `s=15`: small dot size (5000 points would be too dense with large dots). `s1.legend_elements()`: automatically generates legend handles based on the color mapping.

```
Lines 54–57: Cross-tabulation
ct = pd.crosstab(cluster_labels, y_train.values, rownames=['Cluster'], colnames=['Actual Status'])
```
Creates a 2×2 table: for each K-Means cluster (0 or 1), shows how many true class-0 and class-1 samples it contains. If one cluster is dominated by cancellations and the other by non-cancellations, K-Means successfully discovered the class structure without ever seeing the labels.

---

## CELL 41 — Final Summary Output (12 lines)

```
Line 1: # Determine top-performing models
Line 2: best_f1_model = comparison_df.index[0]
Line 3: best_auc_model = comparison_df.sort_values('AUC-ROC', ascending=False).index[0]
```
`comparison_df` was already sorted by F1-Score descending, so `.index[0]` = the model with the highest F1. For AUC, we re-sort separately because F1-best and AUC-best may not be the same model.

```
Lines 5–12: Formatted print block
```
Prints a clean summary box with the best model by each metric and a full comparison table. This is the final output of the notebook — a concise answer to "which model performed best?"

## Cell: Install Libraries
```python
!pip install -q scikit-learn pandas numpy matplotlib seaborn tensorflow
```
`!` = shell command from Colab. `pip install` = Python package installer. `-q` = quiet mode (no verbose output). These are pre-installed in Colab, so this is a safety net for running elsewhere.

---

## Cell: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
DATASET_PATH = '/content/drive/MyDrive/CSE422/11.csv'
```
**Line 1:** Imports Colab's drive module.
**Line 2:** Opens authentication popup. After linking your Google account, your entire Drive is available under `/content/drive/MyDrive/`.
**Line 3:** Path string variable. You MUST place `11.csv` at this exact Drive path: MyDrive → CSE422 → 11.csv.

---

## Cell: Global Imports — Detailed Explanation

```python
import pandas as pd
```
`pandas` provides the `DataFrame` — a 2D labeled table, perfect for datasets. `pd` is universally standard alias.

```python
import numpy as np
```
`numpy` provides N-dimensional array (`ndarray`) and mathematical functions. The backbone of scientific Python. Every ML library uses numpy arrays internally.

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
```
`matplotlib.pyplot` = main plotting interface. `plt` alias is universal. `matplotlib.ticker` provides axis formatters:
- `mticker.FuncFormatter(lambda x, _: f'{int(x):,}')` → formats 75000 as "75,000"
- `mticker.PercentFormatter(xmax=1.0)` → formats 0.8532 as "85.32%"

```python
import seaborn as sns
```
`seaborn` = statistical visualization on top of matplotlib. Much simpler API for heatmaps, box plots, bar plots, violin plots.

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
```
`LabelEncoder` converts text categories to integers: `{'City Hotel':0, 'Resort Hotel':1}`.
`StandardScaler` computes `z = (x - μ) / σ` per feature, resulting in mean=0, std=1.

```python
from sklearn.model_selection import train_test_split
```
Randomly shuffles and splits dataset into training and testing portions in one function call.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
```
Four scikit-learn classifiers. All share the same API:
- `.fit(X_train, y_train)` — train the model
- `.predict(X_test)` — predict class labels
- `.predict_proba(X_test)` — predict class probabilities

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
```
TensorFlow/Keras for Neural Network:
- `Sequential` — Stack layers linearly
- `Dense` — Fully connected layer (every input connected to every output)
- `Dropout` — Regularization: random neuron deactivation during training
- `BatchNormalization` — Normalize layer activations per batch
- `EarlyStopping` — Auto-stop when validation metric stops improving

```python
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay,
    precision_score, recall_score, f1_score
)
```
All evaluation tools:
- `accuracy_score` → (TP+TN)/(Total)
- `classification_report` → Per-class precision/recall/F1 as text table
- `confusion_matrix` → 2×2 prediction matrix
- `ConfusionMatrixDisplay` → Visual confusion matrix
- `roc_curve` → False Positive Rate vs True Positive Rate pairs
- `auc` → Area Under a Curve (any curve, given x, y arrays)
- `precision_score` → TP/(TP+FP)
- `recall_score` → TP/(TP+FN)
- `f1_score` → 2×Precision×Recall/(Precision+Recall)

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
```
`KMeans` = unsupervised clustering algorithm.
`PCA` = Principal Component Analysis for dimensionality reduction (used for cluster visualization).

```python
import warnings
warnings.filterwarnings('ignore')
```
Suppresses non-critical Python warnings (e.g., convergence warnings, deprecation notices). Output stays clean. Actual errors are NOT suppressed.

```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
```
`RANDOM_STATE = 42` — A constant used throughout for reproducibility. Any integer works; 42 is ML community convention (from "The Hitchhiker's Guide to the Galaxy").

Setting seeds ensures: every time you run the notebook, train/test split is identical, model weights initialize identically, dropout drops the same neurons. Without seeds, results differ each run — makes debugging and presentation hard.

`np.random.seed()` seeds NumPy's Mersenne Twister RNG.
`tf.random.set_seed()` seeds TensorFlow's separate RNG (TF has its own independent random system).

```python
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
```
`sns.set_style('whitegrid')` — Global plot style: white background with grey grid lines. Clean and professional.
`plt.rcParams['figure.dpi'] = 100` — Sets figure resolution globally. 100 DPI gives good quality without being too large.

---

## Section 1: Introduction
**Markdown cell** — no Python code. States: hotel cancellations cause revenue loss, the goal is predicting `is_canceled`, this is a classification problem because `is_canceled` is binary (0 or 1). Explains motivation (revenue management, overbooking) and lists all models used.

---

## Section 2: Dataset Description

### Load Dataset
```python
df = pd.read_csv(DATASET_PATH)
```
`pd.read_csv()` reads the CSV file line by line, parses values, infers data types, and creates a DataFrame with column names from row 1. The entire 119,392-row, 32-column dataset is loaded into RAM as `df`.

```python
print(f'Total Rows    : {df.shape[0]:,}')
print(f'Total Columns : {df.shape[1]}')
```
`df.shape` is a tuple `(119392, 32)`. `df.shape[0]` = 119392 (rows). `df.shape[1]` = 32 (columns). `:,` formats numbers with comma separators.

### First 5 Rows
```python
df.head()
```
`.head(n=5)` returns the first 5 rows. Essential "sanity check" — confirms the file loaded correctly, shows column names in context, reveals data types visually before running `df.info()`.

### Data Info
```python
df.info()
```
Prints dtype and non-null count for each column. Immediately reveals:
- `agent`: 119392 total rows but only 103052 non-null → 16,340 nulls
- `company`: only 6799 non-null → 112,593 nulls  
- `children`: 119388 non-null → 4 nulls
- `reservation_status_date`: object dtype (stored as string despite being a date)

### Separate Numerical and Categorical
```python
numerical_cols   = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
```
`select_dtypes(include=['int64', 'float64'])` → filters columns stored as integer or float.
`select_dtypes(include=['object'])` → filters string columns.
`.columns.tolist()` → converts pandas Index to Python list.

Why separate? EDA techniques differ: histograms/box plots for numerical, bar plots for categorical. Encoding only applies to categorical. Correlation only works on numerical.

### Target Distribution and Bar Chart
```python
class_counts      = df['is_canceled'].value_counts()
class_percentages = df['is_canceled'].value_counts(normalize=True) * 100
```
`df['is_canceled']` selects the target column as a pandas Series (1D array with labels).
`.value_counts()` counts each unique value: `{0: 75166, 1: 44224}`.
`normalize=True` divides by total to get proportions. `* 100` converts to percentages.

```python
fig, ax = plt.subplots(figsize=(7, 5))
```
Creates a matplotlib Figure object and one Axes object. Figure = the whole canvas. Axes = the plotting area inside. `figsize=(7, 5)` = 7 inches wide × 5 inches tall.

```python
bars = ax.bar(['Not Canceled (0)', 'Canceled (1)'], class_counts.values,
               color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
```
`ax.bar()` draws vertical bars:
- 1st arg: x-axis tick labels
- 2nd arg: bar heights (count values)
- `color`: green (#2ecc71) for class 0 (positive = kept), red (#e74c3c) for class 1 (negative = canceled)
- `edgecolor='black'`: adds thin black borders around each bar (improves visibility)
- `width=0.5`: bars are 50% of slot width (leaves spacing between bars)

```python
for bar, count, pct in zip(bars, class_counts.values, class_percentages.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
            f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
```
`zip(bars, class_counts.values, class_percentages.values)` — Iterates over the 3 sequences together in lockstep.
`bar.get_x() + bar.get_width()/2` = horizontal center of bar.
`bar.get_height() + 400` = 400 units above bar top (so label doesn't touch bar).
`f'{count:,}\n({pct:.1f}%)'` = count with commas, newline, then percentage with 1 decimal.
`ha='center'` = horizontally centered. `va='bottom'` = bottom of text anchored at y-position.

```python
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.savefig('class_distribution.png', bbox_inches='tight')
```
`FuncFormatter` formats y-axis ticks as comma-separated integers.
`plt.savefig()` saves as PNG. `bbox_inches='tight'` trims whitespace around figure for cleaner saved image.

**Result of this cell:** A bar chart showing ~75,166 (62.96%) not canceled and ~44,224 (37.04%) canceled. Slight imbalance noted — requires stratified splitting.

---

## Section 3: EDA

### Descriptive Statistics
```python
df[numerical_cols].describe().T.round(2)
```
`df[numerical_cols]` selects only numerical columns.
`.describe()` computes: count, mean, std, min, Q1 (25%), median (50%), Q3 (75%), max for each column.
`.T` transposes: features become rows, statistics become columns. More readable with many features.
`.round(2)` rounds to 2 decimal places.

**What to look for:** Features where min=0 and Q1=0 are highly concentrated at zero (e.g., `days_in_waiting_list`: most bookings have 0 wait days). High std relative to mean indicates high variability (e.g., `adr`).

### Missing Values Bar Chart
```python
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
```
`df.isnull()` → boolean DataFrame (True where NaN). `.sum()` → counts per column (True=1).
`missing[missing > 0]` → filters to only columns with at least one null.
`.sort_values(ascending=False)` → most missing first.

```python
ax.bar(missing.index, missing.values, color='#e67e22', edgecolor='black')
```
`missing.index` = column names. `missing.values` = null counts. Orange (#e67e22) color signals a warning/issue.

### Histograms
```python
key_num_cols = ['lead_time', 'adr', 'stays_in_weekend_nights',
                'stays_in_week_nights', 'total_of_special_requests',
                'previous_cancellations', 'days_in_waiting_list']
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
axes = axes.flatten()
```
`plt.subplots(3, 3)` creates a 3×3 grid = 9 subplots. `axes` is shape (3,3). `.flatten()` converts to shape (9,) — a 1D list so we can index as `axes[0]`, `axes[1]`, etc.

```python
for i, col in enumerate(key_num_cols):
    axes[i].hist(df[col].dropna(), bins=40, color='#3498db', edgecolor='white')
```
`enumerate()` gives (index, value) pairs. `df[col].dropna()` removes NaN before plotting.
`bins=40` = 40 equal-width buckets across the data range.
`color='#3498db'` = blue bars. `edgecolor='white'` = thin white lines between bars.

```python
for j in range(len(key_num_cols), len(axes)):
    axes[j].set_visible(False)
```
We have 7 features but 9 subplot slots. Hides the 8th and 9th empty slots.

**Purpose:** Visualizes the SHAPE of each feature's distribution. This reveals:
- `lead_time`: Right-skewed — most bookings made close to arrival, but some made >300 days ahead
- `adr`: Right-skewed with outliers — most rooms average €75-150/night but some luxury rooms cost much more
- `previous_cancellations`: Highly concentrated at 0 — most guests have no cancellation history

### Box Plots
```python
sns.boxplot(y=df[col], ax=ax, color='#9b59b6',
            flierprops=dict(marker='o', markerfacecolor='red', markersize=3))
```
`y=df[col]` = the data values (vertical orientation).
`flierprops` = dictionary of properties for outlier markers:
- `marker='o'` = circle shape
- `markerfacecolor='red'` = red fill
- `markersize=3` = small dots (many outliers would overlap if too large)

**Box plot components:**
- Bottom of box = Q1 (25th percentile)
- Line in box = Q2 (50th percentile = median)
- Top of box = Q3 (75th percentile)
- Whiskers = extend to max/min within 1.5×IQR from Q1/Q3
- Dots = individual outliers beyond whiskers

**Purpose:** Detects outliers. `adr` has extreme outliers (prices >5000). `lead_time` has bookings 700+ days in advance. These are kept (genuine data).

### Categorical Features Bar Plots
```python
ct = pd.crosstab(df[col], df['is_canceled'], normalize='index') * 100
ct.columns = ['Not Canceled (%)', 'Canceled (%)']
ct.plot(kind='bar', ax=axes[i], color=['#2ecc71', '#e74c3c'], rot=25)
```
`pd.crosstab(rows, columns, normalize='index')` = contingency table where each row sums to 100%.
So for each category of `col` (e.g., meal type), we see what percentage canceled.

`ct.plot(kind='bar')` uses pandas built-in plotting which wraps matplotlib.
`rot=25` = rotates x-axis labels 25 degrees to prevent overlap.

**Key finding:** `deposit_type = 'Non Refund'` cancels at ~99.4% — the strongest single predictor. Logic: guests who paid non-refundable often cancel when plans change (they lose the money but can't recover it by going anyway). `deposit_type = 'No Deposit'` has only ~28% cancellation.

### Pearson Correlation Heatmap
```python
pearson_corr = num_df.corr(method='pearson')
mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
```
`num_df.corr(method='pearson')` computes n×n Pearson correlation matrix where n = number of numerical features.

**Pearson formula:** `r = Σ[(xᵢ-x̄)(yᵢ-ȳ)] / [√Σ(xᵢ-x̄)² × √Σ(yᵢ-ȳ)²]`
Range: [-1, +1]. Measures LINEAR relationship strength and direction.

`np.triu(np.ones_like(pearson_corr, dtype=bool))` creates a boolean matrix with True in the upper triangle (diagonal and above). Used as `mask` to hide the upper triangle of the heatmap (since correlation matrix is symmetric: r(A,B) = r(B,A), showing both halves is redundant).

```python
sns.heatmap(pearson_corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
```
- `annot=True` → write correlation values inside cells
- `fmt='.2f'` → 2 decimal place format
- `cmap='coolwarm'` → blue=negative, white=zero, red=positive
- `linewidths=0.5` → thin lines between cells
- `vmin=-1, vmax=1` → fix color scale to full range (so 0.3 correlation doesn't look as red as 0.9)

### Target Correlation (3 Methods)
```python
for ax, method, title in zip(axes, methods, titles):
    corr_with_target = (
        num_df.corr(method=method)[['is_canceled']]
        .drop('is_canceled')
        .sort_values('is_canceled', ascending=False)
    )
```
`num_df.corr(method=method)` → full n×n matrix.
`[['is_canceled']]` → double brackets keep it as DataFrame (single brackets give Series).
`.drop('is_canceled')` → removes the "is_canceled vs itself" row (always 1.0, uninformative).
`.sort_values('is_canceled', ascending=False)` → features most positively correlated at top.

**Three correlation methods:**
- **Pearson:** `r = cov(X,Y)/(σₓσᵧ)`. Linear only. Sensitive to outliers. Best for normal distributions.
- **Spearman:** Pearson applied to RANKS of values, not raw values. Robust to outliers. Captures monotonic (consistently increasing/decreasing) relationships even if nonlinear. Good for ordinal data like `arrival_date_week_number`.
- **Kendall Tau:** `τ = (concordant pairs - discordant pairs) / C(n,2)`. More conservative than Spearman. Better when there are many tied values. Gives different (usually lower) coefficient values.

**Key findings:**
- `previous_cancellations` → HIGH positive correlation → Past cancelers are likely to cancel again
- `total_of_special_requests` → NEGATIVE correlation → Engaged guests (who bother to request things) are less likely to cancel
- `required_car_parking_spaces` → NEGATIVE correlation → Guests who need parking are more committed to coming
- `booking_changes` → NEGATIVE → Guests who modify their booking are invested in the stay

### Skewness
```python
skewness = num_df.skew().sort_values(ascending=False)
```
`.skew()` computes Fisher-Pearson skewness coefficient for each column.
- Skewness = 0 → Symmetric (normal-like)
- Skewness > 0 → Right-skewed (long right tail, most values concentrated left)
- Skewness < 0 → Left-skewed
- |Skewness| > 1 → Highly skewed

**Implications:** Highly skewed features may violate assumptions of Gaussian Naive Bayes (which assumes Gaussian distribution). Standard scaling doesn't fix skewness — would need log transform for that, but we keep it simple.

---

## Section 4: Preprocessing

### Working Copy
```python
data = df.copy()
```
`.copy()` creates a DEEP copy — an entirely independent DataFrame in memory. Any changes to `data` do not affect `df`. This is important because:
1. If you accidentally corrupt `data`, you don't need to re-read the 16MB CSV
2. You can compare `data` with `df` at any point to see what changed

### Drop Data Leakage Columns
```python
LEAKAGE_COLS = ['reservation_status', 'reservation_status_date']
data.drop(columns=LEAKAGE_COLS, inplace=True)
```
**Why `reservation_status` is leakage:** This column contains the string "Canceled" for canceled bookings and "Check-Out" or "No-Show" for the rest. It directly encodes the answer we're trying to predict. A model that sees this feature would learn: if reservation_status == "Canceled" → predict 1. That's not prediction — that's lookup. In deployment, we wouldn't know reservation_status at booking time (that's the future).

**Why `reservation_status_date` is leakage:** This is the date when the reservation_status was recorded. For a canceled booking, this is the cancellation date. Even without `reservation_status`, a model might infer that a date before the arrival date signals a cancellation. It contains timing information about the outcome.

`inplace=True` — modifies `data` directly. Without it, `data.drop(...)` would return a new DataFrame and `data` would be unchanged.

### Drop High-Cardinality Column
```python
print(f"Unique values in 'country': {data['country'].nunique()}")
data.drop(columns=['country'], inplace=True)
```
`country` has 178 unique values. Problems with encoding:
1. **One-hot encoding** → Creates 178 binary columns → Curse of dimensionality, massive memory use, slow training
2. **Label encoding** → Assigns integers 0-177 → Implies ordinal relationship (Austria=3 > Albania=0) which is FALSE for country codes
3. **Frequency encoding** → Could work but adds complexity

Decision: Drop. We accept losing some information from country to keep the model manageable.

### Handle Missing Values
```python
data['agent'].fillna(0, inplace=True)
data['company'].fillna(0, inplace=True)
data['children'].fillna(0, inplace=True)
```
`fillna(value)` replaces all NaN entries with `value`.

**`agent` — 16,340 NULLs (13.7%):**
NULL in the original dataset documentation means: "booking was made without a travel agent." Filling with 0 is semantically correct — it creates a distinct "no agent" value. Agent IDs in the real data start from non-zero numbers, so 0 is unambiguous.

**`company` — 112,593 NULLs (94.3%):**
NULL means "booking made by an individual, not a company." Filling with 0 = "no company." This preserves the information that most bookings are personal.

**`children` — 4 NULLs (0.003%):**
Only 4 missing out of 119,392. Almost certainly data entry errors for bookings with 0 children. Safe to fill with 0.

**Why not use mean imputation?**
- For IDs (agent, company): the mean of IDs (e.g., mean of 240, 9, 304...) is not a real agent ID and has no meaning
- For children: filling 4 rows with mean (≈0.1 children) would give a fractional child, which is nonsensical

### Duplicate Check
```python
n_duplicates = data.duplicated().sum()
```
`data.duplicated()` → boolean Series: True if a row is an exact copy of a previous row.
`.sum()` → counts True values (True=1, False=0).

`data.drop_duplicates(inplace=True)` → keeps only the first occurrence of each duplicate row.

### Categorical Encoding
```python
categorical_features = data.select_dtypes(include='object').columns.tolist()
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
```
After dropping `reservation_status`, `reservation_status_date`, and `country`, the remaining string columns are: `hotel`, `arrival_date_month`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `assigned_room_type`, `deposit_type`, `customer_type`.

For each:
- `LabelEncoder()` → new encoder instance
- `.astype(str)` → converts all values to strings first (prevents errors if any NaN slipped through)
- `.fit_transform(data[col])` → two operations:
  - `fit`: scans the column, finds all unique string values, sorts them alphabetically, assigns integers 0, 1, 2...
  - `transform`: replaces each string with its assigned integer
- `label_encoders[col] = le` → stores the encoder. Important: if you need to decode predictions later, you use `le.inverse_transform(array)`.

**Example encoding of `deposit_type`:**
- Sorted unique: ['No Deposit', 'Non Refund', 'Refundable']
- Mapping: {'No Deposit': 0, 'Non Refund': 1, 'Refundable': 2}

### Convert Types
```python
data['agent']    = data['agent'].astype(int)
data['company']  = data['company'].astype(int)
data['children'] = data['children'].astype(int)
```
After `fillna(0)`, these columns became float64 (because NaN is stored as float in pandas, and fillna preserves that type). Converting to int64 is cleaner and more memory-efficient for ID/count data.

### Separate Features and Target
```python
TARGET = 'is_canceled'
X = data.drop(columns=[TARGET])
y = data[TARGET]
```
`data.drop(columns=[TARGET])` → removes the target column, leaving all 27 feature columns as `X`.
`data[TARGET]` → extracts just the `is_canceled` column as `y`.

Shape after preprocessing:
- `X`: (118,898 rows × 27 features) [row count may be slightly less if duplicates were removed]
- `y`: (118,898,)

---

## Section 5: Dataset Splitting

### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
```
`train_test_split` shuffles the data (using `random_state=42`) then assigns 80% to training and 20% to testing.

**Returns 4 arrays:**
- `X_train` — Feature matrix for training (~95,000 rows)
- `X_test` — Feature matrix for testing (~23,800 rows)
- `y_train` — Labels for training
- `y_test` — Labels for testing (ground truth for evaluation)

**`stratify=y` is critical** for imbalanced datasets:
Without stratify: random chance might give 40% canceled in train but only 34% in test → models evaluated on different distributions → misleading metrics.
With stratify: Both sets have identical class proportions to the original (~63% class 0, ~37% class 1).

### Feature Scaling
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```
**StandardScaler formula per feature:**
`z = (x - μ_train) / σ_train`

`scaler.fit_transform(X_train)`:
- `fit` → computes μ (mean) and σ (standard deviation) for each of the 27 features, using ONLY training data
- `transform` → applies `z = (x - μ) / σ` to every value in X_train
- Combined as `fit_transform` for convenience

`scaler.transform(X_test)`:
- Uses the SAME μ and σ computed from training data
- Does NOT recompute statistics from test data
- This is **essential**: using test statistics would leak information about the test distribution into preprocessing

**Result:** Both X_train_scaled and X_test_scaled are numpy arrays of float64.
- X_train_scaled: each feature column has mean≈0, std≈1
- X_test_scaled: normalized by training statistics (may not be exactly mean=0 if test has different distribution)

**Who needs scaling and why:**
- **KNN:** Distances compute `√Σ(xᵢ-yᵢ)²`. Without scaling, `adr` (range 0-5400) completely dominates `is_repeated_guest` (range 0-1). The model would essentially only look at `adr`.
- **Logistic Regression:** Gradient descent for weight optimization converges much faster when features are on similar scales. Poorly scaled features cause slow, oscillating convergence.
- **Neural Network:** Same as Logistic Regression. Also, gradient magnitudes depend on feature scale — mismatched scales can cause some gradients to vanish and others to explode.
- **Decision Tree:** Does NOT need scaling. It splits by `feature < threshold`, so relative ordering matters, not absolute scale. `adr < 80` works equally whether adr is raw or scaled.
- **Naive Bayes:** Not strictly required (it estimates per-class Gaussians for each feature), but we use scaled data for consistency across all models.

---

## Section 6: Model Training

### evaluate_model() Helper Function
```python
def evaluate_model(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score   = auc(fpr, tpr)
    results[name] = {...}
```

**All metric formulas:**
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN) — Overall correct predictions
- **Precision** = TP / (TP + FP) — "Of predicted cancellations, how many were real?" High precision = few false alarms. A hotel optimizing for this wants to only flag bookings it's very sure will cancel.
- **Recall** = TP / (TP + FN) — "Of all real cancellations, how many did we catch?" High recall = few missed cancellations. A hotel optimizing for this wants to never miss a cancellation (even at cost of false alarms).
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) — Harmonic mean. Most important for imbalanced classification. Rewards models that are BOTH precise and have good recall.

Why harmonic mean instead of arithmetic mean?
- Arithmetic: (0.9 + 0.1) / 2 = 0.5 — a model with 90% precision but 10% recall looks "okay"
- Harmonic: 2×0.9×0.1/(0.9+0.1) = 0.18 — correctly shows this model is terrible

**ROC Curve:**
`roc_curve(y_true, y_prob)` — Varies the classification threshold from 0 to 1 and at each threshold:
- Computes FPR = FP/(FP+TN) — fraction of true negatives we incorrectly flagged
- Computes TPR = TP/(TP+FN) = Recall — fraction of true positives we caught
- Returns arrays of (FPR, TPR) pairs + corresponding thresholds

`auc(fpr, tpr)` — Computes area under the FPR-TPR curve using trapezoidal rule.

**The `_` in `fpr, tpr, _ = roc_curve(...)`:** The function returns 3 values; third is the threshold array which we don't need. `_` is Python convention for "discard this value."

`zero_division=0` — Some models might predict all class 0 or all class 1 for edge cases. This prevents division-by-zero errors (e.g., if no positive predictions: precision = TP/(TP+0) = TP/0 → undefined).

### Model 1: KNN

```python
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, n_jobs=-1)
knn.fit(X_train_scaled, y_train)
```

**KNN Algorithm:**
1. Store all training samples (lazy learning — no actual "training")
2. For each test point:
   a. Compute distance to every training point
   b. Find 5 nearest training points (k=5)
   c. Take majority vote of their y_train labels
   d. Return that label as prediction

**Why n_neighbors=5?** Classic default. Odd numbers avoid ties. Higher k → smoother decision boundary but potentially misses local patterns. Lower k → sensitive to noise. 5 is a reasonable balance for this dataset size.

**metric='minkowski', p=2:** Minkowski distance with p=2 = Euclidean distance:
`d = (Σ|xᵢ - yᵢ|^p)^(1/p)` with p=2 → `d = √(Σ(xᵢ - yᵢ)²)`

**n_jobs=-1:** Use ALL available CPU cores for parallel distance computation. KNN's distance calculations are embarrassingly parallel — each test point's nearest neighbors can be found independently.

**Why X_train_scaled?** KNN is entirely distance-based. Without scaling, `adr` (range 0-5400) would contribute 5400× more to the distance than `is_repeated_guest` (range 0-1). The model would essentially only look at the largest-valued features.

```python
y_pred_knn = knn.predict(X_test_scaled)
y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
```
`knn.predict()` → for each test point, returns the majority class among k=5 nearest training points.
`knn.predict_proba()` → returns probability matrix of shape (23879, 2). Column 0 = P(class 0), Column 1 = P(class 1). `[:, 1]` selects all rows, column 1 = probability of cancellation. For KNN, probability = fraction of k neighbors that are class 1 (e.g., 3 out of 5 → 0.6).

### Model 2: Decision Tree

```python
dt = DecisionTreeClassifier(
    criterion='gini', max_depth=10, min_samples_split=20, random_state=RANDOM_STATE
)
dt.fit(X_train, y_train)  # Note: unscaled X_train
```

**Decision Tree Algorithm:**
1. Start with all training data at root node
2. Try every feature and every possible threshold value
3. Choose the split that maximizes Gini information gain
4. Recursively split each child node
5. Stop when max_depth or min_samples_split reached, or node is pure

**Gini Impurity:**
`Gini(node) = 1 - Σᵢ pᵢ²`
For binary class: `Gini = 1 - (p₀² + p₁²)` where p₀, p₁ = class proportions.
- Pure node (all one class): `Gini = 1 - (1² + 0²) = 0` (perfect)
- Equally mixed: `Gini = 1 - (0.5² + 0.5²) = 0.5` (worst)

**Information Gain:**
`IG = Gini(parent) - [w_left × Gini(left) + w_right × Gini(right)]`
Choose the feature+threshold that maximizes IG (maximally purifies the children).

**max_depth=10:** Without depth limit, tree grows until each leaf has 1 sample = perfect training accuracy but terrible generalization (the model just memorizes every training example). Depth 10 allows up to 2^10 = 1024 leaf nodes — expressive but regulated.

**min_samples_split=20:** A node must have at least 20 samples to be considered for splitting. Prevents overfitting on tiny groups of similar points.

**Why unscaled X_train?** Decision Trees split on thresholds: "adr < 80" or "lead_time < 45". Whether adr is 80 raw or scaled to 0.23, the RELATIVE ORDER of samples is the same, so the same split is found. Scaling is completely irrelevant.

### Feature Importance Plot

```python
feat_imp = pd.Series(dt.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(15)
```
`dt.feature_importances_` — After training, each feature gets an importance score = total (weighted) reduction in Gini impurity it contributed across ALL splits.

Formula: `importance(feature f) = Σ_nodes [w_node × Gini_reduction(node)] / Σ_all_nodes [...]`
where the sum is over all nodes where feature f was used for splitting.

Values are normalized to sum to 1 (fractions of total impurity reduction).

**Interpretation:** A feature with importance 0.35 contributed 35% of all impurity reduction. Features with importance ~0 were barely used and could be dropped.

`pd.Series(values, index=X.columns)` creates a Series with feature names as labels (otherwise it would just be indices 0, 1, 2...).

### Model 3: Logistic Regression

```python
lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_STATE, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
```

**Logistic Regression Algorithm:**
1. Initialize weights `w` and bias `b` randomly
2. For each sample, compute: `z = w·x + b` (linear combination of features)
3. Apply sigmoid: `ŷ = σ(z) = 1 / (1 + e^(-z))` → probability of class 1
4. Compute binary cross-entropy loss: `L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`
5. Compute gradients: `∂L/∂w`, `∂L/∂b`
6. Update weights: `w ← w - α·∂L/∂w`
7. Repeat until convergence

**sigmoid function:** `σ(z) = 1/(1+e^(-z))`
- z → -∞: σ(z) → 0
- z = 0: σ(z) = 0.5
- z → +∞: σ(z) → 1

**max_iter=1000:** Default is 100, which often hits the iteration limit for large datasets without converging. 1000 ensures convergence while being fast enough.

**solver='lbfgs':** Limited-memory Broyden-Fletcher-Goldfarb-Shanno — a quasi-Newton method that approximates the inverse Hessian (second derivative) matrix. Converges much faster than gradient descent by using curvature information. Recommended for L2-penalized logistic regression on moderate-sized datasets.

**Why scaled data?** Gradient descent convergence depends on feature scales. If `adr` has range 0-5400 and `is_repeated_guest` has range 0-1, the gradient for `w_adr` is much larger in magnitude, causing the optimizer to take tiny steps for `w_adr` while making huge steps for `w_is_repeated_guest`. This causes slow, oscillating convergence. After scaling, all gradients are on similar scales → fast, stable convergence.

### Model 4: Gaussian Naive Bayes

```python
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
```

**Naive Bayes Algorithm:**
Apply Bayes' Theorem: `P(y|X) ∝ P(y) × P(X|y)`

For each feature: `P(xᵢ|y=k) = (1/√(2πσ²ₖᵢ)) × exp(-(xᵢ-μₖᵢ)² / (2σ²ₖᵢ))`
(Gaussian probability density function with class-specific mean μₖᵢ and variance σ²ₖᵢ)

**Training:** Estimate μₖᵢ and σ²ₖᵢ for each feature i and class k from training data.
**Prediction:** `argmax_k [log P(y=k) + Σᵢ log P(xᵢ|y=k)]` (log sum more numerically stable than product)

**"Naive" assumption:** Features are conditionally independent given the class:
`P(X|y=k) = ∏ᵢ P(xᵢ|y=k)`

This is almost never true in reality — lead_time and booking_changes probably correlate. But despite the false assumption, NB often works well because:
1. Decision boundaries are often still reasonable
2. Marginal probabilities still provide useful signal
3. Very fast to train

**GaussianNB limitation here:** Many features (previous_cancellations, days_in_waiting_list) are highly right-skewed — not Gaussian at all. The Gaussian assumption may reduce accuracy. NB tends to perform worse than other models on this type of mixed/skewed data.

**No explicit parameters for GaussianNB:** It has no hyperparameters to tune (unlike `n_neighbors` for KNN or `max_depth` for DT). It simply estimates the Gaussian parameters from data.

### Model 5: Neural Network

#### Architecture

```python
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

**Sequential:** A linear stack — output of layer L becomes input to layer L+1.

**Dense(128, activation='relu', input_shape=(27,)):**
- Creates a fully connected layer with 128 neurons
- `input_shape=(27,)` = accepts 27 features as input (only needed on first layer)
- For each of 128 neurons: computes `output_j = ReLU(Σᵢ wⱼᵢ·inputᵢ + bⱼ)`
- Weight matrix W: shape (27, 128) — 27 inputs × 128 neurons = 3,456 weights
- Bias vector b: shape (128,)
- Total parameters: 27×128 + 128 = 3,584

**ReLU activation:** `f(x) = max(0, x)`
Why ReLU over sigmoid/tanh for hidden layers?
1. **No vanishing gradient:** sigmoid/tanh saturate for large |x| → gradient ≈ 0 → weights stop updating (vanishing gradient problem). ReLU has gradient = 1 for x>0, so it doesn't vanish.
2. **Sparse activation:** About half the neurons output 0 → sparse representation → more efficient computation
3. **Computationally cheap:** Just a max operation vs exponential for sigmoid

**BatchNormalization():**
After each Dense layer, BN normalizes the layer outputs (activations):
`output_BN = γ × (activation - μ_batch)/σ_batch + β`
where μ_batch, σ_batch are computed per-batch, and γ, β are learned scale+shift parameters.

**Why BN?** "Internal covariate shift" problem: as weights update each batch, the distribution of layer inputs shifts constantly, forcing subsequent layers to continuously adapt. BN normalizes these distributions, providing:
1. **Faster training** (can use higher learning rates)
2. **Regularization** (slight noise from batch statistics)
3. **Stability** (less sensitive to initialization)

**Dropout(0.3):**
During training only: randomly sets 30% of the 128 neuron outputs to 0 each forward pass.
- Different neurons dropped each batch
- Forces the network to learn redundant representations
- Prevents any single neuron from becoming "too important"
- At inference time: all neurons active, outputs scaled by (1-0.3)=0.7

The "co-adaptation" problem without dropout: neurons develop dependencies on each other. With dropout: each neuron must work independently, improving generalization.

**Layer sizes 128→64→32:** Funnel architecture — progressively compresses the representation into a lower-dimensional space. Forces the network to extract only the most important features at each level.

**Dense(1, activation='sigmoid'):**
Single output neuron. Sigmoid squashes output to [0,1] = probability of cancellation.
- If output > 0.5: predict class 1 (canceled)
- If output ≤ 0.5: predict class 0 (not canceled)

**Total parameter count:**
- Dense(128): 27×128+128 = 3,584
- BN(128): 128×4 = 512 (γ, β, moving_mean, moving_var)
- Dense(64): 128×64+64 = 8,256
- BN(64): 256
- Dense(32): 64×32+32 = 2,080
- Dense(1): 32×1+1 = 33
- Total: ~14,720 parameters

#### Compile
```python
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Adam optimizer:** Adaptive Moment Estimation.
- Maintains exponential moving averages of gradients (1st moment = mean) and squared gradients (2nd moment = variance)
- Adapts learning rate per parameter: rarely-updated parameters get larger updates
- `m_t = β₁·m_{t-1} + (1-β₁)·g_t` (momentum)
- `v_t = β₂·v_{t-1} + (1-β₂)·g_t²` (adaptive scale)
- `θ_t = θ_{t-1} - α·m̂_t/√(v̂_t + ε)`
- Default: α=0.001, β₁=0.9, β₂=0.999, ε=1e-8
- Combines benefits of Momentum (faster convergence) and RMSprop (adaptive rates)

**Binary cross-entropy loss:** For single binary output:
`L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`
- y=1, ŷ=0.99 → L = -log(0.99) ≈ 0.01 (small loss, nearly correct)
- y=1, ŷ=0.01 → L = -log(0.01) ≈ 4.6 (large loss, very wrong)
- y=0, ŷ=0.01 → L = -log(0.99) ≈ 0.01 (small loss, nearly correct)

This loss heavily penalizes confident wrong predictions.

#### Training
```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

nn_history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=50, batch_size=512, validation_split=0.1, callbacks=[early_stop]
)
```

**EarlyStopping — detailed:**
- After each epoch, monitor val_loss
- If val_loss did NOT improve for 5 consecutive epochs (`patience=5`): stop training
- `restore_best_weights=True`: after stopping, roll back to the epoch with lowest val_loss

Why patience=5? A balance:
- Too low (patience=1): might stop prematurely due to noisy val_loss fluctuations
- Too high (patience=20): might overfit before stopping

**epochs=50:** Maximum 50 training iterations over the full dataset. With patience=5, training typically stops at 15-30 epochs.

**batch_size=512:**
Mini-batch gradient descent — process 512 samples, compute gradients, update weights. Repeat.
- One epoch = `ceil(n_train/512)` weight updates
- For 85,923 training samples: `ceil(85923/512)` ≈ 168 updates per epoch
- Larger batch → more stable gradient estimates → smoother training
- 512 is a good choice: fits in GPU VRAM, fast, stable

**validation_split=0.1:**
Takes the LAST 10% of X_train_scaled as validation. So if n_train=85,923: ~8,592 validation samples, ~77,331 true training samples. Note: NOT shuffled before splitting — the last 10% of records are used.

**callbacks=[early_stop]:** After each epoch, Keras calls `early_stop.on_epoch_end()` which checks if val_loss improved. If not, increments patience counter.

#### Learning Curves
```python
ax1.plot(nn_history.history['loss'],     label='Train Loss')
ax1.plot(nn_history.history['val_loss'], label='Validation Loss')
ax2.plot(nn_history.history['accuracy'], label='Train Accuracy')
ax2.plot(nn_history.history['val_accuracy'], label='Validation Accuracy')
```
`nn_history.history` is a dictionary stored during `model.fit()`:
- `'loss'`: list of training losses per epoch
- `'val_loss'`: list of validation losses per epoch
- `'accuracy'`: list of training accuracies per epoch
- `'val_accuracy'`: list of validation accuracies per epoch

**What healthy learning curves look like:**
- Both loss curves decrease, then plateau
- Small gap between train and validation (overfitting = large gap)
- val_loss should not start increasing (early stopping handles this)

#### Prediction
```python
y_prob_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()
y_pred_nn = (y_prob_nn >= 0.5).astype(int)
```
`nn_model.predict()` → Returns numpy array of shape (23879, 1) — the sigmoid output for each test sample.
`.flatten()` → Converts (23879, 1) to (23879,) — removes the redundant dimension.

`(y_prob_nn >= 0.5)` → Boolean array: True where probability ≥ 0.5.
`.astype(int)` → Converts True→1 (canceled), False→0 (not canceled).

The 0.5 threshold can be adjusted: lower threshold (e.g., 0.3) → higher recall (catch more cancellations) but lower precision (more false alarms). For a hotel, this trade-off depends on the cost of missed cancellations vs false alarms.

---

## Section 7: Comparison Analysis

### Summary Table
```python
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    ...
})
comparison_df.style.highlight_max(axis=0, color='#d5f5e3')
```
Creates a pandas DataFrame from the results dictionary. `comparison_df.style.highlight_max(axis=0)` applies a green background to the maximum value in each column, visually identifying the best-performing model for each metric. `axis=0` = "find max across rows within each column."

### Accuracy Bar Chart
```python
bars = ax.bar(model_names, accuracies, color=colors, edgecolor='black', width=0.6)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
```
Each bar height = accuracy value (0 to 1). Labels show percentage with 2 decimal places.
`PercentFormatter(xmax=1.0)` formats 0.85 as "85.00%" on the y-axis.

**Interpretation:** The bar chart is required by the project template. It provides an instant visual comparison — taller bar = better accuracy. However, note that accuracy alone can be misleading for imbalanced datasets; F1 and AUC are more reliable metrics.

### Grouped Bar Chart: Precision, Recall, F1
```python
metrics_df = pd.DataFrame({'Precision': [...], 'Recall': [...], 'F1-Score': [...]}, index=model_names)
metrics_df.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black', rot=15, width=0.7)
```
`metrics_df.plot(kind='bar')` creates a grouped bar chart — 3 bars side by side for each model.
`rot=15` rotates x-axis model names 15 degrees to prevent overlap.

**Interpretation:** A model with high F1 but low recall is missing too many cancellations. A model with high recall but low precision is raising too many false alarms. The best model has both high precision AND high recall (high F1).

### Confusion Matrices
```python
cm = confusion_matrix(y_test, results[model_name]['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Canceled', 'Canceled'])
disp.plot(ax=axes[i], cmap=color, colorbar=False)
```
**Confusion Matrix reading:**
```
               Predicted: Not Canceled    Predicted: Canceled
Actual: Not Canceled    [ TN=14900  ,     FP=250    ]
Actual: Canceled        [ FN=1200   ,     TP=7500   ]
```
- **TN:** Model correctly predicted "will not cancel" — good
- **TP:** Model correctly predicted "will cancel" — good
- **FP (Type I error):** Hotel prepared for cancellation but guest arrived — waste of resources but not catastrophic
- **FN (Type II error):** Hotel thought guest was coming but they canceled — room sat empty, revenue lost

For hotels, **FN is more costly** — missing a cancellation means an unoccupied room with no chance to fill it. So we want high Recall (low FN).

`cmap=color` — Each model gets its own color scheme (Blues, Greens, etc.) for visual distinction.
`colorbar=False` — Removes the color scale bar (cleaner appearance in subplot grid).

### ROC Curves
```python
for model_name, color, ls in zip(model_names, line_colors, line_styles):
    fpr = results[model_name]['fpr']
    tpr = results[model_name]['tpr']
    auc_val = results[model_name]['auc']
    ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
            label=f'{model_name} (AUC = {auc_val:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Baseline (AUC = 0.5)')
```

**ROC Curve construction:**
`roc_curve(y_test, y_prob)` tries every possible threshold t from 0 to 1:
- At very low t (e.g., t=0.01): predict EVERYTHING as canceled → TPR=1.0, FPR=1.0 (top-right corner)
- At very high t (e.g., t=0.99): predict NOTHING as canceled → TPR=0.0, FPR=0.0 (bottom-left corner)
- Intermediate thresholds: trade-off between catching cancellations and false alarms

**AUC interpretation:**
- AUC=1.0: Perfect separation (model assigns P>threshold to all true positives, P<threshold to all true negatives)
- AUC=0.5: Random classifier (diagonal line)
- AUC=0.85: Model is 85% likely to score a random positive higher than a random negative

**The `_` in `fpr, tpr, _ = roc_curve(...)`:** The third return value is the threshold array. We don't need it for plotting, so we discard it.

`ax.plot([0, 1], [0, 1], 'k--')` — Plots the diagonal "random baseline" reference line. Any good model must curve above this.

---

## Section 8: K-Means Clustering

### Remove Target
```python
X_unsup = X_train_scaled.copy()
```
We use `X_train_scaled` (the feature matrix WITHOUT `is_canceled`). The target was already separated in Section 4. The unsupervised model receives ONLY the 27 features — no labels.

### Elbow Method
```python
sample_idx = np.random.choice(len(X_unsup), size=min(10000, len(X_unsup)), replace=False)
X_sample   = X_unsup[sample_idx]
```
`np.random.choice(n, size=10000, replace=False)` → random sample of 10,000 row indices without replacement.
`X_unsup[sample_idx]` → selects those rows from the scaled feature matrix.

Why subsample? Running KMeans with k=1 to 10, each with n_init=5 restarts, on 95,000 samples would take several minutes. 10,000 samples gives reliable elbow detection in seconds.

```python
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=RANDOM_STATE)
    km.fit(X_sample)
    inertia_values.append(km.inertia_)
```
`km.inertia_` = WCSS (Within-Cluster Sum of Squares):
`WCSS = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²`
where Cᵢ is cluster i and μᵢ is the centroid of cluster i.

WCSS always decreases as k increases (with k=n_samples, each point is its own cluster → WCSS=0).
The "elbow" is where the decrease becomes less steep — marginal benefit of adding another cluster drops.

`ax.axvline(x=2, color='red', linestyle='--')` draws a vertical red dashed line at k=2 indicating our chosen optimal k.

### K-Means Final Model
```python
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, random_state=RANDOM_STATE)
kmeans.fit(X_unsup)
cluster_labels = kmeans.labels_
```

**K-Means++ initialization:**
1. Choose first centroid randomly from data points
2. For each subsequent centroid: choose a data point with probability proportional to squared distance from nearest existing centroid
3. This spreads centroids far apart → avoids worst-case random initializations → faster, better convergence

**K-Means algorithm (per run):**
1. Initialize 2 centroids (via k-means++)
2. **Assign step:** Assign each point to nearest centroid (by Euclidean distance)
3. **Update step:** Recompute each centroid as mean of its assigned points
4. Repeat 2-3 until convergence (centroids stop moving significantly) or `max_iter=300` reached
5. Return cluster assignments and inertia

`n_init=10` — Run 10 times from different random starts, keep the run with lowest inertia. With k-means++, 10 restarts is usually sufficient.

`cluster_labels = kmeans.labels_` — Array of shape (n_train_samples,) where each value is 0 or 1.

### PCA Visualization
```python
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_unsup[:5000])
cluster_sample = cluster_labels[:5000]
```
**PCA algorithm:**
1. Center data (subtract mean)
2. Compute covariance matrix: `Σ = (1/n)·X^T·X`
3. Find eigenvectors (principal components) and eigenvalues
4. Sort eigenvectors by eigenvalue (highest variance first)
5. Project data onto top 2 eigenvectors: `X_pca = X·V[:, :2]`

Result: 27-dimensional data → 2-dimensional data for scatter plotting.

`pca.explained_variance_ratio_` — Array of [var_PC1/total_var, var_PC2/total_var]. Tells what fraction of original variance is captured. If only 15% total, the 2D plot misses 85% of the data structure — clusters may appear blended even if separable in 27D.

**Side-by-side plots:**
- Left: Colored by K-Means cluster assignment (what the unsupervised algorithm found)
- Right: Colored by actual `is_canceled` label (the ground truth)

Comparing these visualizes how well K-Means discovered the true structure. If both plots look similar, K-Means successfully grouped bookings by cancellation likelihood without ever seeing the labels.

### Cluster Profile
```python
ct = pd.crosstab(X_train_df['cluster'], X_train_df['actual_label'],
                  rownames=['Cluster'], colnames=['Actual (is_canceled)'])
```
`pd.crosstab()` creates a 2×2 cross-tabulation showing: for K-Means Cluster 0 and Cluster 1, how many samples are actually class 0 and how many are class 1.

**Example interpretation:**
```
Actual (is_canceled)     0       1
Cluster
0                      52000   5000    → Cluster 0 ≈ "not canceled" group
1                      20000  30000    → Cluster 1 ≈ "canceled" group
```
If one cluster maps cleanly to "canceled" and the other to "not canceled," the unsupervised method successfully recovered the class structure — strong evidence that the 27 features genuinely discriminate between cancellations and non-cancellations.

---

## Section 9: Conclusion
A **markdown cell** summarizing:
- Which model performed best (by F1-Score and AUC)
- Why some models may underperform (Naive Bayes independence assumption, KNN scalability)
- Key challenges (data leakage detection, country cardinality, class imbalance, NN training time)
- Future improvements (SMOTE, hyperparameter tuning, ensemble methods)

The code cell in Section 9 prints a formatted final summary using the stored `results` dictionary.

---

# PART 4: HOW TO SUBMIT AND DEMO

## Colab Notebook Checklist
- [ ] Upload `11.csv` to Google Drive → `MyDrive/CSE422/11.csv`
- [ ] Open `CSE422_Group11_Hotel_Booking.ipynb` in Colab (File → Upload Notebook)
- [ ] Run all cells: Runtime → Run All
- [ ] Verify all 15 figures are generated
- [ ] Save generated PNG files for report

## Common Questions Your Faculty May Ask

**Q: Why did you drop `reservation_status`?**
A: Data leakage. It directly reveals the cancellation outcome. A model trained with this feature would "cheat" and fail completely in production.

**Q: Why use stratified splitting?**
A: The dataset is imbalanced (~63/37). Random splitting might give different class ratios in train vs test, making evaluation unreliable. Stratified splitting preserves the 63/37 ratio in both sets.

**Q: Why scale features for KNN but not Decision Tree?**
A: KNN uses Euclidean distance — without scaling, high-range features (adr) dominate. Decision Tree splits by thresholds — relative order matters, not absolute scale.

**Q: What is EarlyStopping doing in the Neural Network?**
A: It monitors validation loss and stops training when it stops improving for 5 consecutive epochs, then restores the best weights. This prevents overfitting.

**Q: Why does K-Means not use labels?**
A: K-Means is unsupervised — it only uses the feature values to find natural groupings. The target column is removed. We then compare what it found with the actual labels to validate that the data has learnable structure.

**Q: Which metric is most important for this problem?**
A: F1-Score and AUC-ROC. Accuracy can be misleading for imbalanced data (a model predicting all "not canceled" would get ~63% accuracy). F1 balances precision and recall. AUC measures discrimination ability across all thresholds.

**Q: Why Gaussian Naive Bayes likely performs worst?**
A: GaussianNB assumes (1) feature independence and (2) Gaussian distributions. Both are violated here — features like `deposit_type` and `previous_cancellations` are strongly correlated, and most features are highly skewed (not Gaussian).
