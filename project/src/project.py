import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, f1_score

# ==========================================
# GOLBAL CONFIGURATION FOR REPRODUCIBILITY (STEP 6)
# ==========================================
# تنظیم دقیق Random Seed برای تضمین تکرارپذیری تمام نتایج
SEED = 42
np.random.seed(SEED)

# ==========================================
# 1. معرفی مسئله و داده‌ها (Data Introduction)
# ==========================================
print("--- Step 1: Data Introduction & Loading ---")
# بارگذاری داده‌ها
try:
    df = pd.read_excel(r'project/data/Telco_customer_churn.xlsx')
    print("Data Loaded Successfully.")
except FileNotFoundError:
    print("Error: File 'Telco_customer_churn.xlsx' not found. Please upload the file.")
    exit()

# نمایش اطلاعات کلی
print(f"\nShape of dataset: {df.shape}")
print("\nData Types:")
print(df.dtypes.value_counts())

# بررسی توزیع کلاس هدف (Target Imbalance)
target_col = 'Churn Value'  # فرض بر این است که نام ستون هدف Churn است
print(f"\nTarget Distribution ({target_col}):")
print(df[target_col].value_counts(normalize=True))

# # بررسی مقادیر گم‌شده
# print("\nMissing Values per column (Top 5):")
# print(df.isnull().sum().sort_values(ascending=False).head(5))

# ==========================================
# 2. پیش‌پردازش داده‌ها (Data Preprocessing)
# ==========================================
print("\n--- Step 2: Data Preprocessing & Effect Analysis ---")

# الف) تمیزکاری داده‌ها
# ستون Total Charges معمولاً به صورت Object است (رشته) و ممکن است فضای خالی داشته باشد
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# پر کردن مقادیر NaN در Total Charges با 0
# منطق: این‌ها مشتریانی با Tenure Months = 0 هستند که هنوز قبض نگرفته‌اند
df['Total Charges'] = df['Total Charges'].fillna(0)

# # حذف CustomerID (چون ویژگی پیش‌بینی‌کننده نیست)
# if 'CustomerID' in df.columns:
#     df.drop('CustomerID', axis=1, inplace=True)

# # پر کردن مقادیر گم‌شده (Imputation)
# # برای متغیرهای عددی از میانه (Median) استفاده می‌کنیم تا به داده‌های پرت حساس نباشد
# num_cols = df.select_dtypes(include=['float64', 'int64']).columns
# cat_cols = df.select_dtypes(include=['object']).columns

# imputer = SimpleImputer(strategy='median')
# df[num_cols] = imputer.fit_transform(df[num_cols])

# # ب) تبدیل متغیر هدف به 0 و 1
# df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

# # ج) تبدیل متغیرهای کتگوریال (One-Hot Encoding)
# df_encoded = pd.get_dummies(df, drop_first=True)

# # جدا کردن X و y
# X = df_encoded.drop(target_col, axis=1)
# y = df_encoded[target_col]

# لیست ستون‌های حذفی (شامل ستون‌های نشتی‌دهنده و اطلاعات مکانی غیرضروری)
cols_to_drop = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude',
    'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'
]

# حذف ستون‌ها
df_clean = df.drop(columns=cols_to_drop)

# جدا کردن متغیر هدف (y) و ویژگی‌ها (X)
X = df_clean.drop(columns=['Churn Value'])
y = df_clean['Churn Value']

# # د) تقسیم داده‌ها (قبل از نرمال‌سازی برای جلوگیری از Data Leakage)
# # استفاده از stratify برای حفظ نسبت کلاس‌ها در آموزش و تست
# X_train_raw, X_test_raw, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=SEED
# )

# # هـ) نرمال‌سازی (Standardization) و تحلیل اثر آن
# scaler = StandardScaler()
# # فیت کردن فقط روی داده‌های آموزش
# X_train = scaler.fit_transform(X_train_raw)
# X_test = scaler.transform(X_test_raw)

# شناسایی ستون‌های عددی و دسته‌ای
numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
categorical_cols = [col for col in X.columns if col not in numerical_cols]

# تعریف تبدیل‌گر (Transformer)
preprocessor = ColumnTransformer(
    transformers=[
        # استانداردسازی داده‌های عددی (Mean=0, Std=1)
        ('num', StandardScaler(), numerical_cols),
        # انکدینگ داده‌های دسته‌ای (One-Hot)
        # drop='first' برای جلوگیری از هم‌خطی (Multicollinearity)
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ],
    verbose_feature_names_out=False
)

# تقسیم داده‌ها به آموزش و تست (با رعایت نسبت کلاس‌ها)
# random_state=42 برای تکرارپذیری
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# اعمال پیش‌پردازش روی داده‌های آموزش و تست
# نکته: fit فقط روی داده‌های آموزش انجام می‌شود تا نشت داده (Data Leakage) رخ ندهد
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# استخراج نام ویژگی‌ها بعد از انکدینگ (برای استفاده‌های بعدی در تحلیل اهمیت ویژگی‌ها)
feature_names = preprocessor.get_feature_names_out()

# --- ANALYSIS: Visualization of Preprocessing Effect ---
# نمایش تأثیر استانداردسازی روی توزیع یک ویژگی (مثلاً Monthly Charges)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(X_train_raw['Monthly Charges'], kde=True, color='blue')
plt.title('Before Scaling (Original Distribution)')
plt.xlabel('Monthly Charges')

plt.subplot(1, 2, 2)
# پیدا کردن ایندکس ستون Monthly Charges در آرایه نامپای
mc_idx = list(feature_names).index('Monthly Charges')
sns.histplot(X_train[:, mc_idx], kde=True, color='green')
plt.title('After StandardScaler (Mean=0, Std=1)')
plt.xlabel('Monthly Charges (Scaled)')
plt.tight_layout()
plt.show()

# ==========================================
# 3. کاهش ابعاد (Dimensionality Reduction - PCA)
# ==========================================
print("\n--- Step 3: Dimensionality Reduction (PCA) & Cost-Benefit Analysis ---")

# هدف: حفظ 95 درصد واریانس داده‌ها
pca = PCA(n_components=0.95, random_state=SEED)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Original Features: {X_train.shape[1]}")
print(f"Features after PCA (95% Variance): {X_train_pca.shape[1]}")

# --- ANALYSIS: Speed vs Accuracy Trade-off ---
print("\nRunning quick comparison: PCA vs. No-PCA (using Random Forest)...")

def benchmark_model(X_tr, y_tr, X_te, y_te, name):
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1)
    start_time = time.time()
    clf.fit(X_tr, y_tr)
    train_time = time.time() - start_time
    
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    return train_time, acc

time_raw, acc_raw = benchmark_model(X_train, y_train, X_test, y_test, "Original Data")
time_pca, acc_pca = benchmark_model(X_train_pca, y_train, X_test_pca, y_test, "PCA Data")

print(f"{'Method':<15} | {'Training Time (s)':<18} | {'Accuracy':<10}")
print("-" * 50)
print(f"{'Original':<15} | {time_raw:.4f}             | {acc_raw:.4f}")
print(f"{'With PCA':<15} | {time_pca:.4f}             | {acc_pca:.4f}")

# ==========================================
# 4 & 5. انتخاب مدل‌ها و تنظیم هایپرپارامترها (Model Selection & Tuning)
# ==========================================
print("\n--- Step 4 & 5: Model Selection (Bagging vs Boosting) & Hyperparameter Tuning ---")

# تعریف مدل‌ها
models = {
    'Random Forest (Bagging)': RandomForestClassifier(random_state=SEED, class_weight='balanced'),
    'Gradient Boosting (Boosting)': GradientBoostingClassifier(random_state=SEED)
}

# فضای جستجوی پارامترها (Search Space)
param_grids = {
    'Random Forest (Bagging)': {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    },
    'Gradient Boosting (Boosting)': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
}

best_estimators = {}
results = {}

# استفاده از Stratified K-Fold در GridSearch
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for name, model in models.items():
    print(f"\nTraining {name} with GridSearchCV...")
    
    # تنظیم Step 5: Systematic Search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=cv,
        scoring='roc_auc', # 'f1', # تمرکز بر F1 Score به دلیل نامتوازن بودن دیتا
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    best_estimators[name] = grid.best_estimator_
    results[name] = grid.best_score_
    
    print(f"Best Params for {name}: {grid.best_params_}")
    print(f"Best CV F1-Score: {grid.best_score_:.4f}")

# ==========================================
# 6 & 7. ارزیابی نهایی، تحلیل و مصورسازی (Evaluation & Visualization)
# ==========================================
print("\n--- Step 6 & 7: Final Evaluation, Reproducibility & Visualization ---")

plt.figure(figsize=(14, 6))

# حلقه برای ارزیابی هر دو مدل بهینه شده
for i, (name, model) in enumerate(best_estimators.items()):
    
    # # پیش‌بینی روی داده تست (داده‌هایی که مدل هرگز ندیده است)
    # y_pred = model.predict(X_test)
    # y_prob = model.predict_proba(X_test)[:, 1]
    
    # اصلاح استراتژی ۴: محاسبه احتمالات و اعمال دستی آستانه 0.35
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.35).astype(int)

    
    # گزارش متنی دقیق
    print(f"\n{'='*20} {name} Evaluation {'='*20}")
    print(classification_report(y_test, y_pred))
    
    # 1. رسم Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if i==0 else 'Greens', alpha=0.6 if i==1 else 1)
    plt.title('Confusion Matrix Comparison')
    
    # 2. رسم ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# تنظیمات نهایی نمودار ROC
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# --- ANALYSIS: Learning Curves (Why Overfitting?) ---
print("\nGenerating Learning Curves for Gradient Boosting (Best Model Analysis)...")

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (F1)")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

# رسم منحنی یادگیری برای مدل GB (معمولاً مدل قوی‌تر است)
plot_learning_curve(best_estimators['Gradient Boosting (Boosting)'], 
                    "Learning Curve (Gradient Boosting)", 
                    X_train, y_train, cv=cv, n_jobs=-1)