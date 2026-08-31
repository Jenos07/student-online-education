# 📊 Online Education Student Performance Analysis

## 📌 Project Overview

This project analyzes student data from an online education platform to understand how **student engagement and learning activity** are related to academic performance.

The analysis focuses mainly on **total clicks, engagement level, pass status, and final result**. Exploratory data analysis and Logistic Regression are used to identify patterns in student performance and examine whether online engagement can help predict student outcomes.

---

## 🎯 Objectives

* Analyze student performance in an online education environment.
* Understand the relationship between **student engagement and pass rate**.
* Examine the distribution of final student results.
* Identify patterns between learning activity and academic outcomes.
* Build a simple **Logistic Regression model** using `total_clicks` to predict `pass_flag`.
* Visualize important findings using charts.

---

## 📂 Dataset

The dataset contains **32,593 student records** and **14 features**.

### Main Features

| Feature             | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `id_student`        | Unique student identifier                                  |
| `gender`            | Student gender                                             |
| `region`            | Student region                                             |
| `highest_education` | Highest education qualification                            |
| `studied_credits`   | Number of credits studied                                  |
| `imd_band`          | Socio-economic/deprivation band                            |
| `total_clicks`      | Total online learning activity/clicks                      |
| `avg_score`         | Average student score                                      |
| `engagement_level`  | Low, Medium, or High engagement                            |
| `performance_level` | Low, Medium, or High performance                           |
| `risk_level`        | Student risk category                                      |
| `pass_flag`         | Binary pass indicator                                      |
| `dropout_flag`      | Binary dropout indicator                                   |
| `final_result`      | Final result such as Pass, Fail, Withdrawn, or Distinction |

The dataset structure and feature information are present in the notebook.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine Learning
* **Logistic Regression** – Prediction model
* **StandardScaler** – Feature scaling
* **Google Colab / Jupyter Notebook**

---

## 🔍 Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas:

```python
import pandas as pd

df = pd.read_csv("online_education_dataset.csv")
```

### 2. Data Exploration

Basic dataset exploration is performed using:

```python
df.head()
df.tail()
df.info()
```

The dataset contains 32,593 rows and 14 columns.

### 3. Final Result Analysis

The distribution of final results was analyzed.

| Final Result | Students |
| ------------ | -------: |
| Pass         |   12,361 |
| Withdrawn    |   10,156 |
| Fail         |    7,052 |
| Distinction  |    3,024 |

### 4. Engagement vs Pass Rate

The project calculates the average `pass_flag` for each engagement level.

| Engagement Level | Pass Rate |
| ---------------- | --------: |
| Low              |    19.92% |
| Medium           |    57.15% |
| High             |    78.14% |

The analysis shows a strong difference in observed pass rates across engagement groups, with highly engaged students having the highest pass rate.

### 5. Data Visualization

A bar chart is created to visualize the relationship between:

**Engagement Level → Pass Rate**

```python
sns.barplot(
    x="engagement_level",
    y="pass_flag",
    data=df,
    order=["Low", "Medium", "High"]
)
```

### 6. Missing Value Handling

Missing values in `total_clicks` and `pass_flag` are handled by replacing them with zero:

```python
df["total_clicks"] = df["total_clicks"].fillna(0)
df["pass_flag"] = df["pass_flag"].fillna(0)
```

### 7. Machine Learning Model

A **Logistic Regression** model is trained using `total_clicks` as the input feature and `pass_flag` as the target.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X = df[["total_clicks"]]
y = df["pass_flag"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)
```

The notebook reports a model coefficient of approximately **1.616**, indicating a positive model relationship between the scaled `total_clicks` feature and the modeled pass outcome.

---

## 📈 Key Findings

* The dataset contains **32,593 student records**.
* `Pass` is the most common final result, followed by `Withdrawn`, `Fail`, and `Distinction`.
* Students with **High engagement** have a substantially higher observed pass rate than students with Medium or Low engagement.
* The observed pass rate increases from approximately **19.9% for Low engagement** to **78.1% for High engagement**.
* Logistic Regression was applied to study the relationship between online activity (`total_clicks`) and passing.
* The model coefficient for `total_clicks` is positive.

---

## 💡 Business / Educational Insight

The analysis suggests that **student engagement is an important indicator associated with academic outcomes**.

An online education platform could potentially use engagement information to identify students who may need additional support. For example, students showing consistently low engagement could be identified for reminders, mentoring, additional learning resources, or academic intervention.

> **Note:** The current notebook demonstrates an association/predictive modeling approach. It does not establish that increasing clicks directly causes students to pass.

---

## 🚀 Future Improvements

The current model uses only `total_clicks` as the predictive feature. The project can be improved by:

* Including `avg_score`
* Including `studied_credits`
* Including `engagement_level`
* Including `performance_level`
* Including `risk_level`
* Encoding categorical variables
* Splitting the data into training and testing sets
* Evaluating model accuracy
* Generating a confusion matrix
* Calculating Precision, Recall, and F1-score
* Comparing Logistic Regression with other ML algorithms
* Building an interactive dashboard using **Power BI**
* Developing an early-warning system for students at risk of failing or withdrawing

---

## 📁 Project Structure

```text
Online-Education-Student-Performance/
│
├── Untitled2.ipynb
├── online_education_dataset.csv
└── README.md
```

---

## ▶️ How to Run

### Using Google Colab

1. Open the `.ipynb` file in Google Colab.
2. Upload the dataset CSV file.
3. Update the CSV file path if required.
4. Run the notebook cells sequentially.

### Using Jupyter Notebook

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then open:

```bash
jupyter notebook Untitled2.ipynb
```

---

## 👩‍💻 Author

**Jenos**

BCA Student | Aspiring Data Analyst

---

## 📜 License

This project is created for **educational and portfolio purposes**.
