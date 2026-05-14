# Salary Analysis & Prediction

## Project Overview
This project analyzes global data science salary trends using Python and machine learning techniques. The goal of this project is to understand salary distributions, identify important trends in the data science industry, visualize insights, and build a predictive model for salary estimation.

The project was completed as part of a Data Science internship task focusing on:
- Data Collection & Understanding
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Data Visualization
- Predictive Modeling

---

## Dataset
Dataset Used:  
Data Science Job Salaries Dataset  
https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries

License: CC0 Public Domain

### Dataset Features
The dataset contains information about:
- Job titles
- Experience levels
- Employment types
- Company sizes
- Remote work ratios
- Employee residence
- Company location
- Salaries in USD

---

# Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Project Tasks

## Task 1: Data Collection & Dataset Understanding
- Loaded the dataset
- Explored dataset structure
- Identified columns and data types
- Analyzed dataset size and features

## Task 2: Data Cleaning & Preprocessing
- Checked missing values
- Removed duplicates
- Formatted and standardized data
- Prepared dataset for analysis

## Task 3: Exploratory Data Analysis (EDA)
Performed analysis to identify:
- Salary trends
- Highest paying job roles
- Impact of experience level on salary
- Remote work trends
- Company size analysis

## Task 4: Data Visualization
Created visualizations including:
- Salary distribution histogram
- Top job titles bar chart
- Experience level vs salary boxplot
- Remote work distribution pie chart
- Correlation heatmap

## Task 5: Predictive Modeling
Built a machine learning regression model to:
- Predict salaries using job-related features
- Evaluate model performance using:
  - Mean Absolute Error (MAE)
  - R² Score

---

# Project Structure

```text
Data-Science-Salary-Analysis/
│
├── data/
│   └── ds_salaries.csv
├── week03_tasks.ipynb
├── week03_tasks.pyw
├── src/
│   ├── Salary_Distribution_Histogram.png
│   ├── Top_10_Jobs_Titles.png
│   ├── Experience_Level_Vs_Salary.png
│   └── Remote_Work_Distribution.png
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Data-Science-Salary-Analysis.git
```

Install required libraries:

```bash
pip install -r requirements.txt
```

Run the Jupyter Notebook:

```bash
jupyter notebook
```

---

# Machine Learning Model
The project uses a Linear Regression model from Scikit-learn to predict salaries based on:
- Experience level
- Employment type
- Job title
- Company size
- Remote work ratio

---

# Key Insights
- Senior-level roles generally receive higher salaries
- Remote jobs are increasingly common in data science
- Certain job titles have significantly higher average salaries
- Large companies tend to offer better compensation

---

# Skills Demonstrated
- Data Cleaning
- Data Analysis
- Data Visualization
- Machine Learning
- Regression Modeling
- Statistical Analysis
- Python Programming
