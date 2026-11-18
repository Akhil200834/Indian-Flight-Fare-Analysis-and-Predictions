# Indian-Flight-Fare-Analysis-and-Predictions
Bleep Education mini project 1


An end-to-end Exploratory Data Analysis (EDA) and Machine Learning regression project analyzing domestic flight ticket pricing dynamics across major Indian airlines and metro routes.

## Problem Statement
As a Data Analyst for an Indian Online Travel Agency (OTA), the objective of this project is to evaluate how airline positioning, route distance, journey characteristics (stops, duration, departure/arrival timing), and advance booking windows (`days_left`) influence flight ticket prices, and to build a predictive model for fare estimation.

## Repository Structure
```text
Indian-Flight-Fare-Analysis-and-Predictions/
├── data/
│   └── Clean_Dataset.csv          # Primary dataset (~300,153 flight booking records)
├── notebooks/
│   └── flight_analysis.ipynb      # Complete 54-question EDA & Random Forest modeling notebook
├── outputs/
│   └── graphs/                    # Generated visualizations and charts
├── src/                           # Helper scripts and modular utilities
├── README.md                      # Project documentation and key findings
└── requirements.txt               # Python dependencies
```

## Dataset Overview
Primary Dataset: data/Clean_Dataset.csv (300,153 rows x 11 features)

Airlines Covered: Vistara, Air India, IndiGo, GO_FIRST, AirAsia, SpiceJet

Metro Cities: Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai

Cabin Classes: Economy & Business

## Tech Stack
Language: Python 3

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Statistical Analysis & ML: SciPy, Scikit-Learn (Random Forest Regressor)

## Key Analytical Sections (notebooks/flight_analysis.ipynb)
Data Acquisition & Preprocessing (Q1–Q10): Schema inspection, memory downcasting, missing value verification, duplicate analysis, and categorical consistency checks.

Univariate Analysis (Q11–Q20): Ticket price distribution skewness, IQR outlier detection, percentile breakdowns, and categorical frequency distributions.

Airline-Wise Fare Analysis (Q21–Q28): Mean vs. median fare rankings, price variance across carriers, and Economy vs. Business fare gaps.

Route, Source & Destination Analysis (Q29–Q37): Route feature engineering (Route_City), directional price symmetry (e.g., Delhi–Mumbai vs. Mumbai–Delhi), and median price heatmaps.

Booking Window Analysis (Q38–Q46): Fare trends across days_left buckets (0–3, 4–7, 8–15, 16–30, 31–45, 46–60 days).

Duration, Stops & Time-of-Day Analysis (Q47–Q54): Pearson/Spearman correlations and fare variations across departure/arrival time slots.

Fare Prediction Model: Random Forest Regressor trained to predict ticket fares based on route, airline, class, stops, duration, and booking lead time.

## Predictive Model Results
R² Score: 0.9763
MAE: ₹1792.13
RMSE: ₹3498.24


## How to Run Locally
1. **Clone the repository:**
   `git clone https://github.com/Akhil200834/Indian-Flight-Fare-Analysis-and-Predictions.git`
2. **Navigate to the project directory:**
   `cd Indian-Flight-Fare-Analysis-and-Predictions`
3. **Install the required dependencies:**
   `pip install -r requirements.txt`
4. **Execute the analysis notebook:**
   Open `notebooks/flight_analysis.ipynb` in VS Code or Jupyter Notebook and click **Run All** to reproduce all 54 EDA outputs, visualizations, and Random Forest model predictions.

