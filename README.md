
# Titanic Survival Prediction 🚢

This repository contains a machine learning project for the classic Kaggle competition: **"Titanic - Machine Learning from Disaster"**. The goal is to build a model that predicts whether a passenger survived the Titanic shipwreck based on their personal and travel data.

This project notebook (`Titanic_kaggle_competition.ipynb`) demonstrates a complete machine learning workflow, including data cleaning, feature engineering, and building a hybrid model that achieves **83.15%** accuracy on the validation set.

-----

## ✨ Technologies Used

-----

## 🎯 Problem Statement

Given the `train.csv` dataset, the task is to build a predictive model that accurately determines the 'Survived' status (0 = No, 1 = Yes) for passengers in the `test.csv` dataset. The final predictions are compiled into `submission.csv`.

-----

## 📂 Project Structure

```
.
├── Titanic_kaggle_competition.ipynb    # Main Jupyter Notebook with all analysis and modeling
├── train.csv                           # Training data with features and survival labels
├── test.csv                            # Test data with features, awaiting prediction
└── submission.csv                      # Generated prediction file for Kaggle
```

-----

## 💡 Methodology

The solution is built following these steps detailed in the notebook:

1.  **Data Loading:** Import `train.csv` and `test.csv` using Pandas.
2.  **Exploratory Data Analysis (EDA):**
      * Analyzed missing values, identifying gaps in `Age` (177 entries), `Cabin` (687 entries), and `Embarked` (2 entries).
      * Visualized feature correlations using a Seaborn heatmap.
3.  **Feature Engineering & Preprocessing:**
      * **Dropped irrelevant columns:** `PassengerId`, `Name`, `Ticket`, and `Cabin` (due to \>70% missing data) were removed.
      * **Imputed missing values:**
          * `Age`: Filled missing entries with the median age.
          * `Embarked`: Dropped the two rows with missing values.
          * `Fare` (in test set): Filled one missing fare with the median from the training set.
      * **Encoded categorical features:**
          * `Sex`: Mapped to `0` (male) and `1` (female).
          * `Embarked`: Converted (S, C, Q) to numerical values using `LabelEncoder`.
4.  **Modeling:**
      * A hybrid **Voting Classifier** was chosen for robust prediction.
      * It combines two models:
        1.  `LogisticRegression`
        2.  `XGBClassifier`
      * The models are combined using `voting='soft'` to average their predicted probabilities, leading to a more stable and accurate result.

-----

## 📊 Results

The hybrid model achieved a validation accuracy of **83.15%** on the 20% hold-out test set.

The correlation heatmap below shows the relationship between features after preprocessing. As expected, **`Sex`** (being female) has the highest positive correlation with survival.

-----

## 🚀 How to Use

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/valterugowtham/titanic-survival-prediction.git
    cd titanic-survival-prediction
    ```

2.  **Install dependencies:**
    (It's recommended to use a virtual environment)

    ```bash
    pip install pandas scikit-learn xgboost jupyter seaborn matplotlib
    ```

3.  **Run the notebook:**
    Open and run the `Titanic_kaggle_competition.ipynb` notebook in Jupyter.

    ```bash
    jupyter notebook Titanic_kaggle_competition.ipynb
    ```

4.  **Get Output:**
    Running the notebook will execute the complete analysis and generate the `submission.csv` file, which is ready for submission to Kaggle.
