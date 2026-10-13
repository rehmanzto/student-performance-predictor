# Student Performance Predictor

A machine learning web app that estimates a student's final grade (G3, scale 0-20) from background information and, optionally, earlier grades. Built as a learning project: data exploration, model comparison, testing, and a Streamlit interface.

## What it does

- **Early estimate:** predicts G3 from 30 background features (study time, absences, family, etc.) before any grades exist.
- **Later estimate:** predicts G3 using the same features plus the first two period grades (G1, G2).
- **Analytics page:** model quality charts and an exploration of which features relate to grades.

## Screenshots

![Prediction form](docs/app.png)
![Analytics page](docs/analytics.png)

## Results

Scores on a held-out test set of 130 students the models never saw (Random Forest):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Early (no grades) | 2.05 | 2.81 | 0.19 |
| Later (with G1 + G2) | 0.75 | 1.25 | 0.84 |

The early model is only a rough guide: with 5-fold cross-validation its R² was about 0.28, and a single 130-student split is noisy. The later model is much more accurate because earlier grades strongly predict the final grade. The two models are not directly comparable as "early prediction" tools, since the later model has extra information.

## What I found

TODO: write 3 observations in your own words from the Analytics page.
For example: which features mattered most, what surprised you, what the scatter plots showed.

## Limitations

- The data comes from two Portuguese secondary schools (649 students). It is **not validated for Bangladeshi HSC students**.
- Patterns in the data are correlations, not proof of cause and effect.
- The early model explains only part of the variation in final grades. Treat results as rough estimates.

## Run it locally

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-predictor.git
cd student-performance-predictor
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Retrain the models: `python 06_train_and_save.py`

Run the tests: `pip install pytest`, then `python -m pytest`

## Project structure

- `app.py`: Streamlit prediction page
- `pages/1_Analytics.py`: analytics page
- `predictor.py`: loading and prediction logic (tested)
- `06_train_and_save.py`: trains and saves both models
- `01`-`05_*.py`: the exploration and experiment scripts
- `tests/`: automated tests
- `data/`, `models/`: dataset and saved models

## Data and credits

Dataset: Student Performance, UCI Machine Learning Repository. Cortez, P. and Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008). Check the license on the dataset page: https://archive.ics.uci.edu/dataset/320/student+performance