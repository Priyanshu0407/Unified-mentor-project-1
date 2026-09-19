
# UAC Predictive Forecasting — Professional Streamlit Dashboard

## What this project does
- Cleans and standardizes the UAC daily dataset.
- Checks missing calendar dates and missing values.
- Builds lag, rolling, flow-pressure, and calendar features.
- Compares Naive, SARIMA, and Gradient Boosting models.
- Forecasts HHS care load for 7/14/30 days.
- Shows uncertainty bands using quantile Gradient Boosting.
- Provides transfer scenarios: Normal, +20%, +40%.
- Estimates short-term discharge/placement demand.
- Flags potential capacity stress.
- Provides a professional Streamlit dashboard.

## Dataset expected
The Excel/CSV should contain these six logical columns:
1. Date
2. Children apprehended and placed in CBP custody
3. Children in CBP custody
4. Children transferred out of CBP custody
5. Children in HHS Care
6. Children discharged from HHS Care

The app automatically maps common variations of these names.

## Important data-handling choice
Missing calendar dates are not assumed to mean zero activity. The app reindexes to daily frequency and interpolates numeric observations for modeling, while retaining a missing-date flag. For a formal research paper, document and justify this treatment after checking the source/data dictionary.

## Run
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Then upload the actual Excel/CSV file in the sidebar.

## Recommended project/report structure
1. Introduction
2. Problem statement
3. Dataset and data quality
4. Exploratory data analysis
5. Feature engineering
6. Forecasting methodology
7. Model evaluation
8. Forecast results
9. Capacity-risk analysis
10. Recommendations
11. Limitations
12. Conclusion

## Notes
The dashboard uses recent average transfers/discharges as future-flow assumptions because future exogenous flows are unknown. This is a scenario assumption, not a certainty. For a stronger production model, separately forecast transfers and discharges and feed those forecasts into the care-load model.
