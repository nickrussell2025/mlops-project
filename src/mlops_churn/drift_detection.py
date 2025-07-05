import json
from datetime import datetime, timedelta
import pandas as pd
import psycopg2
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

DB_CONN = dict(host="localhost", port="5432", database="monitoring", user="postgres", password="example")


def get_recent_predictions():
    """Get recent prediction data from database"""
    
    with psycopg2.connect(**DB_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT input_data FROM predictions WHERE timestamp >= %s ORDER BY timestamp DESC LIMIT 50",
                       [datetime.now() - timedelta(hours=24)])
            rows = cur.fetchall()
            return pd.DataFrame([row[0] for row in rows]) if len(rows) >= 10 else None


def log_drift_result(drift_detected, drift_score, drifted_columns, sample_size):
    """Save drift detection results to database"""
    
    with psycopg2.connect(**DB_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO drift_reports (drift_detected, drift_score, feature_name, drift_type, report_data) VALUES (%s, %s, %s, %s, %s)",
                       (drift_detected, drift_score, f"{drifted_columns}_columns_drifted", "data_drift", 
                        json.dumps({"drifted_columns": drifted_columns, "drift_share": drift_score, "sample_size": sample_size})))


def detect_drift():
    """Main drift detection function"""
    
    # load reference data
    reference_df = pd.read_parquet('/workspaces/mlops-project/monitoring/reference_data.parquet')
    
    # get current predictions
    current_df = get_recent_predictions()
    if current_df is None:
        return False
    
    # fix categorical data types
    for col in ['Geography', 'Gender']:
        reference_df[col] = reference_df[col].astype('object')
        current_df[col] = current_df[col].astype('object')
    
    # define evidently schema
    schema = DataDefinition(
        numerical_columns=[col for col in reference_df.columns if col not in ['Geography', 'Gender']],
        categorical_columns=['Geography', 'Gender']
    )
    
    # run drift detection
    report = Report([DataDriftPreset()])
    result = report.run(Dataset.from_pandas(current_df, data_definition=schema), 
                      Dataset.from_pandas(reference_df, data_definition=schema))
    
    # extract results
    drift_data = result.dict()["metrics"][0]["value"]
    drift_share = round(float(drift_data["share"]), 3)
    drifted_columns = int(drift_data["count"])
    dataset_drift = drift_share > 0.5
    
    # log results to database
    log_drift_result(dataset_drift, drift_share, drifted_columns, len(current_df))
    
    # print results
    print(f"DRIFT DETECTED! {drifted_columns} columns drifted, share: {drift_share}" if dataset_drift else f"No drift. Share: {drift_share}")
    
    return True


if __name__ == "__main__":
    detect_drift()