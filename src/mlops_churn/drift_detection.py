import json
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from database import get_db_connection, log_drift_result
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def get_recent_predictions(hours=24, limit=50):
    """Get recent customer data from database"""
    
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        query = """
        SELECT input_data 
        FROM predictions 
        WHERE timestamp >= %s
        ORDER BY timestamp DESC
        LIMIT %s
        """
        
        since_time = datetime.now() - timedelta(hours=hours)
        cur.execute(query, [since_time, limit])
        rows = cur.fetchall()
        cur.close()
        
        if len(rows) < 10:
            logger.warning(f"Only {len(rows)} recent predictions, need at least 10")
            return None
            
        # Convert to DataFrame
        features_list = [row[0] for row in rows]
        current_df = pd.DataFrame(features_list)
        
        logger.info(f"Got {len(current_df)} recent predictions")
        return current_df
        
    except Exception as e:
        logger.error(f"Error getting recent predictions: {e}")
        return None
    finally:
        conn.close()


def detect_drift():
    """Main drift detection function"""
    
    # Load reference data
    try:
        reference_df = pd.read_parquet('../monitoring/reference_data.parquet')
        logger.info(f"Reference data: {reference_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load reference data: {e}")
        return False
    
    # Get recent data
    current_df = get_recent_predictions()
    if current_df is None or len(current_df) < 10:
        logger.warning("Not enough recent data for drift detection")
        return False
    
    # Ensure data types match
    for col in ['Geography', 'Gender']:
        if col in reference_df.columns and col in current_df.columns:
            reference_df[col] = reference_df[col].astype('object')
            current_df[col] = current_df[col].astype('object')
    
    try:
        # Run drift detection
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        
        # Extract actual results from the report
        result_dict = report.as_dict()
        
        # Get dataset-level drift detection
        dataset_drift = result_dict['metrics'][0]['result']['dataset_drift']
        drift_share = result_dict['metrics'][0]['result']['drift_share']
        number_of_drifted_columns = result_dict['metrics'][0]['result']['number_of_drifted_columns']
        
        # Get individual column drift results
        drift_by_columns = result_dict['metrics'][0]['result']['drift_by_columns']
        
        # Log the actual results
        log_drift_result(
            drift_detected=dataset_drift,
            drift_score=drift_share,
            affected_features=list(drift_by_columns.keys()) if drift_by_columns else None,
            drift_type="data_drift",
            metadata={
                "total_columns": len(current_df.columns),
                "drifted_columns": number_of_drifted_columns,
                "drift_share": drift_share,
                "sample_size": len(current_df)
            }
        )
        
        if dataset_drift:
            logger.warning(f"DRIFT DETECTED! {number_of_drifted_columns} columns drifted, drift share: {drift_share:.2f}")
        else:
            logger.info(f"No drift detected. Drift share: {drift_share:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Evidently drift detection failed: {e}")
        return False


if __name__ == "__main__":
    success = detect_drift()
    if success:
        print("Drift detection completed successfully")
    else:
        print("Drift detection failed")
        exit(1)