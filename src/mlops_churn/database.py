import json
import logging
import os
import sys

import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# set up logging
logger = logging.getLogger(__name__)

def get_db_connection():
   """Creates a connection to PostgreSQL database"""

   try:
       conn = psycopg2.connect(
           host=os.getenv('DATABASE_HOST', 'localhost'),
           port=os.getenv('DATABASE_PORT', '5432'),
           database=os.getenv('DATABASE_NAME', 'monitoring'),
           user=os.getenv('DATABASE_USER', 'postgres'),
           password=os.getenv('DATABASE_PASSWORD', 'example')
       )
       logger.info("Database connection successful")
       return conn
   except psycopg2.OperationalError as e:
       logger.error(f"Database connection failed: {e}")
       return None

def log_prediction(input_data, prediction, model_version):
   """Saves a prediction to the database"""

   conn = get_db_connection()
   if not conn:
       return False

   try:
       cur = conn.cursor()

       # insert the prediction into the predictions table
       cur.execute("""
           INSERT INTO predictions (input_data, prediction, model_version)
           VALUES (%s, %s, %s)
       """, (
           json.dumps(input_data),
           float(prediction),
           model_version
       ))

       conn.commit()
       logger.info(f"Logged prediction: {prediction} from model {model_version}")
       return True

   except Exception as e:
       logger.error(f"Error logging prediction: {e}")
       conn.rollback()
       return False

   finally:
       cur.close()
       conn.close()

def log_drift_result(drift_detected, drift_score, feature_name=None, drift_type="data_drift", report_data=None):
   """Saves drift detection results to the database"""

   conn = get_db_connection()
   if not conn:
       return False

   try:
       cur = conn.cursor()
       cur.execute("""
           INSERT INTO drift_reports (drift_detected, drift_score, feature_name, drift_type, report_data)
           VALUES (%s, %s, %s, %s, %s)
       """, (drift_detected, drift_score, feature_name, drift_type, json.dumps(report_data) if report_data else None))
       conn.commit()
       return True
   except Exception as e:
       logger.error(f"Error logging drift result: {e}")
       conn.rollback()
       return False
   finally:
       cur.close()
       conn.close()
