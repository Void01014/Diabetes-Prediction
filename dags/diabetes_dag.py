import sys
sys.path.insert(0, "/opt/airflow")  # makes `from src.train import ...` work

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from src.train import extract, preprocess, retrain

with DAG(
    dag_id="diabetes_dag",
    schedule=None,       # manual trigger only
    catchup=False,
    start_date=datetime(2024, 1, 1),
    tags=["diabetes"],
) as dag:

    t_extract = PythonOperator(
        task_id="extract",
        python_callable=extract,
    )

    t_preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess,
    )

    t_retrain = PythonOperator(
        task_id="retrain",
        python_callable=retrain,
    )

    t_extract >> t_preprocess >> t_retrain
