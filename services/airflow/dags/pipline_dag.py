from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests

PROJECT_DIR = "/opt/airflow/project"

default_args = {"retries": 1}

with DAG(
    "pmldl_pipeline",
    schedule_interval="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
) as dag:

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"cd {PROJECT_DIR} && python code/datasets/prepare_data.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_DIR} && python code/models/train_model.py",
    )

    def reload_model():
        r = requests.post("http://api:8000/reload", timeout=30)
        r.raise_for_status()

    reload_api = PythonOperator(
        task_id="reload_api",
        python_callable=reload_model,
    )

    prepare_data >> train_model >> reload_api