import logging
from datetime import timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from pendulum import datetime

logger = logging.getLogger(__name__)


def alert_failure(context):
    """Registra no log os dados necessários para diagnosticar uma falha."""
    ti = context["task_instance"]
    logger.error(
        "Task %s falhou na DAG %s (run %s). Log: %s",
        ti.task_id,
        ti.dag_id,
        ti.run_id,
        ti.log_url,
    )


default_args = {
    "owner": "hospital_triage",
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=1),
    "on_failure_callback": alert_failure,
}

with DAG(
    dag_id="hospital_triage",
    description="Pipeline de ML end-to-end para app de triagem hospitalar",
    default_args=default_args,
    start_date=datetime(2026, 8, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["hospital_triage"],
) as dag:
    ingest_and_validate = BashOperator(
        task_id="ingest_and_validate_source",
        cwd="/app",
        bash_command="poetry run python ./data/download_dataset.py",
    )
    prepare_data = BashOperator(
        task_id="prepare_data",
        cwd="/app",
        bash_command="poetry run python -m hospital_triage.data_preparation",
    )
    validate_data = BashOperator(
        task_id="validate_prepared_data",
        cwd="/app",
        bash_command=(
            "poetry run python -m hospital_triage.data_preparation --validate-only"
        ),
    )
    train_model = BashOperator(
        task_id="train_model",
        cwd="/app",
        bash_command=(
            "poetry run python -m hospital_triage.training "
            '--git-sha "${GIT_SHA:-local}"'
        ),
    )
    evaluate_model = BashOperator(
        task_id="evaluate_model",
        cwd="/app",
        bash_command=(
            "poetry run python -m hospital_triage.training "
            "--validate-evaluation-only"
        ),
    )
    register_model = BashOperator(
        task_id="validate_model_registration",
        cwd="/app",
        bash_command=(
            "poetry run python -m hospital_triage.training "
            "--validate-registration-only"
        ),
    )

    (
        ingest_and_validate
        >> prepare_data
        >> validate_data
        >> train_model
        >> evaluate_model
        >> register_model
    )
