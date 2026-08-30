import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

logger = logging.getLogger(__name__)


def alert_failure(context):
    """Callback chamado quando task falha (substitua por Slack/PagerDuty em prod)."""
    ti = context["task_instance"]
    logger.error(
        "🚨 ALERTA: Task %s falhou na DAG %s (run %s). Log: %s",
        ti.task_id, ti.dag_id, ti.run_id, ti.log_url,
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
    start_date=datetime(2026, 8, 1),
    schedule=None,
    catchup=False,
    tags=["hospital_triage"],
) as dag:

    t_download = BashOperator(task_id="download_data", cwd="/app", bash_command="python ./data/download_dataset.py")
    t_prepare = BashOperator(task_id="prepare_data", cwd="/app", bash_command="poetry run python -m hospital_triage.data_preparation")
    t_train = BashOperator(task_id="train_model", cwd="/app", bash_command="poetry run python -m hospital_triage.training")

    t_download >> t_prepare >> t_train
