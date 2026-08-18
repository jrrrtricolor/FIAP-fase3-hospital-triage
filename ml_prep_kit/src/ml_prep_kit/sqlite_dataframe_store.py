"""Utilitários para leitura e gravação em SQLite."""

import sqlite3
from pathlib import Path

import pandas as pd


class SQLiteDataFrameStore:
    """Centraliza a troca de dados entre SQLite e pandas.

    Use esta classe quando a base preparada estiver salva em um banco SQLite
    local. Ela permite carregar tabelas como DataFrames, gravar DataFrames em
    tabelas e gerar um resumo simples do volume ou da distribuição do alvo.

    Exemplo:
        store = SQLiteDataFrameStore("data/training_data.db")

        data = store.load_dataframe(
            table_name="training_data",
            columns=["user_id", "product_id", "target"],
        )

        summary = store.summarize_table(
            table_name="training_data",
            target_column="target",
        )
    """

    def __init__(self, database_path: str | Path) -> None:
        """Guarda o caminho do banco usado pelos métodos da classe."""
        self.database_path = Path(database_path)

    def load_dataframe(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where: str | None = None,
    ) -> pd.DataFrame:
        """Lê uma tabela SQLite como DataFrame.

        Exemplo:
            data = store.load_dataframe("training_data")
        """
        selected_columns = ", ".join(columns) if columns else "*"
        query = f"SELECT {selected_columns} FROM {table_name}"

        if where:
            query = f"{query} WHERE {where}"

        with sqlite3.connect(self.database_path) as conn:
            return pd.read_sql_query(query, conn)

    def save_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
        chunksize: int = 100_000,
    ) -> None:
        """Grava um DataFrame em uma tabela SQLite.

        Exemplo:
            store.save_dataframe(df, "training_data")
        """
        with sqlite3.connect(self.database_path) as conn:
            df.to_sql(
                table_name,
                conn,
                if_exists=if_exists,
                index=False,
                chunksize=chunksize,
            )

    def summarize_table(
        self,
        table_name: str,
        target_column: str | None = None,
    ) -> pd.DataFrame:
        """Retorna linhas e, se informado, a distribuição do alvo.

        Exemplo:
            summary = store.summarize_table("training_data", "target")
        """
        with sqlite3.connect(self.database_path) as conn:
            summary = pd.read_sql_query(
                f"SELECT COUNT(*) AS rows FROM {table_name}",
                conn,
            )

            if target_column is None:
                return summary

            target_distribution = pd.read_sql_query(
                f"""
                SELECT {target_column}, COUNT(*) AS rows
                FROM {table_name}
                GROUP BY {target_column}
                ORDER BY {target_column}
                """,
                conn,
            )

        target_distribution["rate"] = (
            target_distribution["rows"] / target_distribution["rows"].sum()
        )

        return target_distribution
