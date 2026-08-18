"""Utilitários para leitura de dados."""

from pathlib import Path

import pandas as pd


class CSVDataLoader:
    """Carrega arquivos CSV a partir de um diretório base.

    Esta classe centraliza a leitura de CSVs. Assim, projetos podem reutilizar
    a mesma lógica sem repetir chamadas a `pd.read_csv` em notebooks e scripts.

    Exemplo:
        loader = CSVDataLoader("data/raw")
        tables = loader.load({"orders": "orders.csv"})
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Guarda o diretório onde os arquivos CSV estão armazenados."""
        self.base_dir = Path(base_dir)

    def load(
        self,
        table_files: dict[str, str],
        read_csv_kwargs: dict[str, object] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Carrega um ou mais arquivos CSV e retorna cada tabela pelo nome.

        Exemplo:
            tables = loader.load({
                "orders": "orders.csv",
                "products": "products.csv",
            })
        """
        read_csv_kwargs = read_csv_kwargs or {}

        return {
            table_name: pd.read_csv(
                self.base_dir / file_name,
                **read_csv_kwargs,
            )
            for table_name, file_name in table_files.items()
        }
