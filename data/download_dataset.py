from pathlib import Path
from shutil import copy2, copyfileobj
from zipfile import ZipFile

import kagglehub

DATASET_NAME = "saharalaa/medical-abstracts-tc-corpus"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "medical_abstracts_tc"
EXPECTED_FILES = {
    "medical_tc_labels.csv",
    "medical_tc_test.csv",
    "medical_tc_train.csv",
}


def clean_raw_data_dir() -> None:
    """
    Remove CSVs antigos antes de copiar uma nova versão do dataset.

    Exemplo:
        >>> clean_raw_data_dir()
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Mantém a pasta previsível quando o DVC executa o estágio novamente.
    for csv_file in RAW_DATA_DIR.glob("*.csv"):
        csv_file.unlink()


def copy_csv_files(downloaded_path: Path) -> int:
    """
    Copia arquivos CSV encontrados no dataset baixado.

    Exemplo:
        >>> copy_csv_files(Path("dataset"))
        3
    """
    copied_files = 0

    # Alguns downloads do Kaggle deixam os CSVs em subpastas.
    for csv_file in downloaded_path.rglob("*.csv"):
        copy2(csv_file, RAW_DATA_DIR / csv_file.name)
        copied_files += 1

    return copied_files


def extract_zip_files(downloaded_path: Path) -> int:
    """
    Extrai arquivos CSV quando o Kaggle entrega o dataset compactado.

    Exemplo:
        >>> extract_zip_files(Path("dataset"))
        3
    """
    extracted_files = 0

    # O Kaggle pode entregar os arquivos do dataset dentro de arquivos .zip.
    for zip_file in downloaded_path.rglob("*.zip"):
        with ZipFile(zip_file) as compressed_file:
            for file_name in compressed_file.namelist():
                if file_name.endswith(".csv"):
                    output_path = RAW_DATA_DIR / Path(file_name).name

                    with compressed_file.open(file_name) as source_file:
                        with output_path.open("wb") as target_file:
                            copyfileobj(source_file, target_file)

                    extracted_files += 1

    return extracted_files


def validate_dataset_files() -> None:
    """
    Valida se os arquivos necessários para o pipeline estão disponíveis.

    Exemplo:
        >>> validate_dataset_files()
    """
    available_files = {file_path.name for file_path in RAW_DATA_DIR.glob("*.csv")}
    missing_files = EXPECTED_FILES - available_files

    if missing_files:
        missing_names = ", ".join(sorted(missing_files))
        raise FileNotFoundError(
            "Medical Abstracts TC Corpus incompleto. "
            f"Arquivos ausentes: {missing_names}."
        )


def download_dataset(force_download: bool = False) -> Path:
    """
    Baixa o Medical Abstracts TC Corpus usando o cache local quando possível.

    Exemplo:
        >>> download_dataset()
        PosixPath('...')
    """
    return Path(
        kagglehub.dataset_download(
            DATASET_NAME,
            force_download=force_download,
        )
    )


def main() -> None:
    """
    Baixa o Medical Abstracts TC Corpus e copia os arquivos para data/raw.

    O corpus possui rótulos de condições médicas, não de urgência. Ele pode ser
    explorado como candidato técnico, mas não deve treinar o modelo final de
    triagem sem uma decisão documentada sobre o target.

    Exemplo:
        >>> main()
    """
    clean_raw_data_dir()
    downloaded_path = download_dataset()

    copied_files = copy_csv_files(downloaded_path)
    extracted_files = extract_zip_files(downloaded_path)

    try:
        validate_dataset_files()
    except FileNotFoundError:
        clean_raw_data_dir()
        downloaded_path = download_dataset(force_download=True)
        copied_files = copy_csv_files(downloaded_path)
        extracted_files = extract_zip_files(downloaded_path)
        validate_dataset_files()

    print(f"Dataset baixado em: {downloaded_path}")
    print(f"Arquivos copiados para: {RAW_DATA_DIR}")
    print(f"Total de arquivos CSV copiados: {copied_files + extracted_files}")


if __name__ == "__main__":
    main()
