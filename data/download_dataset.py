"""Download reproduzível do NHAMCS Emergency Department 2021."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen
from zipfile import ZipFile

DATASET_VERSION = "2021"
DATASET_URL = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/"
    "NHAMCS/stata/ed2021-stata.zip"
)
ARCHIVE_SHA256 = "d1e33d8189077b6ad3271068f9765dd96c3affbfe628f2079611326968623e47"
DATA_FILE_SHA256 = "c1438773d451c78652bf371fc369da5281aa2c82eb5983cccc2646b66d218038"
DATA_FILE_NAME = "ed2021-stata.dta"
ICD10CM_URL = (
    "https://www.cms.gov/files/zip/"
    "2021-code-descriptions-tabular-order-updated-12162020.zip"
)
ICD10CM_ARCHIVE_SHA256 = (
    "479be4d5773fb2e8357c5edb8a56bbd48da4d2bca443a7c42527ad102e3b388c"
)
ICD10CM_FILE_SHA256 = (
    "024c8cdaab815c987f3d43544758047992ef7b5677c92369b1a04ac69b83b48c"
)
ICD10CM_MEMBER_NAME = (
    "2021-code-descriptions-tabular-order/icd10cm_order_2021.txt"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "nhamcs" / DATASET_VERSION
ARCHIVE_PATH = RAW_DATA_DIR / "ed2021-stata.zip"
DATA_FILE_PATH = RAW_DATA_DIR / DATA_FILE_NAME
REFERENCE_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "references" / "icd10cm" / "2021"
ICD10CM_ARCHIVE_PATH = REFERENCE_DATA_DIR / "icd10cm-2021-descriptions.zip"
ICD10CM_FILE_PATH = REFERENCE_DATA_DIR / "icd10cm_order_2021.txt"


def sha256(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo sem carregá-lo inteiro em memória."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checksum(path: Path, expected_checksum: str) -> None:
    """Falha quando o arquivo não corresponde à versão fixada do dataset."""
    actual_checksum = sha256(path)
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Checksum inválido para {path.name}: "
            f"esperado {expected_checksum}, obtido {actual_checksum}."
        )


def download_file(
    url: str,
    destination: Path,
    expected_checksum: str,
    force_download: bool = False,
) -> Path:
    """Baixa um arquivo de referência e valida sua integridade."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force_download:
        validate_checksum(destination, expected_checksum)
        return destination

    temporary_path = destination.with_suffix(f"{destination.suffix}.part")
    request = Request(url, headers={"User-Agent": "FIAP-TC3/1.0"})

    try:
        with urlopen(request, timeout=120) as response:
            with temporary_path.open("wb") as target_file:
                copyfileobj(response, target_file)
        validate_checksum(temporary_path, expected_checksum)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def extract_member(
    archive_path: Path,
    member_name: str,
    destination: Path,
    expected_checksum: str,
    force_extract: bool = False,
) -> Path:
    """Extrai um membro específico de um ZIP e valida seu SHA-256."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force_extract:
        validate_checksum(destination, expected_checksum)
        return destination

    temporary_path = destination.with_suffix(f"{destination.suffix}.part")

    try:
        with ZipFile(archive_path) as archive:
            if member_name not in archive.namelist():
                raise FileNotFoundError(
                    f"{member_name} não foi encontrado em {archive_path.name}."
                )
            with archive.open(member_name) as source_file:
                with temporary_path.open("wb") as target_file:
                    copyfileobj(source_file, target_file)
        validate_checksum(temporary_path, expected_checksum)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def download_dataset(force_download: bool = False) -> Path:
    """Obtém e extrai a versão fixada do NHAMCS-ED 2021."""
    archive_path = download_file(
        DATASET_URL,
        ARCHIVE_PATH,
        ARCHIVE_SHA256,
        force_download=force_download,
    )
    return extract_member(
        archive_path,
        DATA_FILE_NAME,
        DATA_FILE_PATH,
        DATA_FILE_SHA256,
        force_extract=force_download,
    )


def download_icd10cm_reference(force_download: bool = False) -> Path:
    """Obtém as descrições oficiais ICD-10-CM usadas na EDA."""
    archive_path = download_file(
        ICD10CM_URL,
        ICD10CM_ARCHIVE_PATH,
        ICD10CM_ARCHIVE_SHA256,
        force_download=force_download,
    )
    return extract_member(
        archive_path,
        ICD10CM_MEMBER_NAME,
        ICD10CM_FILE_PATH,
        ICD10CM_FILE_SHA256,
        force_extract=force_download,
    )


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Baixa e valida o NHAMCS Emergency Department 2021."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Baixa e extrai novamente mesmo quando os arquivos já existem.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa a aquisição oficial e informa os artefatos locais."""
    args = parse_args()
    data_file_path = download_dataset(force_download=args.force)
    icd10cm_file_path = download_icd10cm_reference(force_download=args.force)

    print(f"Dataset: NHAMCS Emergency Department {DATASET_VERSION}")
    print(f"Arquivo original: {ARCHIVE_PATH}")
    print(f"Arquivo de dados: {data_file_path}")
    print(f"SHA-256 validado: {DATA_FILE_SHA256}")
    print(f"Referência ICD-10-CM: {icd10cm_file_path}")
    print(f"SHA-256 ICD-10-CM validado: {ICD10CM_FILE_SHA256}")


if __name__ == "__main__":
    main()
