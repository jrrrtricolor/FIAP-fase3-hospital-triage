"""Teste do contrato de preparação do NHAMCS."""

import hashlib

import pandas as pd
import pytest

from ml_prep_kit import SQLiteDataFrameStore
from src.hospital_triage.data_preparation import (
    DATASET_VERSION,
    TRAINING_COLUMNS,
    build_training_data,
    validate_prepared_database,
)


def test_builds_training_data_without_leakage() -> None:
    """Valida texto, target, campos permitidos e isolamento dos splits."""
    rows = []
    rfv_lookup = {}
    triage_levels = ((4, 5), (3,), (1, 2))

    # São 20 exemplos por classe porque o pipeline utiliza 20 folds.
    for class_index, levels in enumerate(triage_levels):
        for row_index in range(20):
            is_first_row = class_index == 0 and row_index == 0
            rfv_code = 20_000 + class_index * 100 + row_index
            rfv_lookup[rfv_code] = f"Complaint {rfv_code}"
            rows.append(
                {
                    "AGE": 93 if is_first_row else 20 + row_index,
                    "AGEDAYS": -7,
                    "SEX": 1 + row_index % 2,
                    "IMMEDR": levels[row_index % len(levels)],
                    "RFV1": rfv_code,
                    "RFV2": -9,
                    "RFV3": -9,
                    "RFV4": -9,
                    "RFV5": -9,
                    "TEMPF": 980 + row_index % 10,
                    "PULSE": 60 + row_index,
                    "RESPR": 15 + row_index % 5,
                    "BPSYS": 120,
                    # 998: diastólica ausente; sistólica ainda deve ser usada.
                    "BPDIAS": 998 if is_first_row else 80,
                    "POPCT": 95 + row_index % 5,
                    "PAINSCALE": row_index % 11,
                    # Campo pós-triagem proibido como feature.
                    "DIAG1": "blocked",
                }
            )

    # Duplica um texto para testar o isolamento entre splits.
    rows[1] = rows[0].copy()

    # IMMEDR inválido deve ser removido antes da criação do target.
    excluded = rows[0].copy()
    excluded["IMMEDR"] = -9
    rows.append(excluded)

    result = build_training_data(
        pd.DataFrame(rows),
        rfv_lookup=rfv_lookup,
        sex_lookup={1: "Female", 2: "Male"},
    )
    all_text = " ".join(result["clinical_text"])

    # Contrato básico: schema enxuto, três classes e IDs únicos.
    assert tuple(result.columns) == TRAINING_COLUMNS
    assert result["target"].value_counts().to_dict() == {
        "normal": 20,
        "atencao": 20,
        "urgente": 20,
    }
    assert result["record_id"].is_unique

    # O texto repetido existe e permanece inteiro em um único split.
    assert result.groupby("text_hash").size().max() == 2
    assert result.groupby("text_hash")["split"].nunique().max() == 1

    # Regras importantes do texto e de prevenção de leakage.
    assert "Age: 93 years or older." in all_text
    assert "systolic blood pressure 120 mmHg" in all_text
    assert "DIAG1" not in result.columns


def test_validates_database_and_rejects_invalid_hash(tmp_path) -> None:
    """Interrompe o pipeline quando o artefato preparado é inconsistente."""
    texts = ["stable patient", "moderate symptom", "critical emergency"]
    data = pd.DataFrame(
        {
            "record_id": ["record-1", "record-2", "record-3"],
            "clinical_text": texts,
            "target": ["normal", "atencao", "urgente"],
            "text_hash": [
                hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
                for text in texts
            ],
            "split": ["train", "validation", "test"],
            "dataset_version": DATASET_VERSION,
        }
    )

    database_path = tmp_path / "training_data.db"
    store = SQLiteDataFrameStore(database_path)
    store.save_dataframe(data, "training_data")

    validated = validate_prepared_database(
        database_path,
        check_official_counts=False,
    )
    assert len(validated) == 3

    # Um hash adulterado deve falhar antes de qualquer treinamento.
    data.loc[0, "text_hash"] = "invalid"
    store.save_dataframe(data, "training_data")
    with pytest.raises(ValueError, match="hash de texto inválido"):
        validate_prepared_database(
            database_path,
            check_official_counts=False,
        )
