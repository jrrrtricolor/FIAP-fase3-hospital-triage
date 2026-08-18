"""Prepara o NHAMCS-ED 2021 para classificação textual de urgência."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from pandas.io.stata import StataReader
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedGroupKFold

from ml_prep_kit import DataValidator, SQLiteDataFrameStore

# Contrato verificado no notebook 00. O hash fixa a versão inteira da fonte.
DATASET_VERSION = "nhamcs-ed-2021-training-v1"
SOURCE_SHA256 = (
    "c1438773d451c78652bf371fc369da5281aa2c82eb5983cccc2646b66d218038"
)
EXPECTED_ELIGIBLE_ROWS = 10_495
EXPECTED_UNIQUE_TEXTS = 10_492
EXPECTED_CLASS_COUNTS = {
    "normal": 3_186,
    "atencao": 5_429,
    "urgente": 1_880,
}

# IMMEDR informa a prioridade de atendimento registrada na triagem.
TARGET_MAP = {
    1: "urgente",
    2: "urgente",
    3: "atencao",
    4: "normal",
    5: "normal",
}
TARGET_ORDER = ("normal", "atencao", "urgente")
RFV_COLUMNS = tuple(f"RFV{index}" for index in range(1, 6))

# Regra: coluna final, sentinelas de ausência, mínimo, máximo e escala.
VITAL_RULES = {
    "TEMPF": ("temperature_f", {-9}, 600, 1112, 10),
    "PULSE": ("heart_rate", {-9, 998}, 0, 300, 1),
    "RESPR": ("respiratory_rate", {-9}, 0, 200, 1),
    "BPSYS": ("systolic_bp", {-9}, 0, 350, 1),
    "BPDIAS": ("diastolic_bp", {-9, 998}, 0, 250, 1),
    "POPCT": ("oxygen_saturation", {-9}, 0, 100, 1),
    "PAINSCALE": ("pain_scale", {-9, -8}, 0, 10, 1),
}
SOURCE_COLUMNS = (
    "AGE",
    "AGEDAYS",
    "SEX",
    "IMMEDR",
    *RFV_COLUMNS,
    *VITAL_RULES,
)
TRAINING_COLUMNS = (
    "record_id",
    "clinical_text",
    "target",
    "text_hash",
    "split",
    "dataset_version",
)

# Formatos mantêm unidades clínicas e uma ordem estável no texto.
OBSERVATION_FORMATS = (
    ("temperature_f", "temperature {:.1f} F"),
    ("heart_rate", "heart rate {:.0f} bpm"),
    ("respiratory_rate", "respiratory rate {:.0f}/min"),
    ("oxygen_saturation", "oxygen saturation {:.0f}%"),
    ("pain_scale", "pain score {:.0f}/10"),
)

# Vinte folds geram 70% treino, 15% validação e 15% teste.
SPLIT_BY_FOLD = ("train",) * 14 + ("validation",) * 3 + ("test",) * 3
WHITESPACE_PATTERN = re.compile(r"\s+")


def prepare_training_data(
    source_path: str | Path,
    database_path: str | Path,
) -> pd.DataFrame:
    """Valida a fonte, prepara os registros e grava o SQLite final."""
    raw_data, rfv_lookup, sex_lookup, source_hash = _load_source(source_path)
    training_data = build_training_data(
        raw_data,
        rfv_lookup=rfv_lookup,
        sex_lookup=sex_lookup,
    )
    _validate_training_data(training_data, check_official_counts=True)

    metadata = pd.Series(
        {
            "dataset_version": DATASET_VERSION,
            "source_sha256": source_hash,
            "row_count": str(len(training_data)),
            "class_counts": _counts_json(
                training_data["target"].value_counts()
            ),
            "split_counts": _counts_json(
                training_data["split"].value_counts()
            ),
            "split_strategy": "StratifiedGroupKFold: 70/15/15 by text_hash",
            "random_state": "42",
            "sklearn_version": sklearn_version,
        },
        name="value",
    ).rename_axis("key").reset_index()

    # Reutiliza o framework existente em vez de criar outra camada de banco.
    output_path = Path(database_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteDataFrameStore(output_path)
    store.save_dataframe(training_data, "training_data")
    store.save_dataframe(metadata, "metadata")
    return training_data


def build_training_data(
    raw_data: pd.DataFrame,
    *,
    rfv_lookup: dict[int, str],
    sex_lookup: dict[int, str],
) -> pd.DataFrame:
    """Transforma somente campos disponíveis na triagem."""
    missing = sorted(set(SOURCE_COLUMNS).difference(raw_data.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}.")

    # Somente IMMEDR 1 a 5 possui target clínico utilizável.
    visits = raw_data.loc[
        raw_data["IMMEDR"].isin(TARGET_MAP),
        list(SOURCE_COLUMNS),
    ].copy()
    if visits.empty:
        raise ValueError("Nenhuma visita possui target elegível.")

    # A posição na fonte gera um ID sem expor paciente ou hospital.
    visits.insert(0, "source_row", visits.index.astype(int))
    visits["target"] = visits["IMMEDR"].map(TARGET_MAP)
    visits["sex_label"] = visits["SEX"].map(sex_lookup)
    if visits["sex_label"].isna().any():
        raise ValueError("Há códigos de sexo sem descrição.")

    complaint_columns = []
    for column in RFV_COLUMNS:
        output_column = f"{column}_description"
        descriptions = visits[column].map(rfv_lookup)
        if (visits[column].gt(0) & descriptions.isna()).any():
            raise ValueError(f"Há códigos {column} sem descrição.")
        visits[output_column] = descriptions
        complaint_columns.append(output_column)

    # Sentinelas significam ausência; nunca são tratadas como sinais vitais.
    vital_columns = []
    for source_column, rule in VITAL_RULES.items():
        output_column, sentinels, lower, upper, scale = rule
        usable = ~visits[source_column].isin(sentinels)
        if (usable & ~visits[source_column].between(lower, upper)).any():
            raise ValueError(
                f"Há valores fora do domínio em {source_column}."
            )
        visits[output_column] = visits[source_column].where(usable) / scale
        vital_columns.append(output_column)

    has_complaint = visits[complaint_columns].notna().any(axis=1)
    has_vital = visits[vital_columns].notna().any(axis=1)
    if (~has_complaint & ~has_vital).any():
        raise ValueError("Há visita sem queixa ou observação de triagem.")

    texts = visits.apply(_build_clinical_text, axis=1).map(_normalize_text)
    prepared = pd.DataFrame(
        {
            "record_id": visits["source_row"].map(
                lambda row: _sha256(f"{DATASET_VERSION}:{row}")
            ),
            "clinical_text": texts,
            "target": visits["target"],
            # O hash será o grupo que impede leakage de textos repetidos.
            "text_hash": texts.map(
                lambda text: _sha256(text.casefold())
            ),
            "dataset_version": DATASET_VERSION,
        }
    )
    prepared = _assign_splits(prepared)
    _validate_training_data(prepared)
    return prepared


def _load_source(
    source_path: str | Path,
) -> tuple[pd.DataFrame, dict[int, str], dict[int, str], str]:
    """Confere o hash e lê as colunas e codebooks necessários."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo NHAMCS ausente: {path}.")

    source_hash = _file_sha256(path)
    if source_hash != SOURCE_SHA256:
        raise ValueError("O arquivo NHAMCS não corresponde à versão aprovada.")

    labels = StataReader(path, convert_categoricals=False).value_labels()
    raw_data = pd.read_stata(
        path,
        convert_categoricals=False,
        columns=list(SOURCE_COLUMNS),
    )
    rfv_lookup = {
        int(code): label
        for code, label in labels["RFVF"].items()
        if int(code) > 0
    }
    sex_lookup = {
        int(code): label for code, label in labels["SEXF"].items()
    }
    return raw_data, rfv_lookup, sex_lookup, source_hash


def _build_clinical_text(row: pd.Series) -> str:
    """Monta idade, sexo, queixas e observações em ordem estável."""
    if row["AGE"] == 0 and 0 <= row["AGEDAYS"] <= 365:
        age = f"Age: {int(row['AGEDAYS'])} days."
    elif row["AGE"] == 93:
        age = "Age: 93 years or older."
    else:
        age = f"Age: {int(row['AGE'])} years."

    parts = [age, f"Sex: {row['sex_label']}."]
    complaints = [
        row[f"{column}_description"]
        for column in RFV_COLUMNS
        if pd.notna(row[f"{column}_description"])
    ]
    complaints = list(dict.fromkeys(complaints))
    if complaints:
        parts.append("Reasons for visit: " + "; ".join(complaints) + ".")

    observations = [
        template.format(row[column])
        for column, template in OBSERVATION_FORMATS[:3]
        if pd.notna(row[column])
    ]
    systolic = row["systolic_bp"]
    diastolic = row["diastolic_bp"]
    if pd.notna(systolic) and pd.notna(diastolic):
        observations.append(
            f"blood pressure {int(systolic)}/{int(diastolic)} mmHg"
        )
    elif pd.notna(systolic):
        observations.append(f"systolic blood pressure {int(systolic)} mmHg")
    elif pd.notna(diastolic):
        observations.append(
            f"diastolic blood pressure {int(diastolic)} mmHg"
        )
    observations.extend(
        template.format(row[column])
        for column, template in OBSERVATION_FORMATS[3:]
        if pd.notna(row[column])
    )
    if observations:
        parts.append(
            "Triage observations: " + "; ".join(observations) + "."
        )
    return " ".join(parts)


def _assign_splits(data: pd.DataFrame) -> pd.DataFrame:
    """Estratifica classes e mantém textos iguais no mesmo split."""
    result = data.sort_values("record_id").reset_index(drop=True).copy()
    result["fold"] = -1
    splitter = StratifiedGroupKFold(
        n_splits=len(SPLIT_BY_FOLD),
        shuffle=True,
        random_state=42,
    )
    for fold, (_, rows) in enumerate(
        splitter.split(
            result,
            y=result["target"],
            groups=result["text_hash"],
        )
    ):
        result.loc[rows, "fold"] = fold

    result["split"] = result.pop("fold").map(
        dict(enumerate(SPLIT_BY_FOLD))
    )
    return result[list(TRAINING_COLUMNS)]


def _validate_training_data(
    data: pd.DataFrame,
    *,
    check_official_counts: bool = False,
) -> None:
    """Interrompe o pipeline quando um gate essencial falha."""
    groups = data.groupby("text_hash")
    checks = {
        "schema inválido": tuple(data.columns) == TRAINING_COLUMNS,
        "dataset vazio": not data.empty,
        "valores nulos": (
            DataValidator().columns_with_missing_values(data).empty
        ),
        "record_id duplicado": data["record_id"].is_unique,
        "classes inválidas": set(data["target"]) == set(TARGET_ORDER),
        "splits inválidos": set(data["split"]) == set(SPLIT_BY_FOLD),
        "texto em mais de um split": not groups["split"].nunique().gt(1).any(),
        "texto com targets distintos": (
            not groups["target"].nunique().gt(1).any()
        ),
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError("; ".join(failed))

    if check_official_counts:
        observed = (
            len(data),
            data["target"].value_counts().to_dict(),
            data["text_hash"].nunique(),
        )
        expected = (
            EXPECTED_ELIGIBLE_ROWS,
            EXPECTED_CLASS_COUNTS,
            EXPECTED_UNIQUE_TEXTS,
        )
        if observed != expected:
            raise ValueError("As contagens da fonte oficial mudaram.")


def _normalize_text(text: str) -> str:
    """Padroniza Unicode e espaços sem remover informação clínica."""
    normalized = unicodedata.normalize("NFKC", text)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def _sha256(value: str) -> str:
    """Retorna o SHA-256 hexadecimal de um texto."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    """Calcula o SHA-256 sem carregar o arquivo inteiro na memória."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts_json(values: pd.Series) -> str:
    """Serializa contagens NumPy como inteiros JSON."""
    counts = {str(key): int(value) for key, value in values.items()}
    return json.dumps(counts, sort_keys=True)


if __name__ == "__main__":
    # Entrada direta para o futuro stage de preparação do DVC.
    root = Path(__file__).resolve().parents[2]
    source = root / "data/raw/nhamcs/2021/ed2021-stata.dta"
    output = root / "data/processed/training_data.db"
    result = prepare_training_data(source, output)
    print(f"{len(result):,} registros gravados em {output}.")
