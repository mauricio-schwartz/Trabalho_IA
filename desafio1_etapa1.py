"""
Desafio IA – Etapa 1
====================
Treina e avalia dois modelos de classificação para o sistema multi-agente de
resposta às enchentes do Rio Grande do Sul:

  1. Agente de Monitoramento  →  dataset_m.csv  →  alvo: risco  (0–5)
  2. Agente de Triagem        →  dataset_t.csv  →  alvo: prioridade (0–8)

Uso
---
    python desafio1_etapa1.py [--monitoring PATH] [--triage PATH] [--output DIR]

Exemplos
--------
    # Caminhos padrão (dataset_m.csv e dataset_t.csv na pasta atual)
    python desafio1_etapa1.py

    # Caminhos customizados + salvar resultados
    python desafio1_etapa1.py --monitoring data/dataset_m.csv \\
                               --triage    data/dataset_t.csv \\
                               --output    results/
"""

import argparse
import os
import sys

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas que estejam completamente vazias (artefato de cabeçalho)."""
    all_null = [c for c in df.columns if df[c].isna().all()]
    if all_null:
        print(f"  [aviso] Colunas descartadas por estarem 100% vazias: {all_null}")
        df = df.drop(columns=all_null)
    return df


def parse_datetime_features(df: pd.DataFrame, col: str = "data") -> pd.DataFrame:
    """Converte coluna datetime em features numéricas (hora, dia da semana) e remove o original."""
    if col not in df.columns:
        return df
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df["hora"] = df[col].dt.hour
    df["dia_semana"] = df[col].dt.dayofweek
    df = df.drop(columns=[col])
    return df


def _is_string_col(series: pd.Series) -> bool:
    """Retorna True se a coluna for do tipo string/texto (compatível com pandas 1.x e 2.x)."""
    return pd.api.types.is_string_dtype(series) and not pd.api.types.is_bool_dtype(series)


def cast_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas booleanas ou com valores True/False para int (0/1)."""
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
        elif _is_string_col(df[col]):
            unique_lower = set(str(v).strip().lower() for v in df[col].dropna().unique())
            if unique_lower <= {"true", "false"}:
                df[col] = df[col].map(lambda v: 1 if str(v).strip().lower() == "true" else 0)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Constrói o ColumnTransformer adequado ao DataFrame de treino."""
    categorical_cols = [c for c in X.columns if _is_string_col(X[c])]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [
        ("num", numeric_transformer, numeric_cols),
    ]
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_and_print(name: str, y_true, y_pred) -> dict:
    """Calcula e imprime as métricas; retorna dicionário com os valores."""
    labels = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, zero_division=0)

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"RESULTADOS – {name}")
    print(f"{sep}")
    print(f"Acurácia      : {acc:.4f}")
    print(f"F1 (weighted) : {f1w:.4f}")
    print(f"\nMatriz de confusão (linhas=real, colunas=predito), labels={labels}:")
    print(cm)
    print(f"\nClassification report:")
    print(report)
    print(sep)

    return {"accuracy": acc, "f1_weighted": f1w, "confusion_matrix": cm, "labels": labels}


def save_results(output_dir: str, name: str, metrics: dict):
    """Salva métricas em arquivos texto na pasta output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = name.lower().replace(" ", "_").replace("–", "").replace("(", "").replace(")", "").strip("_")
    path = os.path.join(output_dir, f"{safe_name}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"Resultados – {name}\n")
        fh.write(f"Acurácia      : {metrics['accuracy']:.4f}\n")
        fh.write(f"F1 (weighted) : {metrics['f1_weighted']:.4f}\n")
        fh.write(f"\nMatriz de confusão (labels={metrics['labels']}):\n")
        fh.write(str(metrics["confusion_matrix"]))
        fh.write("\n")
    print(f"  [info] Resultado salvo em: {path}")


# ---------------------------------------------------------------------------
# Model training functions
# ---------------------------------------------------------------------------

def train_monitoring_model(path_csv: str = "dataset_m.csv") -> tuple:
    """
    Treina modelo de classificação para o Agente de Monitoramento.
    Target: risco (ordinal 0–5).
    """
    print(f"\n[Monitoramento] Carregando {path_csv} ...")
    df = pd.read_csv(path_csv)
    print(f"  Shape original: {df.shape}")

    df = drop_all_null_columns(df)
    df = parse_datetime_features(df)
    df = cast_booleans(df)

    target = "risco"
    if target not in df.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada em {path_csv}. "
                         f"Colunas disponíveis: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    print(f"  Features: {list(X.columns)}")
    print(f"  Distribuição do alvo:\n{y.value_counts().sort_index().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )

    preprocessor = build_preprocessor(X_train)

    clf = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    print("  Treinando RandomForestClassifier (n_estimators=400)...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = evaluate_and_print("Agente de Monitoramento (alvo: risco)", y_test, y_pred)
    return clf, metrics


def train_triage_model(path_csv: str = "dataset_t.csv") -> tuple:
    """
    Treina modelo de classificação para o Agente de Triagem.
    Target: prioridade (ordinal 0–8).
    """
    print(f"\n[Triagem] Carregando {path_csv} ...")
    df = pd.read_csv(path_csv)
    print(f"  Shape original: {df.shape}")

    df = drop_all_null_columns(df)
    df = parse_datetime_features(df)
    df = cast_booleans(df)

    target = "prioridade"
    if target not in df.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada em {path_csv}. "
                         f"Colunas disponíveis: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    print(f"  Features: {list(X.columns)}")
    print(f"  Distribuição do alvo:\n{y.value_counts().sort_index().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )

    preprocessor = build_preprocessor(X_train)

    clf = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    print("  Treinando RandomForestClassifier (n_estimators=500)...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = evaluate_and_print("Agente de Triagem (alvo: prioridade)", y_test, y_pred)
    return clf, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina e avalia os modelos ML para os agentes de monitoramento e triagem."
    )
    parser.add_argument(
        "--monitoring",
        default="dataset_m.csv",
        metavar="PATH",
        help="Caminho para dataset_m.csv (padrão: dataset_m.csv)",
    )
    parser.add_argument(
        "--triage",
        default="dataset_t.csv",
        metavar="PATH",
        help="Caminho para dataset_t.csv (padrão: dataset_t.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Pasta onde salvar os resultados em texto (opcional, ex: results/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    errors = []
    for label, path in [("Monitoramento", args.monitoring), ("Triagem", args.triage)]:
        if not os.path.isfile(path):
            errors.append(f"  Dataset do Agente de {label} não encontrado: {path!r}")
    if errors:
        print("\n[ERRO] Datasets ausentes:")
        for e in errors:
            print(e)
        print("\nColoque os arquivos CSV nos caminhos indicados e tente novamente.")
        print("Consulte o README.md para instruções detalhadas.")
        sys.exit(1)

    _, metrics_m = train_monitoring_model(args.monitoring)
    _, metrics_t = train_triage_model(args.triage)

    if args.output:
        save_results(args.output, "Agente de Monitoramento (alvo risco)", metrics_m)
        save_results(args.output, "Agente de Triagem (alvo prioridade)", metrics_t)

    print("\n[OK] Treinamento e avaliação concluídos.")


if __name__ == "__main__":
    main()
