"""Treinamento dos dois Random Forests (usado pelos scripts 03/04 e pela interface).

Validacao honesta: validacao cruzada estratificada (k-fold).
Cada audio do dataset e previsto por um modelo que NAO viu esse audio no treino.
Assim a acuracia e a matriz de confusao usam TODOS os audios, sem trapaca.
Depois disso o modelo final e treinado com o dataset completo e salvo.
"""

from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import (
    DATASET_PESSOAS, DATASET_EMOCOES, MODELOS,
    FEATURE_NAMES, familia_feature, find_wavs, extract_features,
    save_json, carregar_config,
)

TIPOS = {
    "pessoas": {
        "dataset": DATASET_PESSOAS,
        "modelo": MODELOS / "modelo_pessoas.pkl",
        "meta": MODELOS / "metadados_pessoas.json",
        "cm": MODELOS / "matriz_confusao_pessoas.png",
        "imp": MODELOS / "importancia_features_pessoas.png",
        "titulo": "Pessoas",
        "minimo": "minimo_por_pessoa",
    },
    "emocoes": {
        "dataset": DATASET_EMOCOES,
        "modelo": MODELOS / "modelo_emocoes.pkl",
        "meta": MODELOS / "metadados_emocoes.json",
        "cm": MODELOS / "matriz_confusao_emocoes.png",
        "imp": MODELOS / "importancia_features_emocoes.png",
        "titulo": "Emocoes",
        "minimo": "minimo_por_emocao",
    },
}


def novo_modelo():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=1,
        n_jobs=-1,
    )


def verificar_balanceamento(counts, minimo):
    avisos = []
    if not counts:
        return ["Nenhuma classe encontrada."]
    for c, n in counts.items():
        if n < minimo:
            avisos.append(f"'{c}' tem {n} audios (minimo recomendado: {minimo}).")
    mx, mn = max(counts.values()), min(counts.values())
    if mn > 0 and mx / mn > 1.5:
        avisos.append(f"Dataset desbalanceado: maior classe {mx} x menor classe {mn} (razao {mx/mn:.1f}).")
    return avisos


def plot_matriz(cm, labels, titulo, path):
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3, 1.1 * len(labels) + 2.5))
    im = ax.imshow(cm, cmap="RdPu")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    lim = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > lim else "#1f2937", fontsize=11, fontweight="bold")
    ax.set_title(f"Matriz de Confusao - {titulo}\n(validacao cruzada)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def importancia_por_familia(model):
    fam = {}
    for nome, imp in zip(FEATURE_NAMES, model.feature_importances_):
        f = familia_feature(nome)
        fam[f] = fam.get(f, 0.0) + float(imp)
    return dict(sorted(fam.items(), key=lambda kv: -kv[1]))


def plot_importancia(fam, titulo, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    nomes, vals = list(fam)[::-1], [fam[k] * 100 for k in list(fam)[::-1]]
    ax.barh(nomes, vals, color="#C8457E")
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    ax.set_xlabel("Importancia no Random Forest (%)")
    ax.set_title(f"O que o modelo de {titulo.lower()} mais usa")
    ax.set_xlim(0, max(vals) * 1.2 if vals else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def treinar(tipo, progresso=None, log=print):
    """Treina 'pessoas' ou 'emocoes'. progresso(fracao, texto) e opcional."""
    t = TIPOS[tipo]
    cfg = carregar_config()
    MODELOS.mkdir(exist_ok=True)

    items = find_wavs(t["dataset"])
    if not items:
        raise RuntimeError(f"Nenhum WAV encontrado em {t['dataset'].name}/")

    counts = {}
    for _, label in items:
        counts[label] = counts.get(label, 0) + 1
    counts = {k: v for k, v in counts.items() if v > 0}

    # Classes com pouquissimos audios (ex.: participante recem-cadastrado) ficam fora
    # do treino para nao quebrar a validacao cruzada. O aviso aparece no log e no JSON.
    MIN_TREINO = 5
    fora = {k: v for k, v in counts.items() if v < MIN_TREINO}
    if fora:
        items = [(p, c) for p, c in items if c not in fora]
        counts = {k: v for k, v in counts.items() if k not in fora}

    log("=== CONTAGEM DO DATASET ===")
    for label in sorted(counts):
        log(f"{label:>12}: {counts[label]}")
    avisos = verificar_balanceamento(counts, cfg[t["minimo"]])
    for k, v in fora.items():
        avisos.insert(0, f"'{k}' ficou FORA do treino: tem so {v} audio(s) (precisa de pelo menos {MIN_TREINO}).")
    for a in avisos:
        log(f"[AVISO] {a}")

    X, y, ignorados = [], [], []
    for i, (path, label) in enumerate(items):
        if progresso:
            progresso(i / len(items) * 0.8, f"Extraindo features: {path.name}")
        try:
            X.append(extract_features(path))
            y.append(label)
        except Exception as exc:
            ignorados.append(f"{path.name}: {exc}")
            log(f"[AVISO] Ignorando {path.name}: {exc}")

    X, y = np.asarray(X), np.asarray(y)
    classes, qtds = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise RuntimeError("Sao necessarias pelo menos duas classes com audios.")
    k = int(min(5, qtds.min()))
    if k < 2:
        raise RuntimeError("Cada classe precisa de pelo menos 2 audios para validar.")

    if progresso:
        progresso(0.82, f"Validacao cruzada ({k} folds)...")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    pred = cross_val_predict(novo_modelo(), X, y, cv=cv)
    acc = accuracy_score(y, pred)
    labels = list(classes)
    cm = confusion_matrix(y, pred, labels=labels)
    rep = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)

    log("\n=== RESULTADO (validacao cruzada) ===")
    log(f"Acuracia: {acc:.4f}")
    log(classification_report(y, pred, labels=labels, zero_division=0))

    # Principais confusoes (para explicar na apresentacao)
    confusoes = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusoes.append({"real": a, "previsto": b, "quantidade": int(cm[i, j])})
    confusoes.sort(key=lambda c: -c["quantidade"])

    if progresso:
        progresso(0.92, "Treinando modelo final com todos os audios...")
    model = novo_modelo().fit(X, y)
    joblib.dump(model, t["modelo"])

    plot_matriz(cm, labels, t["titulo"], t["cm"])
    fam = importancia_por_familia(model)
    plot_importancia(fam, t["titulo"], t["imp"])

    meta = {
        "modelo": "RandomForestClassifier",
        "parametros": {"n_estimators": 300, "class_weight": "balanced", "random_state": 42},
        "treinado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classes": labels,
        "quantidade_total": int(len(y)),
        "contagem_por_classe": {k2: int(v) for k2, v in counts.items()},
        "audios_ignorados": ignorados,
        "avisos_dataset": avisos,
        "validacao": f"StratifiedKFold com {k} folds (cada audio avaliado por modelo que nao o viu)",
        "acuracia_validacao": float(acc),
        "relatorio_por_classe": {
            c: {m: float(rep[c][m]) for m in ("precision", "recall", "f1-score", "support")}
            for c in labels
        },
        "matriz_confusao": cm.tolist(),
        "principais_confusoes": confusoes[:5],
        "importancia_por_familia": fam,
        "features": FEATURE_NAMES,
        "observacao": "Metricas calculadas a partir dos audios reais do dataset.",
    }
    save_json(t["meta"], meta)

    log(f"\nModelo salvo em: {t['modelo']}")
    log(f"Matriz de confusao salva em: {t['cm']}")
    if progresso:
        progresso(1.0, "Pronto!")
    return meta
