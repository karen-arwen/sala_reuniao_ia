"""04 - Treina o Random Forest de emocoes.

Uso:
    python src/04_treinar_emocoes.py

Saidas em modelos/: modelo_emocoes.pkl, metadados_emocoes.json,
matriz_confusao_emocoes.png e importancia_features_emocoes.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from treino import treinar

if __name__ == "__main__":
    try:
        treinar("emocoes")
    except RuntimeError as exc:
        raise SystemExit(f"[ERRO] {exc}")
