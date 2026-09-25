"""03 - Treina o Random Forest de pessoas.

Uso:
    python src/03_treinar_pessoas.py

Saidas em modelos/: modelo_pessoas.pkl, metadados_pessoas.json,
matriz_confusao_pessoas.png e importancia_features_pessoas.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from treino import treinar

if __name__ == "__main__":
    try:
        treinar("pessoas")
    except RuntimeError as exc:
        raise SystemExit(f"[ERRO] {exc}")
