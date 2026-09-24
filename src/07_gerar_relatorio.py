"""07 - Gera o relatorio final a partir do registros.csv.

Uso:
    python src/07_gerar_relatorio.py                       # reuniao mais recente
    python src/07_gerar_relatorio.py --reuniao reuniao_01

Saidas na pasta da reuniao: relatorio.txt, relatorio.pdf e graficos/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REUNIOES
from reuniao import listar_reunioes
from relatorio import gerar_relatorio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuniao", help="nome da pasta em reunioes/")
    args = ap.parse_args()

    reunioes = [r for r in listar_reunioes() if (REUNIOES / r / "registros.csv").exists()]
    if not reunioes:
        raise SystemExit("Nenhuma reuniao com registros.csv. Rode antes o 06_analisar_reuniao.py.")
    nome = args.reuniao or reunioes[-1]
    try:
        gerar_relatorio(REUNIOES / nome)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"[ERRO] {exc}")
    print("\n" + (REUNIOES / nome / "relatorio.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
