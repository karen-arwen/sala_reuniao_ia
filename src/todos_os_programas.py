"""
Menu central do projeto.

Uso:
    python src/todos_os_programas.py

Cada opcao chama um dos programas obrigatorios (01 a 07).
A opcao 8 abre a interface grafica no navegador.
"""

from pathlib import Path
import subprocess
import sys

SRC = Path(__file__).resolve().parent

SCRIPTS = {
    "1": ("01_gravar_pessoas.py", "Gravar dataset de pessoas"),
    "2": ("02_gravar_emocoes.py", "Gravar dataset de emocoes"),
    "3": ("03_treinar_pessoas.py", "Treinar modelo de pessoas"),
    "4": ("04_treinar_emocoes.py", "Treinar modelo de emocoes"),
    "5": ("05_testar_modelos.py", "Testar modelos (microfone ou WAV)"),
    "6": ("06_analisar_reuniao.py", "Analisar reuniao (ao vivo)"),
    "7": ("07_gerar_relatorio.py", "Gerar relatorio + graficos + PDF"),
}


def main():
    while True:
        print("\n=== SALA DE REUNIAO INTELIGENTE ===")
        for k, (arq, desc) in SCRIPTS.items():
            print(f"{k} - {desc:<38} ({arq})")
        print("8 - Abrir interface interativa (navegador)")
        print("0 - Sair")

        op = input("Escolha: ").strip()
        if op == "0":
            break
        if op == "8":
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(SRC / "app.py")], cwd=str(SRC.parent))
            continue
        if op not in SCRIPTS:
            print("Opcao invalida.")
            continue
        try:
            subprocess.run([sys.executable, str(SRC / SCRIPTS[op][0])])
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
