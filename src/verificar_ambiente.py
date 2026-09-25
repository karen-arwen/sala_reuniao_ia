"""Verifica se o computador esta pronto para rodar o projeto.

Uso:
    python src/verificar_ambiente.py

Confere: versao do Python, bibliotecas, microfones e quantidade de audios.
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OK, FALHA = "[OK]   ", "[FALTA]"
problemas = 0

print("=== VERIFICACAO DO AMBIENTE ===")
v = sys.version_info
print(f"{OK if v >= (3, 10) else FALHA} Python {v.major}.{v.minor}.{v.micro}")
if v < (3, 10):
    problemas += 1

for mod in ["numpy", "pandas", "sklearn", "librosa", "soundfile", "matplotlib", "joblib", "streamlit", "plotly"]:
    try:
        m = importlib.import_module(mod)
        print(f"{OK} {mod} {getattr(m, '__version__', '')}")
    except Exception as exc:
        print(f"{FALHA} {mod}: {exc}")
        problemas += 1

try:
    from reuniao import listar_microfones
    mics = listar_microfones()
    print(f"{OK if mics else FALHA} microfones encontrados: {len(mics)}")
    for i, nome in mics[:8]:
        print(f"         [{i}] {nome}")
    if not mics:
        print("         (sem microfone do PC: use o 'Gravador do navegador' na interface)")
except Exception as exc:
    print(f"[AVISO] nao consegui listar microfones ({exc}). Use o 'Gravador do navegador'.")

try:
    from common import DATASET_PESSOAS, DATASET_EMOCOES, MODELOS, contar_dataset, carregar_config
    cfg = carregar_config()
    cp, ce = contar_dataset(DATASET_PESSOAS), contar_dataset(DATASET_EMOCOES)
    print(f"\nPasta do projeto: {DATASET_PESSOAS.parent}")
    print(f"Audios de pessoas: {sum(cp.values())}  " + "  ".join(f"{k}={n}" for k, n in cp.items()))
    print(f"Audios de emocoes: {sum(ce.values())}  " + "  ".join(f"{k}={n}" for k, n in ce.items()))
    for nome in ("modelo_pessoas.pkl", "modelo_emocoes.pkl"):
        print(f"{OK if (MODELOS / nome).exists() else '[AINDA NAO]'} {nome}")
except Exception as exc:
    print(f"{FALHA} leitura das pastas: {exc}")
    problemas += 1

print("\nTudo pronto!" if problemas == 0 else f"\n{problemas} problema(s) encontrado(s).")
sys.exit(1 if problemas else 0)
