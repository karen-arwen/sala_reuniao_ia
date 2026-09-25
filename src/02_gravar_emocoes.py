"""02 - Grava o dataset de emocoes (como a fala soou?).

Uso:
    python src/02_gravar_emocoes.py                     # pergunta quem vai gravar
    python src/02_gravar_emocoes.py --pessoa ana --por-emocao 7

Rodada: a MESMA frase e gravada nas quatro emocoes (alegre, neutro, triste, irritado).
Varias pessoas devem gravar para o modelo generalizar. O nome do arquivo guarda
quem gravou (ex.: alegre_dani_001.wav), entao cada pessoa pode gravar em casa.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATASET_EMOCOES, EMOCOES, carregar_config, contar_dataset, limpar_nome,
                    proximo_arquivo, salvar_wav, eh_silencio)
from reuniao import gravar_trecho

DICAS = {
    "alegre": "sorrindo, entonacao mais animada",
    "neutro": "voz normal, sem enfase",
    "triste": "mais devagar e baixo, sem energia",
    "irritado": "firme e impaciente, SEM gritar no microfone",
}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pessoa", help="quem esta gravando")
    ap.add_argument("--por-emocao", type=int, default=7, help="frases por emocao nesta rodada")
    args = ap.parse_args()

    cfg = carregar_config()
    pessoa = limpar_nome(args.pessoa or input("Quem vai gravar agora? ")) or "anonimo"
    frases = cfg["frases"]
    for e in EMOCOES:
        (DATASET_EMOCOES / e).mkdir(parents=True, exist_ok=True)

    print("=== GRAVACAO DO DATASET DE EMOCOES ===")
    print(f"Pessoa: {pessoa} | {args.por_emocao} frases x 4 emocoes")

    inicio = len(list((DATASET_EMOCOES / "neutro").glob(f"neutro_{pessoa}_*.wav")))
    for i in range(args.por_emocao):
        frase = frases[(inicio + i) % len(frases)]
        print(f'\n--- Rodada {i+1}/{args.por_emocao}: "{frase}" ---')
        for emocao in EMOCOES:
            while True:
                r = input(f"  [{emocao.upper()}] ({DICAS[emocao]}) ENTER = gravar | q = sair: ")
                if r.strip().lower() == "q":
                    print("Encerrado.")
                    return mostrar(cfg)
                y = gravar_trecho(cfg["duracao_gravacao"])
                if eh_silencio(y, cfg) and input("  Audio vazio/baixo. ENTER = regravar | s = salvar: ").lower() != "s":
                    continue
                path = proximo_arquivo(DATASET_EMOCOES / emocao, f"{emocao}_{pessoa}")
                salvar_wav(path, y)
                print(f"  Salvo: {path.name}")
                break
    mostrar(cfg)


def mostrar(cfg):
    print("\n=== CONTAGEM ATUAL ===")
    for c, n in contar_dataset(DATASET_EMOCOES).items():
        ok = "OK" if n >= cfg["minimo_por_emocao"] else f"faltam {cfg['minimo_por_emocao'] - n}"
        print(f"{c:>10}: {n:3d}  ({ok})")


if __name__ == "__main__":
    main()
