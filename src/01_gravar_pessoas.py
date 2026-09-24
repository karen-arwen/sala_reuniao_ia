"""01 - Grava o dataset de pessoas (quem esta falando?).

Uso:
    python src/01_gravar_pessoas.py                 # todas as pessoas, ate 30 audios cada
    python src/01_gravar_pessoas.py --pessoa ana    # so uma pessoa
    python src/01_gravar_pessoas.py --meta 20       # muda a meta por pessoa

Cada pessoa le frases variadas (a lista gira ate atingir a meta).
Audios vazios/baixos sao detectados e podem ser regravados na hora.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATASET_PESSOAS, carregar_config, contar_dataset, listar_classes,
                    limpar_nome, proximo_arquivo, salvar_wav, analisar_nivel, eh_silencio)
from reuniao import gravar_trecho


def gravar_um(pasta, prefixo, dur, cfg):
    while True:
        print("Prepare-se... 3", end="", flush=True)
        for n in (2, 1):
            time.sleep(0.6)
            print(f" {n}", end="", flush=True)
        time.sleep(0.6)
        print(f"  >>> FALE AGORA ({dur:.1f}s)")
        y = gravar_trecho(dur)
        nivel = analisar_nivel(y, cfg)
        if eh_silencio(y, cfg):
            r = input(f"[AVISO] Audio muito baixo/vazio (fala em {nivel['fracao_fala']:.0%} do tempo). "
                      "ENTER = regravar | s = salvar mesmo assim: ").strip().lower()
            if r != "s":
                continue
        if nivel["pico"] > 0.99:
            print("[AVISO] Audio estourado (muito alto). Afaste um pouco o microfone.")
        path = proximo_arquivo(pasta, prefixo)
        salvar_wav(path, y)
        print(f"Salvo: {path.relative_to(DATASET_PESSOAS.parent)}")
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pessoa", help="nome da pessoa (cria a pasta se nao existir)")
    ap.add_argument("--meta", type=int, default=30, help="quantidade desejada por pessoa")
    args = ap.parse_args()

    cfg = carregar_config()
    dur = cfg["duracao_gravacao"]
    frases = cfg["frases"]
    DATASET_PESSOAS.mkdir(exist_ok=True)

    pessoas = [limpar_nome(args.pessoa)] if args.pessoa else listar_classes(DATASET_PESSOAS)
    if not pessoas:
        pessoas = [limpar_nome(input("Nome do primeiro participante: "))]

    print("=== GRAVACAO DO DATASET DE PESSOAS ===")
    print("Ambiente silencioso, mesma distancia do microfone, voz natural.")
    print("Somente participantes que concordaram em ser gravados.")

    for pessoa in pessoas:
        pasta = DATASET_PESSOAS / pessoa
        pasta.mkdir(parents=True, exist_ok=True)
        ja = len(list(pasta.glob("*.wav")))
        print(f"\n>>> PARTICIPANTE: {pessoa}  ({ja}/{args.meta} audios)")
        if ja >= args.meta:
            print("Meta ja atingida. Pulando.")
            continue
        for i in range(ja, args.meta):
            frase = frases[i % len(frases)]
            r = input(f'\n[{i+1}/{args.meta}] Frase: "{frase}"\nENTER = gravar | q = proxima pessoa: ')
            if r.strip().lower() == "q":
                break
            gravar_um(pasta, pessoa, dur, cfg)

    print("\n=== CONTAGEM FINAL ===")
    for c, n in contar_dataset(DATASET_PESSOAS).items():
        ok = "OK" if n >= cfg["minimo_por_pessoa"] else f"faltam {cfg['minimo_por_pessoa'] - n}"
        print(f"{c:>12}: {n:3d}  ({ok})")


if __name__ == "__main__":
    main()
