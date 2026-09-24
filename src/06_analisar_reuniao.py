"""06 - Analisa uma reuniao em blocos de ~3 s.

Uso:
    python src/06_analisar_reuniao.py                       # nova reuniao ao vivo (microfone)
    python src/06_analisar_reuniao.py --reuniao reuniao_01  # continua uma reuniao existente
    python src/06_analisar_reuniao.py --arquivo gravacao.wav  # analisa um audio ja gravado
    python src/06_analisar_reuniao.py --bloco 4             # blocos de 4 s

Ao vivo a gravacao e continua (sem buracos entre blocos). Ctrl+C encerra.
Cada bloco: salva WAV -> classifica pessoa -> classifica emocao -> grava no CSV.
Blocos sem fala sao registrados como "silencio" (nao sao apagados).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REUNIOES, carregar_modelos, carregar_config, modelos_existem
from reuniao import SessaoReuniao, CapturaContinua, nova_reuniao, analisar_arquivo, descrever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuniao", help="nome da pasta em reunioes/ (ex.: reuniao_01)")
    ap.add_argument("--arquivo", help="WAV de uma reuniao gravada para analisar")
    ap.add_argument("--bloco", type=float, help="duracao do bloco em segundos (3, 4 ou 5)")
    args = ap.parse_args()

    if not modelos_existem():
        raise SystemExit("Treine os dois modelos antes de iniciar a reuniao.")
    modelos = carregar_modelos()
    cfg = carregar_config()
    dur = args.bloco or cfg["duracao_bloco"]
    pasta = REUNIOES / args.reuniao if args.reuniao else nova_reuniao()
    pasta.mkdir(parents=True, exist_ok=True)

    print("=== SALA DE REUNIAO INTELIGENTE ===")
    print(f"Pasta: {pasta}")
    print(f"Blocos de {dur:.0f} s. Uma pessoa fala por vez.")

    if args.arquivo:
        rs = analisar_arquivo(args.arquivo, pasta, modelos, cfg, dur, callback=lambda r: print(descrever(r)))
        print(f"\n{len(rs)} blocos analisados.")
    else:
        input("Pressione ENTER para iniciar a reuniao (Ctrl+C encerra)...")
        print("Reuniao iniciada. Gravando...\n")
        with SessaoReuniao(pasta, modelos, cfg, dur) as sessao, CapturaContinua(dur) as cap:
            try:
                while True:
                    print(descrever(sessao.processar_bloco(cap.ler_bloco())))
            except KeyboardInterrupt:
                print("\nReuniao encerrada.")

    print(f"CSV salvo em: {pasta / 'registros.csv'}")
    print(f"Audios em: {pasta / 'blocos_audio'}")
    print(f"Proximo passo: python src/07_gerar_relatorio.py --reuniao {pasta.name}")


if __name__ == "__main__":
    main()
