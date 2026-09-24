"""05 - Testa os modelos treinados.

Uso:
    python src/05_testar_modelos.py              # mostra metricas + menu de teste
    python src/05_testar_modelos.py arquivo.wav  # testa um WAV direto

Teste do professor: pessoa cadastrada com frase NOVA, pessoa NAO cadastrada,
e a mesma frase em emocoes diferentes. Tudo pelo microfone, ao vivo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MODELOS, carregar_modelos, carregar_config, carregar_audio, classificar,
                    load_json, modelos_existem)
from reuniao import gravar_trecho


def mostrar_metricas():
    for tipo in ("pessoas", "emocoes"):
        meta_path = MODELOS / f"metadados_{tipo}.json"
        if not meta_path.exists():
            continue
        m = load_json(meta_path)
        print(f"\n=== MODELO DE {tipo.upper()} (treinado em {m['treinado_em']}) ===")
        print(f"Audios: {m['quantidade_total']} | {m['validacao']}")
        print(f"Acuracia (validacao cruzada): {m['acuracia_validacao']:.1%}")
        print("Classe        precisao  recall  f1")
        for c, r in m["relatorio_por_classe"].items():
            print(f"{c:<12} {r['precision']:8.0%} {r['recall']:7.0%} {r['f1-score']:5.0%}")
        print("Matriz de confusao (linhas = real, colunas = previsto):")
        print("            " + " ".join(f"{c[:8]:>8}" for c in m["classes"]))
        for c, linha in zip(m["classes"], m["matriz_confusao"]):
            print(f"{c[:10]:>10}  " + " ".join(f"{v:8d}" for v in linha))
        if m["principais_confusoes"]:
            print("Principais confusoes:")
            for cf in m["principais_confusoes"][:3]:
                print(f"  {cf['real']} -> {cf['previsto']}: {cf['quantidade']}x")


def barra(p, largura=25):
    return "#" * int(round(p * largura)) + "." * (largura - int(round(p * largura)))


def mostrar_resultado(r):
    print("\n------------------------------------------")
    if r["status"] == "silencio":
        print("Nenhuma fala detectada (silencio). Fale mais perto do microfone.")
        return
    print(f"PESSOA: {r['pessoa'].upper()}   (mais provavel: {r['pessoa_mais_provavel']}, "
          f"confianca {r['conf_pessoa']:.0%})")
    if r.get("motivo"):
        print(f"  -> marcado como desconhecido: {r['motivo']}")
    for c, p in sorted(r["probs_pessoa"].items(), key=lambda kv: -kv[1]):
        print(f"   {c:<12} {barra(p)} {p:5.0%}")
    print(f"EMOCAO VOCAL ESTIMADA: {r['emocao']}   (confianca {r['conf_emocao']:.0%})")
    for c, p in sorted(r["probs_emocao"].items(), key=lambda kv: -kv[1]):
        print(f"   {c:<12} {barra(p)} {p:5.0%}")
    print("------------------------------------------")


def main():
    if not modelos_existem():
        raise SystemExit("Treine primeiro os dois modelos (03 e 04).")
    modelos = carregar_modelos()
    cfg = carregar_config()

    if len(sys.argv) > 1:
        mostrar_resultado(classificar(carregar_audio(sys.argv[1]), modelos, cfg))
        return

    mostrar_metricas()
    while True:
        op = input("\nENTER = gravar e testar pelo microfone | caminho de um .wav | q = sair: ").strip()
        if op.lower() == "q":
            break
        if op:
            p = Path(op.strip('"'))
            if not p.exists():
                print("Arquivo nao encontrado.")
                continue
            y = carregar_audio(p)
        else:
            print(f"Fale uma frase NOVA agora ({cfg['duracao_gravacao']:.1f}s)...")
            y = gravar_trecho(cfg["duracao_gravacao"])
        mostrar_resultado(classificar(y, modelos, cfg))


if __name__ == "__main__":
    main()
