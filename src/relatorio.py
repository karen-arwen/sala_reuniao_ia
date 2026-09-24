"""Relatorio final calculado 100% a partir do registros.csv (usado pelo 07 e pela interface)."""

from pathlib import Path
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from common import (
    MODELOS, EMOCOES, SILENCIO, DESCONHECIDO, INCERTO, parse_time, carregar_config,
)

CORES_EMOCAO = {"alegre": "#F2B544", "neutro": "#A9A3B8", "triste": "#6C9BD2", "irritado": "#E0607E"}
PALETA = ["#C8457E", "#8B6FD6", "#E38FB4", "#5B8DEF", "#E9A15B", "#4FB39B", "#B06AB3", "#7FB5E8"]
COR_ESPECIAL = {DESCONHECIDO: "#BFB3BA", INCERTO: "#D8CDD3", SILENCIO: "#EDE6EA"}

AVISO_ETICO = ("As porcentagens representam classificacoes do modelo a partir de caracteristicas "
               "acusticas (energia, ritmo, frequencia, timbre). Elas NAO provam o sentimento real "
               "de ninguem e nao devem ser usadas para expor ou constranger participantes.")


def cor_pessoa(nome, todas):
    if nome in COR_ESPECIAL:
        return COR_ESPECIAL[nome]
    normais = [p for p in todas if p not in COR_ESPECIAL]
    return PALETA[normais.index(nome) % len(PALETA)] if nome in normais else "#C8457E"


def ler_registros(pasta):
    csv_path = Path(pasta) / "registros.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"inicio": str, "fim": str})
    if df.empty:
        return df
    df["inicio_s"] = df["inicio"].map(parse_time)
    df["fim_s"] = df["fim"].map(parse_time)
    df["duracao_s"] = (df["fim_s"] - df["inicio_s"]).clip(lower=0)
    df["conf_pessoa"] = pd.to_numeric(df["conf_pessoa"], errors="coerce").fillna(0)
    df["conf_emocao"] = pd.to_numeric(df["conf_emocao"], errors="coerce").fillna(0)
    return df


def calcular(df, cfg=None):
    """Todos os numeros do relatorio. Nada e inventado: tudo sai do CSV."""
    cfg = cfg or carregar_config()
    fala = df[df["pessoa"] != SILENCIO]
    pessoas = sorted(p for p in fala["pessoa"].unique() if p not in (DESCONHECIDO, INCERTO))
    pessoas += [p for p in (DESCONHECIDO, INCERTO) if p in set(fala["pessoa"])]

    por_pessoa = {}
    for p in pessoas:
        d = fala[fala["pessoa"] == p]
        validas = d[d["emocao"].isin(EMOCOES)]
        cont = {e: int((validas["emocao"] == e).sum()) for e in EMOCOES}
        tot = sum(cont.values())
        por_pessoa[p] = {
            "tempo_s": float(d["duracao_s"].sum()),
            "trechos": int(len(d)),
            "conf_media": float(d["conf_pessoa"].mean()) if len(d) else 0.0,
            "emocoes": cont,
            "percentuais": {e: (cont[e] / tot * 100 if tot else 0.0) for e in EMOCOES},
        }

    baixa = fala[(fala["conf_pessoa"] < cfg["limiar_pessoa"]) |
                 (fala["conf_emocao"] < cfg["limiar_emocao"]) |
                 (fala["pessoa"].isin([DESCONHECIDO, INCERTO]))]

    geral = {e: int((fala["emocao"] == e).sum()) for e in EMOCOES}
    tempo_fala = float(fala["duracao_s"].sum())
    return {
        "total_blocos": int(len(df)),
        "blocos_fala": int(len(fala)),
        "blocos_silencio": int((df["pessoa"] == SILENCIO).sum()),
        "duracao_total_s": float(df["fim_s"].max() - df["inicio_s"].min()) if len(df) else 0.0,
        "tempo_fala_s": tempo_fala,
        "pessoas": pessoas,
        "por_pessoa": por_pessoa,
        "baixa_confianca": baixa,
        "emocoes_geral": geral,
        "mais_falou": max(pessoas, key=lambda p: por_pessoa[p]["tempo_s"]) if pessoas else None,
    }


def mmss(seg):
    seg = int(round(seg))
    return f"{seg // 60} min {seg % 60:02d} s"


# ---------------------------------------------------------------------------
# Graficos (matplotlib)
# ---------------------------------------------------------------------------
def _estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def graf_tempo(res, path):
    ps = res["pessoas"]
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [res["por_pessoa"][p]["tempo_s"] for p in ps]
    bars = ax.bar(ps, vals, color=[cor_pessoa(p, ps) for p in ps])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, mmss(v), ha="center", va="bottom", fontsize=9)
    ax.set_title("Tempo estimado de fala por participante", fontweight="bold")
    ax.set_ylabel("Segundos")
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def graf_emocoes_pessoa(res, path):
    ps = res["pessoas"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ps))
    base = np.zeros(len(ps))
    for e in EMOCOES:
        v = np.array([res["por_pessoa"][p]["percentuais"][e] for p in ps])
        ax.bar(x, v, bottom=base, label=e, color=CORES_EMOCAO[e])
        for xi, (b, vi) in enumerate(zip(base, v)):
            if vi >= 8:
                ax.text(xi, b + vi / 2, f"{vi:.0f}%", ha="center", va="center", fontsize=8)
        base += v
    ax.set_xticks(x, ps)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% dos trechos da pessoa")
    ax.set_title("Emocoes vocais classificadas por participante", fontweight="bold")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def graf_geral(res, path):
    g = {e: v for e, v in res["emocoes_geral"].items() if v > 0}
    fig, ax = plt.subplots(figsize=(7, 5))
    if g:
        ax.pie(list(g.values()), labels=list(g), autopct="%1.0f%%", startangle=90,
               colors=[CORES_EMOCAO[e] for e in g], wedgeprops={"width": 0.45, "edgecolor": "white"})
    else:
        ax.text(0.5, 0.5, "Sem trechos classificados", ha="center")
        ax.axis("off")
    ax.set_title("Distribuicao geral das emocoes vocais classificadas", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def graf_linha_tempo(df, res, path):
    ps = res["pessoas"]
    fig, ax = plt.subplots(figsize=(11, 0.55 * max(2, len(ps)) + 1.8))
    for i, p in enumerate(ps):
        d = df[df["pessoa"] == p]
        ax.broken_barh(list(zip(d["inicio_s"], d["duracao_s"])), (i - 0.35, 0.7),
                       facecolors=[CORES_EMOCAO.get(e, "#9CA3AF") for e in d["emocao"]],
                       edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(ps)), ps)
    ax.invert_yaxis()
    ax.set_xlabel("Tempo da reuniao (s)")
    ax.set_title("Linha do tempo: quem falou e emocao vocal estimada (cor)", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CORES_EMOCAO[e]) for e in EMOCOES]
    ax.legend(handles, EMOCOES, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def graf_baixa(res, path):
    baixa = res["baixa_confianca"]
    cols = ["inicio", "fim", "pessoa", "conf_pessoa", "emocao", "conf_emocao", "arquivo"]
    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.35 * min(len(baixa), 40) + 1.5)))
    ax.axis("off")
    if len(baixa):
        data = [[r[c] if not c.startswith("conf") else f"{r[c]:.0%}" for c in cols]
                for _, r in baixa.head(40).iterrows()]
        t = ax.table(cellText=data, colLabels=cols, loc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1, 1.3)
        extra = f" (mostrando 40 de {len(baixa)})" if len(baixa) > 40 else ""
        ax.set_title(f"Trechos com baixa confianca{extra}", fontweight="bold", pad=12)
    else:
        ax.text(0.5, 0.5, "Nenhum trecho abaixo do limiar de confianca.", ha="center", va="center")
        ax.set_title("Trechos com baixa confianca", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------
def texto_relatorio(nome, res, cfg):
    L = [
        "RELATORIO - SALA DE REUNIAO INTELIGENTE",
        f"Reuniao: {nome}",
        f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "",
        "RESUMO GERAL",
        f"Total de blocos analisados: {res['total_blocos']}",
        f"Blocos com fala: {res['blocos_fala']} | blocos de silencio: {res['blocos_silencio']}",
        f"Duracao registrada: {mmss(res['duracao_total_s'])}",
        f"Tempo total de fala: {mmss(res['tempo_fala_s'])}",
        f"Participantes identificados: {', '.join(p for p in res['pessoas'] if p not in (DESCONHECIDO, INCERTO)) or '-'}",
    ]
    if res["mais_falou"]:
        L.append(f"Maior tempo de fala: {res['mais_falou']} "
                 f"({mmss(res['por_pessoa'][res['mais_falou']]['tempo_s'])})")
    tot_g = sum(res["emocoes_geral"].values())
    if tot_g:
        L.append("Distribuicao geral das classificacoes vocais: " + ", ".join(
            f"{e} {v / tot_g * 100:.0f}%" for e, v in res["emocoes_geral"].items()))
    L += ["", "RESUMO POR PARTICIPANTE"]

    for p in res["pessoas"]:
        d = res["por_pessoa"][p]
        L += ["", p.upper(),
              f"Tempo de fala estimado: {mmss(d['tempo_s'])}",
              f"Trechos analisados: {d['trechos']}",
              f"Confianca media na identificacao: {d['conf_media']:.0%}",
              "Classificacoes vocais:"]
        for e in EMOCOES:
            L.append(f"- {e:<9} {d['percentuais'][e]:5.1f}%  ({d['emocoes'][e]} trechos)")
        dom = max(EMOCOES, key=lambda e: d["emocoes"][e])
        if d["emocoes"][dom]:
            L += ["Observacao:",
                  f"{d['percentuais'][dom]:.0f}% dos trechos de fala foram classificados como {dom}.",
                  "Isso nao prova sentimento real; e uma estimativa do modelo."]

    b = res["baixa_confianca"]
    L += ["", "TRECHOS COM BAIXA CONFIANCA",
          f"Criterio: conf_pessoa < {cfg['limiar_pessoa']:.0%} ou conf_emocao < {cfg['limiar_emocao']:.0%} "
          "ou pessoa desconhecida.",
          f"Quantidade: {len(b)}"]
    for _, r in b.iterrows():
        L.append(f"- {r['arquivo']} [{r['inicio']}-{r['fim']}]: pessoa={r['pessoa']} "
                 f"({r['conf_pessoa']:.0%}), emocao={r['emocao']} ({r['conf_emocao']:.0%})")

    L += ["", "AVISO", AVISO_ETICO, "", "GRAFICOS (pasta graficos/)",
          "- tempo_fala_por_participante.png", "- emocoes_por_participante.png",
          "- distribuicao_geral_emocoes.png", "- linha_do_tempo.png", "- baixa_confianca.png",
          "- matriz_confusao_pessoas.png", "- matriz_confusao_emocoes.png"]
    return "\n".join(L)


def gerar_relatorio(pasta, log=print):
    pasta = Path(pasta)
    cfg = carregar_config()
    df = ler_registros(pasta)
    if df.empty:
        raise RuntimeError("O CSV esta vazio.")
    res = calcular(df, cfg)
    graf = pasta / "graficos"
    graf.mkdir(exist_ok=True)

    figs = []
    if res["pessoas"]:
        figs.append(graf_tempo(res, graf / "tempo_fala_por_participante.png"))
        figs.append(graf_emocoes_pessoa(res, graf / "emocoes_por_participante.png"))
        figs.append(graf_linha_tempo(df, res, graf / "linha_do_tempo.png"))
    figs.append(graf_geral(res, graf / "distribuicao_geral_emocoes.png"))
    figs.append(graf_baixa(res, graf / "baixa_confianca.png"))

    # Matrizes de confusao dos modelos tambem entram nos graficos (slide 39)
    for nome in ("matriz_confusao_pessoas.png", "matriz_confusao_emocoes.png"):
        if (MODELOS / nome).exists():
            shutil.copy(MODELOS / nome, graf / nome)

    texto = texto_relatorio(pasta.name, res, cfg)
    (pasta / "relatorio.txt").write_text(texto, encoding="utf-8")

    # PDF (bonus): texto + graficos
    pdf_path = pasta / "relatorio.pdf"
    with PdfPages(pdf_path) as pdf:
        linhas = texto.split("\n")
        for i in range(0, len(linhas), 55):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.07, 0.96, "\n".join(linhas[i:i + 55]), va="top", family="monospace", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
        for fig in figs:
            pdf.savefig(fig)
        for nome in ("matriz_confusao_pessoas.png", "matriz_confusao_emocoes.png"):
            if (graf / nome).exists():
                img = plt.imread(graf / nome)
                fig, ax = plt.subplots(figsize=(8, 7))
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
    for fig in figs:
        plt.close(fig)

    log(f"Relatorio salvo em: {pasta / 'relatorio.txt'}")
    log(f"PDF salvo em: {pdf_path}")
    log(f"Graficos salvos em: {graf}")
    return res
