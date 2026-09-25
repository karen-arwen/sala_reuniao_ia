"""Interface interativa da Sala de Reuniao Inteligente.

Rodar (na pasta principal do projeto):
    streamlit run src/app.py          (ou dois cliques em iniciar.bat no Windows)

A interface usa exatamente os mesmos modulos dos scripts 01-07
(common.py, treino.py, reuniao.py, relatorio.py). Nada aqui e simulado:
todo numero exibido vem dos audios gravados, dos modelos treinados ou do CSV.
"""

import sys
import time
import csv
import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATASET_PESSOAS, DATASET_EMOCOES, MODELOS, REUNIOES, SAMPLE_RATE, EMOCOES,
    SILENCIO, DESCONHECIDO, INCERTO,
    carregar_config, salvar_config, contar_dataset, listar_classes, limpar_nome,
    proximo_arquivo, salvar_wav, carregar_audio, analisar_nivel, eh_silencio,
    modelos_existem, carregar_modelos, classificar, load_json,
)
from reuniao import (  # noqa: E402
    listar_reunioes, nova_reuniao, SessaoReuniao, CapturaContinua, gravar_trecho, blocos_de_arquivo,
)
from relatorio import (  # noqa: E402
    ler_registros, calcular, gerar_relatorio, mmss, CORES_EMOCAO, cor_pessoa, AVISO_ETICO,
)
import treino  # noqa: E402

st.set_page_config(page_title="Sala de Reunião Inteligente", page_icon=":material/graphic_eq:",
                   layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Identidade visual
# ---------------------------------------------------------------------------
ROSA = "#C8457E"
ROSA_CLARO = "#FBEAF1"
TINTA = "#2A1E2C"
SUAVE = "#7B6B7E"
BORDA = "#EFE2E8"
FONTE = "Plus Jakarta Sans"
LOG_TESTES = MODELOS / "testes_ao_vivo.csv"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, p, li, label, input, textarea, button, h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"], [data-testid="stMetricValue"], [data-testid="stWidgetLabel"] {{
    font-family: '{FONTE}', system-ui, sans-serif !important;
}}
.stApp {{ background: radial-gradient(1200px 500px at 85% -10%, #FCE7F0 0%, rgba(252,231,240,0) 60%), #FCF9FA; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}
.block-container {{ padding-top: 4.2rem; max-width: 1240px; }}
h1, h2, h3 {{ color: {TINTA} !important; letter-spacing: -.02em; }}
h2 {{ font-size: 1.25rem !important; font-weight: 700 !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 700 !important; }}

/* cabecalho de pagina */
.hero {{ margin: 6px 0 22px 0; }}
.eyebrow {{ font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: {ROSA}; }}
.hero h1 {{ font-size: 2.05rem; font-weight: 800; margin: 6px 0 6px 0; padding: 0; color: {TINTA}; }}
.hero p {{ color: {SUAVE}; margin: 0; font-size: .98rem; max-width: 760px; }}

/* cartoes */
.card {{ background: #FFFFFF; border: 1px solid {BORDA}; border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(42,30,44,.04), 0 8px 24px rgba(200,69,126,.06); margin-bottom: 12px; }}
.card-title {{ font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: {SUAVE}; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 18px; }}
.kpi {{ background: #FFF; border: 1px solid {BORDA}; border-radius: 16px; padding: 14px 18px;
       box-shadow: 0 8px 24px rgba(200,69,126,.05); }}
.kpi .l {{ font-size: .78rem; color: {SUAVE}; font-weight: 600; }}
.kpi .v {{ font-size: 1.7rem; font-weight: 800; color: {TINTA}; margin-top: 2px; letter-spacing: -.02em; }}
.kpi .s {{ font-size: .78rem; color: {SUAVE}; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.kpi.destaque {{ background: linear-gradient(135deg, #D2548A 0%, #B23A70 100%); border: none; }}
.kpi.destaque .l, .kpi.destaque .s {{ color: rgba(255,255,255,.85); }}
.kpi.destaque .v {{ color: #FFF; }}

.big {{ font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; color: {TINTA}; line-height: 1.15; margin: 6px 0; }}
.muted {{ color: {SUAVE}; font-size: .88rem; }}
.frase {{ font-size: 1.35rem; font-weight: 600; color: {TINTA}; margin-top: 6px; line-height: 1.4; }}

/* checklist e etapas */
.check {{ display:flex; align-items:center; gap:10px; padding: 9px 0; border-bottom: 1px dashed {BORDA}; font-size: .93rem; color: {TINTA}; }}
.check:last-child {{ border-bottom: none; }}
.check .i {{ width: 20px; height: 20px; border-radius: 50%; flex: none; display:flex; align-items:center; justify-content:center; }}
.check .i.ok {{ background: {ROSA}; }}
.check .i.no {{ border: 2px solid #E3CFD9; }}
.check.pend {{ color: {SUAVE}; }}
.step {{ display:flex; gap: 14px; align-items: flex-start; padding: 10px 0; }}
.step .n {{ width: 30px; height: 30px; border-radius: 10px; background: {ROSA_CLARO}; color: {ROSA};
           font-weight: 800; display:flex; align-items:center; justify-content:center; flex: none; font-size: .9rem; }}
.step .t {{ font-weight: 700; color: {TINTA}; font-size: .95rem; }}
.step .d {{ color: {SUAVE}; font-size: .85rem; }}

/* etiquetas */
.pill {{ display:inline-flex; align-items:center; gap:7px; padding: 5px 11px; margin: 3px 4px 3px 0; border-radius: 999px;
        font-size: .8rem; font-weight: 600; background: #FFF; border: 1px solid {BORDA}; color: {TINTA}; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; display:inline-block; flex: none; }}
.tag {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem; font-weight: 700;
       background: {ROSA_CLARO}; color: {ROSA}; }}
.live {{ display:inline-flex; align-items:center; gap:8px; font-weight:700; color:{ROSA}; font-size:.85rem; }}
.live .dot {{ background:{ROSA}; animation: pulse 1.2s infinite; }}
@keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(200,69,126,.5); }} 100% {{ box-shadow: 0 0 0 10px rgba(200,69,126,0); }} }}
.bar {{ height: 8px; background: #F4E8EE; border-radius: 99px; overflow: hidden; margin: 4px 0 10px 0; }}
.bar > div {{ height: 100%; border-radius: 99px; }}
.row {{ display:flex; justify-content: space-between; align-items: center; font-size: .88rem; color: {TINTA}; }}

/* componentes do streamlit */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    border-radius: 12px !important; font-weight: 600 !important; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {{
    background: linear-gradient(135deg, #D2548A 0%, #B23A70 100%) !important; border: none !important;
    box-shadow: 0 6px 16px rgba(200,69,126,.25); }}
[data-testid="stForm"] {{ background: #FFF; border: 1px solid {BORDA} !important; border-radius: 16px !important; }}
[data-testid="stExpander"] details {{ background: #FFF; border: 1px solid {BORDA} !important; border-radius: 16px !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: #FFF; border: 1.5px dashed #E6C9D6; border-radius: 14px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #FFF; border-color: {BORDA} !important; border-radius: 16px !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDA}; border-radius: 12px; overflow: hidden; }}
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

CHECK_SVG = ('<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3.5" '
             'stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>')


# ---------------------------------------------------------------------------
# Componentes visuais
# ---------------------------------------------------------------------------
def esc(t):
    return html.escape(str(t))


def cabecalho(eyebrow, titulo, descricao=""):
    st.markdown(f'<div class="hero"><div class="eyebrow">{esc(eyebrow)}</div><h1>{esc(titulo)}</h1>'
                f'<p>{esc(descricao)}</p></div>', unsafe_allow_html=True)


def card(conteudo):
    st.markdown(f'<div class="card">{conteudo}</div>', unsafe_allow_html=True)


def kpis(itens):
    """itens: lista de (rotulo, valor, subtitulo) - o primeiro fica em destaque."""
    blocos = "".join(
        f'<div class="kpi{" destaque" if i == 0 else ""}"><div class="l">{esc(l)}</div>'
        f'<div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>'
        for i, (l, v, s) in enumerate(itens))
    st.markdown(f'<div class="kpis">{blocos}</div>', unsafe_allow_html=True)


def pill(texto, cor):
    return f'<span class="pill"><span class="dot" style="background:{cor}"></span>{esc(texto)}</span>'


def barras_html(valores, cores):
    """Barras horizontais simples: {rotulo: fracao 0-1}."""
    out = ""
    for k, v in valores.items():
        out += (f'<div class="row"><span>{esc(k)}</span><b>{v:.0%}</b></div>'
                f'<div class="bar"><div style="width:{v*100:.1f}%;background:{cores.get(k, ROSA)}"></div></div>')
    return out


def curto(seg):
    seg = int(round(seg))
    return f"{seg // 60}:{seg % 60:02d}"


def layout(fig, h=360):
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=46, b=8), template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=f"{FONTE}, sans-serif", color=TINTA, size=12),
        title_font=dict(size=15, color=TINTA), legend=dict(orientation="h", y=-0.18, title_text=""),
        hoverlabel=dict(font_family=FONTE))
    fig.update_xaxes(gridcolor="#F3E9EE", zerolinecolor="#F3E9EE")
    fig.update_yaxes(gridcolor="#F3E9EE", zerolinecolor="#F3E9EE")
    return fig


def show(fig, key=None, h=360):
    with st.container(border=True):
        st.plotly_chart(layout(fig, h), width="stretch", key=key, config={"displayModeBar": False})


def mic_disponivel():
    try:
        import sounddevice as sd
        dev = carregar_config().get("dispositivo_entrada")
        sd.query_devices(dev if dev is not None else sd.default.device[0])
        return True
    except Exception:
        return False


def countdown_e_gravar(dur, ph):
    for n in (3, 2, 1):
        ph.markdown(f'<div class="card"><div class="card-title">Prepare-se</div><div class="big">{n}</div></div>',
                    unsafe_allow_html=True)
        time.sleep(0.6)
    ph.markdown(f'<div class="card"><span class="live"><span class="dot"></span>GRAVANDO</span>'
                f'<div class="big">Fale agora</div><span class="muted">{dur:.1f} segundos</span></div>',
                unsafe_allow_html=True)
    y = gravar_trecho(dur)
    ph.empty()
    return y


def _hash(up):
    import hashlib
    return hashlib.md5(up.getvalue()).hexdigest()


def entrada_audio(key, dur, texto_botao="Gravar"):
    """Microfone do PC, gravador do navegador ou arquivo WAV.

    Retorna (audio, novo): audio e np.array 16 kHz (ou None); novo=True somente na
    execucao em que um audio NOVO chegou (evita salvar o mesmo audio duas vezes).
    """
    modos = ["Microfone do PC", "Gravador do navegador", "Arquivo WAV"]
    modo = st.segmented_control("Fonte do áudio", modos, key=f"{key}_modo",
                                default=modos[0] if mic_disponivel() else modos[1]) or modos[1]
    geracao = st.session_state.get(f"{key}_geracao", 0)   # muda para limpar o gravador apos salvar
    novo = False
    if modo == modos[0]:
        ph = st.empty()
        if st.button(texto_botao, key=f"{key}_btn", type="primary", icon=":material/mic:"):
            try:
                st.session_state[f"{key}_audio"] = countdown_e_gravar(dur, ph)
                novo = True
            except Exception as exc:
                st.error(f"Não consegui usar o microfone do PC ({exc}). Escolha outro microfone em "
                         "Configurações ou use o 'Gravador do navegador'.", icon=":material/mic_off:")
    else:
        if modo == modos[1]:
            up = st.audio_input("Clique no microfone, fale e clique no quadrado para parar",
                                key=f"{key}_ai_{geracao}")
        else:
            up = st.file_uploader("Selecione um arquivo .wav", type=["wav"], key=f"{key}_up_{geracao}")
        if up is not None:
            h = _hash(up)
            if st.session_state.get(f"{key}_hash") != h:
                st.session_state[f"{key}_hash"] = h
                st.session_state[f"{key}_audio"] = carregar_audio(up)
                novo = True
    return st.session_state.get(f"{key}_audio"), novo


def limpar_entrada(key):
    """Depois de salvar/descartar: esquece o audio e reseta o gravador do navegador."""
    st.session_state.pop(f"{key}_audio", None)
    st.session_state.pop(f"{key}_hash", None)
    st.session_state[f"{key}_geracao"] = st.session_state.get(f"{key}_geracao", 0) + 1


def qualidade(y, cfg):
    """Mostra o nivel do audio. Retorna False se parecer vazio (mas ainda deixa salvar)."""
    n = analisar_nivel(y, cfg)
    vazio = eh_silencio(y, cfg)
    if vazio:
        st.warning(f"O áudio parece muito baixo (fala detectada em {n['fracao_fala']:.0%} do tempo, pico {n['pico']:.3f}). "
                   "Ouça antes de salvar. Se estiver mudo, confira o microfone em Configurações.",
                   icon=":material/volume_off:")
    elif n["pico"] > 0.99:
        st.warning("O áudio saturou (volume muito alto). Afaste um pouco o microfone.", icon=":material/warning:")
    st.caption(f"Duração {len(y)/SAMPLE_RATE:.1f} s  ·  fala em {n['fracao_fala']:.0%} do trecho  ·  pico {n['pico']:.2f}")
    return not vazio


# ===========================================================================
# PAGINAS
# ===========================================================================
def pagina_painel():
    cfg = carregar_config()
    cabecalho("Visão geral", "Sala de Reunião Inteligente",
              "Identificação de voz, emoção vocal estimada e relatório automático da reunião.")

    cp, ce = contar_dataset(DATASET_PESSOAS), contar_dataset(DATASET_EMOCOES)
    pessoas_validas = {k: v for k, v in cp.items() if v > 0 and k != DESCONHECIDO}
    reunioes_csv = [r for r in listar_reunioes() if (REUNIOES / r / "registros.csv").exists()]
    tem_relatorio = any((REUNIOES / r / "relatorio.txt").exists() for r in reunioes_csv)
    testes = pd.read_csv(LOG_TESTES) if LOG_TESTES.exists() else pd.DataFrame()

    checklist = [
        ("Pasta principal sala_reuniao_ia/ criada", True),
        (f"5 pessoas em dataset_pessoas/ (atual: {len(pessoas_validas)})", len(pessoas_validas) >= 5),
        (f"Pelo menos {cfg['minimo_por_pessoa']} áudios por pessoa",
         bool(pessoas_validas) and min(pessoas_validas.values()) >= cfg["minimo_por_pessoa"]),
        ("4 emoções em dataset_emocoes/", all(ce.get(e, 0) > 0 for e in EMOCOES)),
        (f"Pelo menos {cfg['minimo_por_emocao']} áudios por emoção",
         all(ce.get(e, 0) >= cfg["minimo_por_emocao"] for e in EMOCOES)),
        ("modelo_pessoas.pkl treinado", (MODELOS / "modelo_pessoas.pkl").exists()),
        ("modelo_emocoes.pkl treinado", (MODELOS / "modelo_emocoes.pkl").exists()),
        ("Matriz de confusão dos dois modelos",
         (MODELOS / "matriz_confusao_pessoas.png").exists() and (MODELOS / "matriz_confusao_emocoes.png").exists()),
        (f"Teste com frase nova ({len(testes)} registrados)", len(testes) > 0),
        ("Reunião simulada executada", len(reunioes_csv) > 0),
        ("registros.csv gerado", len(reunioes_csv) > 0),
        ("relatorio.txt e gráficos gerados", tem_relatorio),
    ]
    feitos = sum(ok for _, ok in checklist)

    kpis([("Progresso da entrega", f"{feitos}/{len(checklist)}", f"{feitos/len(checklist):.0%} concluído"),
          ("Áudios de pessoas", sum(cp.values()), f"{len(pessoas_validas)} participantes"),
          ("Áudios de emoções", sum(ce.values()), "alegre, neutro, triste, irritado"),
          ("Reuniões analisadas", len(reunioes_csv), "com registros.csv")])

    col_a, col_b = st.columns([1.15, 1], gap="large")
    with col_a:
        itens = "".join(
            f'<div class="check{"" if ok else " pend"}"><span class="i {"ok" if ok else "no"}">'
            f'{CHECK_SVG if ok else ""}</span>{esc(txt)}</div>' for txt, ok in checklist)
        card(f'<div class="card-title">Checklist antes de chamar o professor</div><div style="margin-top:6px">{itens}</div>')
    with col_b:
        passos = [("Datasets", "Gravar as vozes das pessoas e as emoções"),
                  ("Treinamento", "Dois Random Forests com validação cruzada"),
                  ("Testar", "Frase nova e pessoa não cadastrada"),
                  ("Reunião", "Blocos de 3 s classificados e salvos no CSV"),
                  ("Relatório", "Percentuais, gráficos e PDF")]
        etapas = "".join(f'<div class="step"><div class="n">{i}</div><div><div class="t">{t}</div>'
                         f'<div class="d">{d}</div></div></div>' for i, (t, d) in enumerate(passos, 1))
        card(f'<div class="card-title">Fluxo do sistema</div>{etapas}')

        modelos_txt = ""
        for nome, rot in (("pessoas", "Modelo de pessoas"), ("emocoes", "Modelo de emoções")):
            meta = MODELOS / f"metadados_{nome}.json"
            if meta.exists():
                m = load_json(meta)
                modelos_txt += (f'<div class="row" style="padding:6px 0"><span>{rot}</span>'
                                f'<span class="tag">{m["acuracia_validacao"]:.1%} de acurácia</span></div>')
        if modelos_txt:
            card(f'<div class="card-title">Modelos treinados</div>{modelos_txt}')


def grafico_contagem(counts, minimo, titulo, cores=None):
    df = pd.DataFrame({"classe": list(counts), "audios": list(counts.values())})
    cores = cores or {c: ROSA for c in counts}
    fig = px.bar(df, x="classe", y="audios", text="audios", title=titulo,
                 color="classe", color_discrete_map=cores)
    fig.update_traces(marker_line_width=0, textposition="outside", width=0.55)
    fig.add_hline(y=minimo, line_dash="dot", line_color="#B9A3AF",
                  annotation_text=f"mínimo {minimo}", annotation_position="top left",
                  annotation_font_color=SUAVE)
    topo = max([minimo] + list(counts.values())) * 1.2
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="áudios", yaxis_range=[0, topo])
    return fig


def lista_audios(pasta, key):
    wavs = sorted(Path(pasta).glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        st.caption("Nenhum áudio nesta classe ainda.")
        return
    with st.expander(f"Áudios gravados nesta classe ({len(wavs)})", icon=":material/library_music:"):
        for w in wavs[:15]:
            c1, c2, c3 = st.columns([2, 6, 1], vertical_alignment="center")
            c1.markdown(f"<span class='muted'>{esc(w.name)}</span>", unsafe_allow_html=True)
            c2.audio(str(w))
            if c3.button("", key=f"del_{key}_{w.name}", icon=":material/delete:", help="Apagar este áudio"):
                w.unlink()
                st.rerun()
        if len(wavs) > 15:
            st.caption(f"Mostrando os 15 mais recentes de {len(wavs)}.")


def gravar_no_dataset(key, dur, cfg, destino_fn, pasta_txt):
    """Fluxo completo: gravar -> (salvar automatico ou revisar) -> confirmacao visivel."""
    auto = st.toggle("Salvar automaticamente logo depois de gravar", value=True, key=f"{key}_auto",
                     help="Desligue para ouvir cada áudio antes de decidir se salva.")
    y, novo = entrada_audio(key, dur, "Gravar frase")

    msg = st.session_state.pop(f"{key}_msg", None)
    if msg:
        st.success(msg["texto"], icon=":material/check_circle:")
        if Path(msg["arquivo"]).exists():
            st.audio(msg["arquivo"])

    if y is None:
        return

    def salvar():
        try:
            nome = destino_fn(y)
        except Exception as exc:
            st.error(f"Erro ao salvar o arquivo: {exc}", icon=":material/error:")
            return
        st.session_state[f"{key}_msg"] = {
            "texto": f"Áudio salvo: {pasta_txt}/{nome}", "arquivo": str(nome_para_caminho(pasta_txt, nome))}
        limpar_entrada(key)
        st.rerun()

    if auto and novo and not eh_silencio(y, cfg):
        salvar()

    st.audio(y, sample_rate=SAMPLE_RATE)
    bom = qualidade(y, cfg)
    c1, c2, _ = st.columns([1.2, 1, 3])
    rotulo = "Salvar no dataset" if bom else "Salvar mesmo assim"
    if c1.button(rotulo, key=f"save_{key}", type="primary", icon=":material/check:"):
        salvar()
    if c2.button("Descartar", key=f"disc_{key}", icon=":material/close:"):
        limpar_entrada(key)
        st.rerun()


def nome_para_caminho(pasta_txt, nome):
    base = DATASET_PESSOAS.parent
    return base / pasta_txt / nome


def pagina_datasets():
    cfg = carregar_config()
    cabecalho("Etapa 1", "Datasets de voz",
              "Grave em ambiente silencioso, com a mesma distância do microfone e voz natural. "
              "Grave apenas quem concordou em participar.")
    aba_p, aba_e = st.tabs(["Pessoas  ·  quem falou?", "Emoções  ·  como soou?"])
    dur = cfg["duracao_gravacao"]

    with aba_p:
        counts = contar_dataset(DATASET_PESSOAS)
        c1, c2 = st.columns([1.5, 1], gap="large")
        with c1:
            if counts:
                show(grafico_contagem(counts, cfg["minimo_por_pessoa"], "Áudios por pessoa",
                                      {c: cor_pessoa(c, list(counts)) for c in counts}), "cnt_p", 320)
            else:
                st.info("Nenhum participante cadastrado ainda.", icon=":material/info:")
        with c2:
            with st.form("nova_pessoa", clear_on_submit=True):
                st.markdown("**Cadastrar participante**")
                nome = st.text_input("Nome", placeholder="Ex.: Ana Souza", label_visibility="collapsed")
                st.caption("O nome vira a pasta sem acentos e sem espaços (ana_souza).")
                if st.form_submit_button("Criar participante", type="primary", icon=":material/person_add:") and nome:
                    limpo = limpar_nome(nome)
                    (DATASET_PESSOAS / limpo).mkdir(parents=True, exist_ok=True)
                    st.session_state["ds_pessoa"] = limpo
                    st.rerun()
            st.caption("Bônus: crie a classe **desconhecido** com vozes de pessoas de fora do grupo "
                       "para o modelo aprender a reconhecer quem não é cadastrado.")

        classes = listar_classes(DATASET_PESSOAS)
        if classes:
            st.markdown("### Gravar")
            pessoa = st.selectbox("Participante", classes, key="ds_pessoa")
            n = counts.get(pessoa, 0)
            frase = cfg["frases"][n % len(cfg["frases"])]
            meta = 30
            card(f'<div class="row"><span class="card-title">Frase {n + 1}</span>'
                 f'<span class="muted">{n} de {meta} áudios · mínimo {cfg["minimo_por_pessoa"]}</span></div>'
                 f'<div class="bar"><div style="width:{min(1, n/meta)*100:.0f}%;background:{ROSA}"></div></div>'
                 f'<div class="frase">“{esc(frase)}”</div>')
            def salvar(y):
                path = proximo_arquivo(DATASET_PESSOAS / pessoa, pessoa)
                salvar_wav(path, y)
                return path.name
            gravar_no_dataset("rec_p", dur, cfg, salvar, f"dataset_pessoas/{pessoa}")
            lista_audios(DATASET_PESSOAS / pessoa, f"p_{pessoa}")

    with aba_e:
        for e in EMOCOES:
            (DATASET_EMOCOES / e).mkdir(parents=True, exist_ok=True)
        counts = contar_dataset(DATASET_EMOCOES)
        c1, c2 = st.columns([1.5, 1], gap="large")
        with c1:
            show(grafico_contagem({e: counts.get(e, 0) for e in EMOCOES}, cfg["minimo_por_emocao"],
                                  "Áudios por emoção", CORES_EMOCAO), "cnt_e", 320)
        with c2:
            linhas = []
            for e in EMOCOES:
                for w in (DATASET_EMOCOES / e).glob("*.wav"):
                    partes = w.stem.split("_")
                    linhas.append({"emocao": e, "pessoa": "_".join(partes[1:-1]) or "sem_nome", "arquivo": w.name})
            log = pd.DataFrame(linhas)
            if len(log):
                tab = log.pivot_table(index="pessoa", columns="emocao", values="arquivo",
                                      aggfunc="count", fill_value=0)
                tab = tab.reindex(columns=[e for e in EMOCOES if e in tab.columns])
                st.markdown("**Quem gravou cada emoção**")
                st.caption("Variar as pessoas ajuda o modelo a generalizar.")
                st.dataframe(tab, width="stretch")
            else:
                card('<div class="card-title">Dica</div><p class="muted" style="margin-top:6px">'
                     'Grave a mesma frase nas quatro emoções e peça para mais de uma pessoa gravar. '
                     'Classes equilibradas evitam um modelo viciado.</p>')

        st.markdown("### Gravar")
        c1, c2 = st.columns([1, 2])
        quem = c1.selectbox("Quem está gravando", listar_classes(DATASET_PESSOAS) or ["anonimo"], key="ds_quem")
        with c2:
            emocao = st.segmented_control("Emoção", EMOCOES, key="ds_emo", default="neutro") or "neutro"
        dicas = {"alegre": "sorrindo, entonação animada", "neutro": "voz normal, sem ênfase",
                 "triste": "mais devagar e mais baixo", "irritado": "firme e impaciente, sem gritar"}
        feitas = len(list((DATASET_EMOCOES / "neutro").glob(f"neutro_{quem}_*.wav")))
        frase = cfg["frases"][feitas % len(cfg["frases"])]
        card(f'<div class="row"><span class="card-title">Emoção: {emocao}</span>'
             f'<span class="muted">{counts.get(emocao, 0)} áudios · mínimo {cfg["minimo_por_emocao"]}</span></div>'
             f'<div class="muted" style="margin-top:4px">{dicas[emocao]}</div>'
             f'<div class="frase">“{esc(frase)}”</div>')
        def salvar_e(y):
            # o nome inclui quem gravou (alegre_dani_001.wav): cada pessoa grava em casa
            # sem conflito de nomes ao juntar tudo no GitHub
            path = proximo_arquivo(DATASET_EMOCOES / emocao, f"{emocao}_{quem}")
            salvar_wav(path, y)
            return path.name
        gravar_no_dataset("rec_e", dur, cfg, salvar_e, f"dataset_emocoes/{emocao}")
        lista_audios(DATASET_EMOCOES / emocao, f"e_{emocao}")


def mostrar_treino(nome):
    meta_path = MODELOS / f"metadados_{nome}.json"
    if not meta_path.exists():
        st.info("Este modelo ainda não foi treinado.", icon=":material/info:")
        return
    m = load_json(meta_path)
    kpis([("Acurácia", f"{m['acuracia_validacao']:.1%}", "validação cruzada"),
          ("Áudios usados", m["quantidade_total"], "no treino"),
          ("Classes", len(m["classes"]), ", ".join(m["classes"])),
          ("Treinado em", m["treinado_em"][11:16], m["treinado_em"][:10])])
    for a in m.get("avisos_dataset", []):
        st.warning(a, icon=":material/warning:")

    labels, cm = m["classes"], np.array(m["matriz_confusao"])
    escala = [[0, "#FDF6F9"], [0.5, "#E48BB0"], [1, "#9E2F62"]]
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale=escala,
                    labels=dict(x="Previsto", y="Real", color="Qtd"),
                    title="Matriz de confusão  ·  diagonal = acertos")
    fig.update_coloraxes(showscale=False)
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        show(fig, f"cm_{nome}", 430)
    with c2:
        rel = pd.DataFrame(m["relatorio_por_classe"]).T[["precision", "recall", "f1-score", "support"]]
        rel.columns = ["Precisão", "Recall", "F1", "Áudios"]
        st.dataframe(rel.style.format({"Precisão": "{:.0%}", "Recall": "{:.0%}", "F1": "{:.0%}",
                                       "Áudios": "{:.0f}"}), width="stretch")
        if m["principais_confusoes"]:
            conf = "".join(f'<div class="row" style="padding:5px 0"><span><b>{esc(c["real"])}</b> previsto como '
                           f'<b>{esc(c["previsto"])}</b></span><span class="tag">{c["quantidade"]}x</span></div>'
                           for c in m["principais_confusoes"][:3])
            card(f'<div class="card-title">Onde o modelo mais confunde</div>{conf}'
                 f'<div class="muted" style="margin-top:6px">Explique essas confusões na apresentação.</div>')
        fam = m["importancia_por_familia"]
        tot = sum(fam.values()) or 1
        card('<div class="card-title">O que o Random Forest mais usa</div><div style="margin-top:8px">'
             + barras_html({k: v / tot for k, v in fam.items()}, {}) + "</div>")


def pagina_treino():
    cabecalho("Etapa 2", "Treinamento dos modelos",
              "Features acústicas (MFCC, energia, pitch, ZCR e espectro) alimentam dois Random Forests "
              "de 300 árvores, avaliados com validação cruzada estratificada.")
    abas = st.tabs(["Modelo de pessoas", "Modelo de emoções"])
    for aba, nome in zip(abas, ["pessoas", "emocoes"]):
        with aba:
            counts = contar_dataset(treino.TIPOS[nome]["dataset"])
            linha = "".join(pill(f"{k}: {v}", cor_pessoa(k, list(counts)) if nome == "pessoas"
                                 else CORES_EMOCAO.get(k, ROSA)) for k, v in counts.items())
            st.markdown(f'<div class="muted" style="margin-bottom:6px">Contagem antes do treino</div>{linha}',
                        unsafe_allow_html=True)
            st.write("")
            if st.button(f"Treinar modelo de {'pessoas' if nome == 'pessoas' else 'emoções'}",
                         key=f"train_{nome}", type="primary", icon=":material/model_training:"):
                barra = st.progress(0.0, text="Iniciando...")
                logs = []
                try:
                    treino.treinar(nome, progresso=lambda f, t: barra.progress(min(f, 1.0), text=t),
                                   log=logs.append)
                    st.success("Modelo treinado e salvo em modelos/", icon=":material/check_circle:")
                except Exception as exc:
                    st.error(str(exc))
                with st.expander("Log do treinamento", icon=":material/terminal:"):
                    st.code("\n".join(logs), language=None)
            mostrar_treino(nome)


def pagina_testar():
    cfg = carregar_config()
    cabecalho("Etapa 3", "Testar os modelos",
              "Peça para uma pessoa cadastrada falar uma frase nova, para alguém de fora falar, "
              "ou repita a mesma frase com emoções diferentes.")
    if not modelos_existem():
        st.warning("Treine os dois modelos primeiro.", icon=":material/warning:")
        return
    modelos = carregar_modelos()
    y, _ = entrada_audio("teste", cfg["duracao_gravacao"], "Gravar e classificar")
    if y is None:
        return
    st.audio(y, sample_rate=SAMPLE_RATE)
    r = classificar(y, modelos, cfg)
    if r["status"] == SILENCIO:
        st.warning("Nenhuma fala detectada. Fale mais perto do microfone.", icon=":material/volume_off:")
        return

    todas = list(r["probs_pessoa"])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        probs = dict(sorted(r["probs_pessoa"].items(), key=lambda kv: -kv[1]))
        aviso = f'<div class="muted" style="margin-top:6px">{esc(r["motivo"])}</div>' if r["motivo"] else ""
        card(f'<div class="card-title">Quem está falando</div>'
             f'<div class="big">{esc(r["pessoa"])}</div>'
             f'<span class="tag">confiança {r["conf_pessoa"]:.0%}</span>{aviso}'
             f'<div style="margin-top:16px">{barras_html(probs, {p: cor_pessoa(p, todas) for p in todas})}</div>')
    with c2:
        probs = dict(sorted(r["probs_emocao"].items(), key=lambda kv: -kv[1]))
        card(f'<div class="card-title">Emoção vocal estimada</div>'
             f'<div class="big">{esc(r["emocao"])}</div>'
             f'<span class="tag">confiança {r["conf_emocao"]:.0%}</span>'
             f'<div style="margin-top:16px">{barras_html(probs, CORES_EMOCAO)}</div>')

    with st.form("registrar_teste"):
        st.markdown("**Registrar este teste como evidência**")
        st.caption("Fica salvo em modelos/testes_ao_vivo.csv para mostrar na apresentação.")
        c1, c2, c3 = st.columns(3)
        opcoes = modelos["pessoas"].classes_.tolist() + ["nao_cadastrado"]
        prev = r["pessoa"] if r["pessoa"] in opcoes else "nao_cadastrado"
        real_p = c1.selectbox("Quem falou de verdade", opcoes, index=opcoes.index(prev))
        real_e = c2.selectbox("Emoção pedida", ["-"] + EMOCOES)
        tipo = c3.selectbox("Tipo de teste", ["frase nova", "pessoa nao cadastrada", "mesma frase outra emocao", "outro"])
        if st.form_submit_button("Registrar teste", type="primary", icon=":material/save:"):
            acerto_p = (r["pessoa"] == real_p) or (real_p == "nao_cadastrado" and r["pessoa"] == DESCONHECIDO)
            novo = not LOG_TESTES.exists()
            with LOG_TESTES.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if novo:
                    w.writerow(["data", "tipo", "pessoa_real", "pessoa_prevista", "conf_pessoa",
                                "emocao_pedida", "emocao_prevista", "conf_emocao", "acertou_pessoa"])
                w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tipo, real_p, r["pessoa"],
                            f"{r['conf_pessoa']:.4f}", real_e, r["emocao"], f"{r['conf_emocao']:.4f}", acerto_p])
            if acerto_p:
                st.success("Acertou a pessoa.", icon=":material/check_circle:")
            else:
                st.error("Errou a pessoa. Anote o motivo para explicar na apresentação.", icon=":material/cancel:")

    if LOG_TESTES.exists():
        df = pd.read_csv(LOG_TESTES)
        if len(df):
            st.markdown("### Histórico de testes")
            kpis([("Acerto de pessoa", f"{df['acertou_pessoa'].mean():.0%}", "nos testes ao vivo"),
                  ("Testes registrados", len(df), "em testes_ao_vivo.csv")])
            st.dataframe(df.iloc[::-1], width="stretch", hide_index=True)


def painel_ao_vivo(r, historico, ph_card, ph_chips, ph_graf, ph_tab, n):
    pessoa, emo = r["pessoa"], r["emocao"]
    if pessoa == SILENCIO:
        corpo = '<div class="big" style="color:#B9A3AF">Silêncio</div>'
    else:
        corpo = (f'<div class="big">{esc(pessoa)}</div>'
                 f'<span class="tag">pessoa {r["conf_pessoa"]:.0%}</span>&nbsp; '
                 + pill(f'{emo} · {r["conf_emocao"]:.0%}', CORES_EMOCAO.get(emo, "#B9A3AF")))
    ph_card.markdown(f'<div class="card"><div class="row"><span class="card-title">Falando agora</span>'
                     f'<span class="muted">{r["inicio"]} – {r["fim"]}</span></div>{corpo}</div>',
                     unsafe_allow_html=True)

    todas = sorted({h["pessoa"] for h in historico})
    chips = "".join(pill(f'{h["inicio"]}  {h["pessoa"]}', cor_pessoa(h["pessoa"], todas)) for h in historico[-14:])
    ph_chips.markdown(f'<div class="card"><div class="card-title">Últimos blocos</div>'
                      f'<div style="margin-top:8px">{chips}</div></div>', unsafe_allow_html=True)

    fala = [h for h in historico if h["pessoa"] != SILENCIO]
    if fala:
        df = pd.DataFrame(fala)
        df["seg"] = df["fim_s"] - df["inicio_s"]
        tempo = df.groupby("pessoa")["seg"].sum().sort_values()
        fig = go.Figure(go.Bar(x=tempo.values, y=tempo.index, orientation="h",
                               marker_color=[cor_pessoa(p, todas) for p in tempo.index],
                               text=[mmss(v) for v in tempo.values], textposition="auto"))
        fig.update_layout(title="Tempo de fala até agora", xaxis_title="segundos")
        ph_graf.plotly_chart(layout(fig, 300), width="stretch", key=f"live_{n}",
                             config={"displayModeBar": False})
    ph_tab.dataframe(pd.DataFrame(historico[::-1])[["inicio", "fim", "pessoa", "conf_pessoa", "emocao",
                                                     "conf_emocao", "arquivo"]].head(12),
                     width="stretch", hide_index=True)


def pagina_reuniao():
    cfg = carregar_config()
    cabecalho("Etapa 4", "Reunião ao vivo",
              "Uma pessoa fala por vez. Cada bloco é gravado, classificado (pessoa e emoção) e salvo no CSV.")
    if not modelos_existem():
        st.warning("Treine os dois modelos antes de iniciar a reunião.", icon=":material/warning:")
        return
    rodando = st.session_state.get("rodando", False)

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1, 1.4])
        alvo = c1.selectbox("Reunião", ["Nova reunião"] + listar_reunioes(), disabled=rodando, key="alvo_reuniao")
        with c2:
            dur = st.segmented_control("Duração do bloco", [3.0, 4.0, 5.0], default=float(cfg["duracao_bloco"]),
                                       format_func=lambda v: f"{v:.0f} s", disabled=rodando, key="dur_bloco") \
                or float(cfg["duracao_bloco"])
        with c3:
            modo = st.segmented_control("Entrada", ["Microfone", "Arquivo WAV"], default="Microfone",
                                        disabled=rodando, key="modo_reuniao") or "Microfone"

    if modo == "Arquivo WAV":
        up = st.file_uploader("Áudio de uma reunião gravada (.wav)", type=["wav"])
        if up is not None and st.button("Analisar arquivo", type="primary", icon=":material/play_arrow:"):
            pasta = nova_reuniao() if alvo == "Nova reunião" else REUNIOES / alvo
            modelos = carregar_modelos()
            blocos = list(blocos_de_arquivo(up, dur))
            barra = st.progress(0.0)
            historico = []
            col_a, col_b = st.columns(2, gap="large")
            with col_a:
                ph_card, ph_chips = st.empty(), st.empty()
            with col_b:
                ph_graf = st.empty()
            ph_tab = st.empty()
            with SessaoReuniao(pasta, modelos, cfg, dur) as s:
                for i, b in enumerate(blocos):
                    r = s.processar_bloco(b)
                    historico.append(r)
                    painel_ao_vivo(r, historico, ph_card, ph_chips, ph_graf, ph_tab, i)
                    barra.progress((i + 1) / len(blocos), text=f"Bloco {i+1} de {len(blocos)}")
            st.success(f"{len(blocos)} blocos salvos em reunioes/{pasta.name}/. Abra a página Relatório.",
                       icon=":material/check_circle:")
            st.session_state["reuniao_relatorio"] = pasta.name
        return

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Iniciar reunião", type="primary", disabled=rodando, key="btn_ini", icon=":material/play_arrow:"):
        pasta = nova_reuniao() if alvo == "Nova reunião" else REUNIOES / alvo
        st.session_state.update(rodando=True, pasta_ativa=pasta.name, dur_ativa=dur)
        st.rerun()
    if b2.button("Encerrar", disabled=not rodando, key="btn_fim", icon=":material/stop:"):
        st.session_state["rodando"] = False
        st.session_state["reuniao_relatorio"] = st.session_state.get("pasta_ativa")
        st.rerun()

    if not rodando:
        if st.session_state.get("reuniao_relatorio"):
            st.info(f"Última reunião: {st.session_state['reuniao_relatorio']}. Abra a página Relatório.",
                    icon=":material/info:")
        return

    pasta = REUNIOES / st.session_state["pasta_ativa"]
    st.markdown(f'<span class="live"><span class="dot"></span>GRAVANDO</span>'
                f'<span class="muted">&nbsp; reunioes/{pasta.name}/ · blocos de '
                f'{st.session_state["dur_ativa"]:.0f} s</span>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        ph_card = st.empty()
        ph_chips = st.empty()
    with col_b:
        ph_graf = st.empty()
    ph_tab = st.empty()
    ph_card.markdown('<div class="card"><div class="card-title">Falando agora</div>'
                     '<div class="big" style="color:#B9A3AF">Ouvindo...</div></div>', unsafe_allow_html=True)

    modelos = carregar_modelos()
    historico = []
    try:
        with SessaoReuniao(pasta, modelos, cfg, st.session_state["dur_ativa"]) as s, \
                CapturaContinua(st.session_state["dur_ativa"]) as cap:
            n = 0
            while st.session_state.get("rodando"):
                r = s.processar_bloco(cap.ler_bloco())
                historico.append(r)
                painel_ao_vivo(r, historico, ph_card, ph_chips, ph_graf, ph_tab, n)
                n += 1
    except Exception as exc:
        # StopException/RerunException do Streamlit passam direto (nao herdam de Exception)
        st.session_state["rodando"] = False
        st.error(f"Erro no microfone: {exc}")


def pagina_relatorio():
    cfg = carregar_config()
    cabecalho("Etapa 5", "Relatório da reunião",
              "Todos os números são calculados a partir do registros.csv.")
    reunioes = [r for r in listar_reunioes() if (REUNIOES / r / "registros.csv").exists()]
    if not reunioes:
        st.info("Nenhuma reunião registrada ainda. Rode uma na página Reunião.", icon=":material/info:")
        return
    padrao = st.session_state.get("reuniao_relatorio")
    idx = reunioes.index(padrao) if padrao in reunioes else len(reunioes) - 1
    c1, _ = st.columns([1, 2])
    nome = c1.selectbox("Reunião", reunioes, index=idx)
    pasta = REUNIOES / nome
    df = ler_registros(pasta)
    if df.empty:
        st.warning("CSV vazio.")
        return
    res = calcular(df, cfg)
    ps = res["pessoas"]

    kpis([("Duração", curto(res["duracao_total_s"]), "min:seg registrados"),
          ("Blocos", res["total_blocos"], f"{res['blocos_silencio']} de silêncio"),
          ("Tempo de fala", curto(res["tempo_fala_s"]), "min:seg"),
          ("Participantes", len([p for p in ps if p not in (DESCONHECIDO, INCERTO)]),
           f"mais falou: {res['mais_falou'] or '-'}"),
          ("Baixa confiança", len(res["baixa_confianca"]), "trechos para revisar")])

    a1, a2, a3, a4, a5 = st.tabs(["Visão geral", "Por participante", "Linha do tempo", "Baixa confiança", "CSV"])
    cores_p = {p: cor_pessoa(p, ps) for p in ps}
    with a1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            t = pd.DataFrame({"pessoa": ps, "segundos": [res["por_pessoa"][p]["tempo_s"] for p in ps]})
            fig = px.bar(t, x="pessoa", y="segundos", color="pessoa", color_discrete_map=cores_p,
                         text=[mmss(v) for v in t["segundos"]], title="Tempo de fala por participante")
            fig.update_traces(width=0.55, marker_line_width=0)
            fig.update_layout(showlegend=False, xaxis_title="")
            show(fig, "r_t")
        with c2:
            g = {e: v for e, v in res["emocoes_geral"].items() if v}
            fig = px.pie(names=list(g), values=list(g.values()), hole=.62, color=list(g),
                         color_discrete_map=CORES_EMOCAO, title="Distribuição geral das emoções vocais")
            fig.update_traces(textinfo="percent", marker_line_color="#FFF", marker_line_width=2)
            show(fig, "r_g")
        st.caption(AVISO_ETICO)
    with a2:
        linhas = [{"pessoa": p, "emocao": e, "percentual": res["por_pessoa"][p]["percentuais"][e]}
                  for p in ps for e in EMOCOES]
        fig = px.bar(pd.DataFrame(linhas), x="pessoa", y="percentual", color="emocao", text_auto=".0f",
                     color_discrete_map=CORES_EMOCAO, title="Emoções vocais por participante (%)")
        fig.update_traces(marker_line_width=0, width=0.55)
        fig.update_layout(xaxis_title="", yaxis_title="% dos trechos")
        show(fig, "r_e", 400)
        cols = st.columns(min(3, max(1, len(ps))), gap="medium")
        for i, p in enumerate(ps):
            d = res["por_pessoa"][p]
            dom = max(EMOCOES, key=lambda e: d["emocoes"][e])
            obs = (f'<div class="muted" style="margin-top:8px">{d["percentuais"][dom]:.0f}% dos trechos foram '
                   f'classificados como {dom}. Isso não prova sentimento real; é uma estimativa do modelo.</div>'
                   if d["emocoes"][dom] else "")
            with cols[i % len(cols)]:
                card(f'<div class="row"><span style="font-weight:800;font-size:1.1rem">{esc(p)}</span>'
                     f'<span class="dot" style="background:{cores_p[p]};width:12px;height:12px"></span></div>'
                     f'<div class="muted" style="margin:4px 0 12px 0">{mmss(d["tempo_s"])} · {d["trechos"]} trechos · '
                     f'confiança média {d["conf_media"]:.0%}</div>'
                     + barras_html({e: d["percentuais"][e] / 100 for e in EMOCOES}, CORES_EMOCAO) + obs)
    with a3:
        tl = df[df["pessoa"] != SILENCIO].copy()
        fig = go.Figure()
        for e in EMOCOES + [INCERTO]:
            d = tl[tl["emocao"] == e]
            if len(d):
                fig.add_bar(y=d["pessoa"], x=d["duracao_s"], base=d["inicio_s"], orientation="h",
                            name=e, marker_color=CORES_EMOCAO.get(e, "#C9BCC3"), marker_line_color="#FFF",
                            marker_line_width=1,
                            customdata=d[["inicio", "fim", "conf_pessoa", "conf_emocao"]],
                            hovertemplate="%{y}  %{customdata[0]}–%{customdata[1]}<br>pessoa %{customdata[2]:.0%}"
                                          " · emoção %{customdata[3]:.0%}<extra></extra>")
        fig.update_layout(barmode="overlay", title="Quem falou quando (cor = emoção vocal estimada)",
                          xaxis_title="segundos", yaxis=dict(categoryorder="array", categoryarray=ps[::-1]))
        show(fig, "r_tl", 140 + 48 * len(ps))
    with a4:
        b = res["baixa_confianca"]
        st.caption(f"Critério: confiança de pessoa abaixo de {cfg['limiar_pessoa']:.0%}, de emoção abaixo de "
                   f"{cfg['limiar_emocao']:.0%} ou pessoa desconhecida. Probabilidade baixa não é erro: é informação.")
        if len(b):
            st.dataframe(b[["inicio", "fim", "pessoa", "conf_pessoa", "emocao", "conf_emocao", "arquivo"]]
                         .style.format({"conf_pessoa": "{:.0%}", "conf_emocao": "{:.0%}"}),
                         width="stretch", hide_index=True)
            c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
            ouvir = c1.selectbox("Ouvir trecho", b["arquivo"].tolist())
            wav = pasta / "blocos_audio" / ouvir
            if wav.exists():
                c2.audio(str(wav))
        else:
            st.success("Nenhum trecho abaixo do limiar.", icon=":material/check_circle:")
    with a5:
        st.dataframe(df[["inicio", "fim", "pessoa", "conf_pessoa", "emocao", "conf_emocao", "arquivo"]],
                     width="stretch", hide_index=True)

    st.markdown("### Arquivos da entrega")
    st.caption("Gerar relatório cria relatorio.txt, relatorio.pdf e a pasta graficos/ dentro da reunião.")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Gerar relatório", type="primary", icon=":material/description:"):
        with st.spinner("Gerando..."):
            gerar_relatorio(pasta, log=lambda *_: None)
        st.success(f"Arquivos gerados em reunioes/{nome}/", icon=":material/check_circle:")
    c2.download_button("registros.csv", (pasta / "registros.csv").read_bytes(), f"{nome}_registros.csv",
                       icon=":material/download:")
    if (pasta / "relatorio.txt").exists():
        c3.download_button("relatorio.txt", (pasta / "relatorio.txt").read_bytes(), f"{nome}_relatorio.txt",
                           icon=":material/download:")
    if (pasta / "relatorio.pdf").exists():
        c4.download_button("relatorio.pdf", (pasta / "relatorio.pdf").read_bytes(), f"{nome}_relatorio.pdf",
                           icon=":material/download:")
    if (pasta / "relatorio.txt").exists():
        with st.expander("Ver relatorio.txt", icon=":material/article:"):
            st.code((pasta / "relatorio.txt").read_text(encoding="utf-8"), language=None)


# ---------------------------------------------------------------------------
# Backup (para trocar de computador sem perder gravacoes)
# ---------------------------------------------------------------------------
PASTAS_BACKUP = ["dataset_pessoas", "dataset_emocoes", "modelos", "reunioes"]


def gerar_backup_zip():
    import io
    import zipfile
    raiz = DATASET_PESSOAS.parent
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for pasta in PASTAS_BACKUP:
            for f in (raiz / pasta).rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts:
                    z.write(f, f.relative_to(raiz).as_posix())
                    n += 1
        if (raiz / "config.json").exists():
            z.write(raiz / "config.json", "config.json")
            n += 1
    return buf.getvalue(), n


def restaurar_backup_zip(arquivo):
    """Extrai so as pastas do projeto e nunca sobrescreve arquivos que ja existem."""
    import zipfile
    raiz = DATASET_PESSOAS.parent.resolve()
    novos, pulados = 0, 0
    with zipfile.ZipFile(arquivo) as z:
        for info in z.infolist():
            nome = info.filename.replace("\\", "/")
            partes = [p for p in nome.split("/") if p]
            # aceita zip com uma pasta-mae (ex.: sala_reuniao_ia/dataset_pessoas/...)
            while partes and partes[0] not in PASTAS_BACKUP + ["config.json"]:
                partes = partes[1:]
            if not partes or info.is_dir() or ".." in partes:
                continue
            destino = (raiz / Path(*partes)).resolve()
            if raiz not in destino.parents:
                continue
            if destino.exists():
                pulados += 1
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(z.read(info))
            novos += 1
    return novos, pulados


def secao_backup():
    st.markdown("### Backup e troca de computador")
    st.caption("Leve suas gravações, modelos e reuniões para outro PC: gere o backup aqui, "
               "copie o .zip (pendrive, Drive, e-mail) e restaure no outro computador.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("**Gerar backup**")
            cp, ce = contar_dataset(DATASET_PESSOAS), contar_dataset(DATASET_EMOCOES)
            st.caption(f"{sum(cp.values())} áudios de pessoas · {sum(ce.values())} de emoções · "
                       f"{len(listar_reunioes())} reuniões")
            if st.button("Preparar backup", icon=":material/inventory_2:"):
                dados, n = gerar_backup_zip()
                st.session_state["backup_zip"] = (dados, n)
            if st.session_state.get("backup_zip"):
                dados, n = st.session_state["backup_zip"]
                st.download_button(f"Baixar backup ({n} arquivos, {len(dados)/1e6:.1f} MB)", dados,
                                   f"backup_sala_reuniao_{datetime.now():%Y%m%d_%H%M}.zip",
                                   mime="application/zip", type="primary", icon=":material/download:")
    with c2:
        with st.container(border=True):
            st.markdown("**Restaurar backup**")
            st.caption("Arquivos que já existem não são substituídos.")
            up = st.file_uploader("Backup .zip", type=["zip"], label_visibility="collapsed")
            if up is not None and st.button("Restaurar", type="primary", icon=":material/restore:"):
                try:
                    novos, pulados = restaurar_backup_zip(up)
                    st.success(f"{novos} arquivos restaurados ({pulados} já existiam).",
                               icon=":material/check_circle:")
                except Exception as exc:
                    st.error(f"Não consegui ler o backup: {exc}")


def pagina_config():
    cfg = carregar_config()
    cabecalho("Ajustes", "Configurações", "Limiares de confiança, duração das gravações e frases de cadastro.")
    st.markdown("### Microfone")
    st.caption("Se as gravações saem mudas, provavelmente o Windows está usando outro microfone. "
               "Escolha o certo aqui e faça o teste.")
    try:
        from reuniao import listar_microfones
        mics = listar_microfones()
    except Exception as exc:
        mics = []
        st.error(f"Não consegui listar os microfones: {exc}. Use o 'Gravador do navegador'.")
    if mics:
        opcoes = [None] + [i for i, _ in mics]
        nomes = {None: "Padrão do Windows"} | {i: n for i, n in mics}
        atual = cfg.get("dispositivo_entrada")
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        escolhido = c1.selectbox("Microfone usado pelo 'Microfone do PC' e pela reunião ao vivo", opcoes,
                                 index=opcoes.index(atual) if atual in opcoes else 0,
                                 format_func=lambda i: nomes[i])
        if c2.button("Usar este microfone", icon=":material/save:"):
            cfg["dispositivo_entrada"] = escolhido
            salvar_config(cfg)
            st.success(f"Microfone salvo: {nomes[escolhido]}", icon=":material/check_circle:")

    if st.button("Testar microfone (3 s)", type="primary", icon=":material/graphic_eq:"):
        ph = st.empty()
        try:
            y = countdown_e_gravar(3.0, ph)
            n = analisar_nivel(y, cfg)
            nivel = min(1.0, n["pico"] / 0.5)
            card('<div class="card-title">Resultado do teste</div>'
                 + barras_html({"volume captado": nivel}, {"volume captado": ROSA})
                 + f'<div class="muted">pico {n["pico"]:.3f} · fala detectada em {n["fracao_fala"]:.0%} do tempo</div>')
            st.audio(y, sample_rate=SAMPLE_RATE)
            if n["pico"] < 0.01:
                st.error("Nada foi captado. Escolha outro microfone acima ou confira se ele está ligado/permitido "
                         "no Windows (Configurações > Privacidade > Microfone).", icon=":material/mic_off:")
            elif eh_silencio(y, cfg):
                st.warning("Captou pouco som. Fale mais perto do microfone ou diminua a 'Sensibilidade de silêncio'.",
                           icon=":material/volume_down:")
            else:
                st.success("Microfone funcionando.", icon=":material/check_circle:")
        except Exception as exc:
            ph.empty()
            st.error(f"Microfone indisponível: {exc}")

    st.markdown("### Parâmetros")
    with st.form("cfg"):
        c1, c2 = st.columns(2, gap="large")
        cfg["duracao_gravacao"] = c1.slider("Duração de cada áudio do dataset (s)", 2.0, 5.0,
                                            float(cfg["duracao_gravacao"]), 0.5)
        cfg["duracao_bloco"] = c2.select_slider("Duração padrão do bloco da reunião (s)", [3.0, 4.0, 5.0],
                                                float(cfg["duracao_bloco"]))
        cfg["limiar_pessoa"] = c1.slider("Confiança mínima para pessoa (abaixo vira desconhecido)", 0.2, 0.9,
                                         float(cfg["limiar_pessoa"]), 0.05)
        cfg["limiar_emocao"] = c2.slider("Confiança mínima para emoção (abaixo entra em baixa confiança)", 0.2, 0.9,
                                         float(cfg["limiar_emocao"]), 0.05)
        cfg["limiar_silencio_rms"] = c1.slider("Sensibilidade de silêncio (energia RMS)", 0.001, 0.05,
                                               float(cfg["limiar_silencio_rms"]), 0.001, format="%.3f",
                                               help="Aumente se o ruído da sala estiver sendo tratado como fala.")
        frases = st.text_area("Frases de gravação (uma por linha)", "\n".join(cfg["frases"]), height=220)
        if st.form_submit_button("Salvar configurações", type="primary", icon=":material/save:"):
            cfg["frases"] = [f.strip() for f in frases.splitlines() if f.strip()] or cfg["frases"]
            salvar_config(cfg)
            st.success("Configuração salva em config.json", icon=":material/check_circle:")

    secao_backup()
    st.caption(f"Pasta do projeto neste computador: {DATASET_PESSOAS.parent}")


# ===========================================================================
def main():
    for d in (DATASET_PESSOAS, DATASET_EMOCOES, MODELOS, REUNIOES):
        d.mkdir(exist_ok=True)
    paginas = [
        st.Page(pagina_painel, title="Painel", icon=":material/space_dashboard:", url_path="painel", default=True),
        st.Page(pagina_datasets, title="Datasets", icon=":material/mic:", url_path="datasets"),
        st.Page(pagina_treino, title="Treinamento", icon=":material/model_training:", url_path="treinamento"),
        st.Page(pagina_testar, title="Testar", icon=":material/science:", url_path="testar"),
        st.Page(pagina_reuniao, title="Reunião", icon=":material/groups:", url_path="reuniao"),
        st.Page(pagina_relatorio, title="Relatório", icon=":material/insert_chart:", url_path="relatorio"),
        st.Page(pagina_config, title="Configurações", icon=":material/tune:", url_path="configuracoes"),
    ]
    st.navigation(paginas, position="top").run()
    st.markdown('<div class="muted" style="text-align:center;margin-top:40px;font-size:.78rem">'
                'O sistema estima emoção vocal a partir de características acústicas; não lê sentimentos. '
                'Grave apenas quem concordou e use os áudios somente para a atividade.</div>',
                unsafe_allow_html=True)


main()
