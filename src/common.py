"""Funcoes e configuracoes compartilhadas da Sala de Reuniao Inteligente.

Tudo que mais de um programa usa fica aqui:
- caminhos das pastas do projeto;
- configuracao (config.json);
- extracao de features do audio (o que o Random Forest recebe);
- deteccao de silencio;
- classificacao de um bloco de audio com os dois modelos.
"""

from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATASET_PESSOAS = ROOT / "dataset_pessoas"
DATASET_EMOCOES = ROOT / "dataset_emocoes"
MODELOS = ROOT / "modelos"
REUNIOES = ROOT / "reunioes"
CONFIG_PATH = ROOT / "config.json"

MODELO_PESSOAS = MODELOS / "modelo_pessoas.pkl"
MODELO_EMOCOES = MODELOS / "modelo_emocoes.pkl"

SAMPLE_RATE = 16000
N_MFCC = 13
EMOCOES = ["alegre", "neutro", "triste", "irritado"]

# Rotulos especiais usados no CSV (nunca sao classes de treino)
DESCONHECIDO = "desconhecido"
SILENCIO = "silencio"
INCERTO = "incerto"
ROTULOS_ESPECIAIS = {DESCONHECIDO, SILENCIO, INCERTO}


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "duracao_gravacao": 3.5,      # segundos de cada audio do dataset
    "duracao_bloco": 3.0,         # segundos de cada bloco da reuniao
    "limiar_pessoa": 0.45,        # abaixo disso -> desconhecido
    "limiar_emocao": 0.40,        # abaixo disso -> trecho de baixa confianca
    "limiar_silencio_rms": 0.005,  # energia minima para considerar que ha fala
    "dispositivo_entrada": None,    # microfone escolhido (None = padrao do sistema)
    "minimo_por_pessoa": 20,
    "minimo_por_emocao": 25,
    "frases": [
        "Hoje vamos discutir o projeto da reuniao.",
        "Precisamos organizar melhor as tarefas.",
        "Eu concordo com a proposta apresentada.",
        "Vamos testar o sistema antes da apresentacao.",
        "O grupo precisa analisar os resultados.",
        "A reuniao de hoje sera usada para validar o projeto.",
        "Vamos verificar os dados antes do treinamento.",
        "Cada participante deve falar de forma natural.",
        "O modelo precisa funcionar com frases novas.",
        "Precisamos melhorar a qualidade das gravacoes.",
        "Acho que devemos dividir o trabalho em etapas.",
        "Quem vai apresentar a primeira parte do projeto?",
        "Eu posso revisar o relatorio amanha cedo.",
        "Esse resultado ficou melhor do que o esperado.",
        "Vamos marcar outra reuniao para a semana que vem.",
    ],
}


def carregar_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def salvar_config(cfg):
    save_json(CONFIG_PATH, cfg)


# ---------------------------------------------------------------------------
# Utilidades de arquivos
# ---------------------------------------------------------------------------
def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def limpar_nome(texto):
    """'Rogério Morandi' -> 'rogerio_morandi' (regra: minusculo, sem acento, sem espaco)."""
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    texto = re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")
    return texto


def listar_classes(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return []
    return sorted(p.name for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def contar_dataset(dataset_dir):
    return {c: len(list((Path(dataset_dir) / c).glob("*.wav"))) for c in listar_classes(dataset_dir)}


def find_wavs(dataset_dir):
    items = []
    for classe in listar_classes(dataset_dir):
        for wav in sorted((Path(dataset_dir) / classe).glob("*.wav")):
            items.append((wav, classe))
    return items


def proximo_arquivo(pasta, prefixo):
    """Proximo nome livre: prefixo_001.wav, prefixo_002.wav..."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    n = 1
    while (pasta / f"{prefixo}_{n:03d}.wav").exists():
        n += 1
    return pasta / f"{prefixo}_{n:03d}.wav"


def salvar_wav(path, y, sr=SAMPLE_RATE):
    sf.write(str(path), np.asarray(y, dtype=np.float32).reshape(-1), sr, subtype="PCM_16")


def carregar_audio(fonte, sr=SAMPLE_RATE):
    """Aceita caminho, bytes/arquivo (upload) ou array. Retorna mono em 16 kHz."""
    if isinstance(fonte, np.ndarray):
        return fonte.astype(np.float32).reshape(-1)
    if isinstance(fonte, (str, Path)):
        y, _ = librosa.load(str(fonte), sr=sr, mono=True)
        return y.astype(np.float32)
    # arquivo em memoria (ex.: upload do Streamlit)
    data, file_sr = sf.read(fonte, dtype="float32", always_2d=True)
    y = data.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Qualidade do audio / silencio
# ---------------------------------------------------------------------------
def analisar_nivel(y, cfg=None):
    """Retorna rms global, pico e fracao de quadros com fala (acima do limiar)."""
    cfg = cfg or carregar_config()
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return {"rms": 0.0, "pico": 0.0, "fracao_fala": 0.0}
    rms_frames = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    return {
        "rms": float(np.sqrt(np.mean(y ** 2))),
        "pico": float(np.max(np.abs(y))),
        "fracao_fala": float(np.mean(rms_frames > cfg["limiar_silencio_rms"])),
    }


def eh_silencio(y, cfg=None):
    """Bloco sem fala suficiente (menos de 15% dos quadros com energia)."""
    return analisar_nivel(y, cfg)["fracao_fala"] < 0.15


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
FEATURE_NAMES = (
    [f"mfcc_{i+1}_media" for i in range(N_MFCC)]
    + [f"mfcc_{i+1}_desvio" for i in range(N_MFCC)]
    + [f"delta_mfcc_{i+1}_desvio" for i in range(N_MFCC)]
    + ["rms_media", "rms_desvio",
       "pitch_media", "pitch_desvio", "fracao_vozeada",
       "zcr_media", "zcr_desvio",
       "centroide_media", "centroide_desvio",
       "bandwidth_media", "bandwidth_desvio",
       "rolloff_media", "rolloff_desvio",
       "flatness_media"]
    + [f"contraste_banda_{i+1}" for i in range(7)]
)


def familia_feature(nome):
    """Agrupa features para o grafico de importancia."""
    if nome.startswith("delta_mfcc"):
        return "Delta MFCC (ritmo)"
    if nome.startswith("mfcc"):
        return "MFCC (timbre)"
    if nome.startswith("rms"):
        return "Energia"
    if nome.startswith("pitch") or nome == "fracao_vozeada":
        return "Pitch"
    if nome.startswith("zcr"):
        return "ZCR"
    return "Espectro"


def extract_features(audio_or_path, sr=SAMPLE_RATE):
    """Transforma um audio em um vetor numerico fixo (X do Random Forest)."""
    y = carregar_audio(audio_or_path, sr)

    if y.size == 0:
        raise ValueError("Audio vazio.")

    y, _ = librosa.effects.trim(y, top_db=35)
    if y.size < int(0.5 * sr):
        y = np.pad(y, (0, int(0.5 * sr) - y.size))

    # Normaliza o volume para o modelo nao depender da distancia do microfone.
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=100.0, n_bands=6)

    # Pitch (YIN) somente nos quadros com energia (evita "pitch" de silencio).
    try:
        f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, frame_length=1024, hop_length=256)
        rms_f0 = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][: f0.size]
        vozeado = f0[: rms_f0.size][rms_f0 > 0.1 * (rms_f0.max() + 1e-9)]
        vozeado = vozeado[np.isfinite(vozeado)]
        f0_mean = float(np.mean(vozeado)) if vozeado.size else 0.0
        f0_std = float(np.std(vozeado)) if vozeado.size else 0.0
        frac_voz = float(vozeado.size / max(1, f0.size))
    except Exception:
        f0_mean, f0_std, frac_voz = 0.0, 0.0, 0.0

    features = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1), delta.std(axis=1),
        [rms.mean(), rms.std(),
         f0_mean, f0_std, frac_voz,
         zcr.mean(), zcr.std(),
         centroid.mean(), centroid.std(),
         bandwidth.mean(), bandwidth.std(),
         rolloff.mean(), rolloff.std(),
         flatness.mean()],
        contrast.mean(axis=1),
    ]).astype(np.float32)

    return np.nan_to_num(features)


# ---------------------------------------------------------------------------
# Classificacao (usada no teste, na reuniao ao vivo e na interface)
# ---------------------------------------------------------------------------
def modelos_existem():
    return MODELO_PESSOAS.exists() and MODELO_EMOCOES.exists()


def carregar_modelos():
    import joblib
    if not modelos_existem():
        raise FileNotFoundError("Treine os dois modelos antes (03 e 04).")
    return {
        "pessoas": joblib.load(MODELO_PESSOAS),
        "emocoes": joblib.load(MODELO_EMOCOES),
    }


def classificar(y, modelos, cfg=None):
    """Classifica UM bloco de audio. Retorna um dicionario com tudo que foi decidido."""
    cfg = cfg or carregar_config()
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if eh_silencio(y, cfg):
        return {"status": SILENCIO, "pessoa": SILENCIO, "conf_pessoa": 0.0,
                "emocao": SILENCIO, "conf_emocao": 0.0,
                "probs_pessoa": {}, "probs_emocao": {}}

    feat = extract_features(y)
    X = feat.reshape(1, -1)

    mp, me = modelos["pessoas"], modelos["emocoes"]
    probs_p = mp.predict_proba(X)[0]
    probs_e = me.predict_proba(X)[0]
    idx_p, idx_e = int(np.argmax(probs_p)), int(np.argmax(probs_e))

    pessoa = str(mp.classes_[idx_p])
    conf_p = float(probs_p[idx_p])
    emocao = str(me.classes_[idx_e])
    conf_e = float(probs_e[idx_e])

    # Pessoa desconhecida: o modelo nao tem confianca suficiente em nenhuma classe.
    # (Se o grupo gravar uma pasta dataset_pessoas/desconhecido com vozes de fora,
    #  o proprio Random Forest aprende essa classe - ver README.)
    status, motivo = "ok", ""
    if pessoa == DESCONHECIDO or conf_p < cfg["limiar_pessoa"]:
        status = DESCONHECIDO
        motivo = (f"confianca {conf_p:.0%} abaixo do limiar {cfg['limiar_pessoa']:.0%}"
                  if pessoa != DESCONHECIDO else "classificado na classe 'desconhecido'")

    return {
        "status": status,
        "pessoa": DESCONHECIDO if status == DESCONHECIDO else pessoa,
        "pessoa_mais_provavel": pessoa,
        "conf_pessoa": conf_p,
        "emocao": emocao,
        "conf_emocao": conf_e,
        "probs_pessoa": {str(c): float(p) for c, p in zip(mp.classes_, probs_p)},
        "probs_emocao": {str(c): float(p) for c, p in zip(me.classes_, probs_e)},
        "motivo": motivo,
    }


def format_time(seconds):
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def parse_time(txt):
    m, s = str(txt).split(":")
    return int(m) * 60 + int(s)
