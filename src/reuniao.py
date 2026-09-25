"""Analise de reuniao em blocos (usado pelo 06 e pela interface).

Fluxo obrigatorio de cada bloco:
    gravar bloco -> classificar pessoa -> classificar emocao -> salvar no CSV
"""

from pathlib import Path
import csv
import queue

import numpy as np

from common import (
    REUNIOES, SAMPLE_RATE, SILENCIO, INCERTO,
    classificar, carregar_config, salvar_wav, format_time, parse_time, carregar_audio,
)

COLUNAS = ["inicio", "fim", "pessoa", "conf_pessoa", "emocao", "conf_emocao", "arquivo"]


def listar_reunioes():
    if not REUNIOES.exists():
        return []
    return sorted(p.name for p in REUNIOES.iterdir() if p.is_dir() and p.name.startswith("reuniao_"))


def nova_reuniao():
    """Cria reuniao_01, reuniao_02... (proximo numero livre)."""
    REUNIOES.mkdir(exist_ok=True)
    n = 1
    while (REUNIOES / f"reuniao_{n:02d}").exists():
        n += 1
    pasta = REUNIOES / f"reuniao_{n:02d}"
    (pasta / "blocos_audio").mkdir(parents=True)
    return pasta


class SessaoReuniao:
    """Mantem o CSV aberto e registra cada bloco analisado."""

    def __init__(self, pasta, modelos, cfg=None, duracao_bloco=None):
        self.pasta = Path(pasta)
        self.cfg = cfg or carregar_config()
        self.modelos = modelos
        self.dur = float(duracao_bloco or self.cfg["duracao_bloco"])
        self.audio_dir = self.pasta / "blocos_audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.pasta / "registros.csv"

        # Continua a linha do tempo se a reuniao ja tiver registros.
        self.tempo = 0.0
        if self.csv_path.exists():
            with self.csv_path.open(encoding="utf-8") as f:
                linhas = list(csv.DictReader(f))
            if linhas:
                self.tempo = float(parse_time(linhas[-1]["fim"]))

        novo = not self.csv_path.exists()
        self._f = self.csv_path.open("a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if novo:
            self._w.writerow(COLUNAS)
            self._f.flush()

        self.bloco_num = 1
        while (self.audio_dir / f"bloco_{self.bloco_num:03d}.wav").exists():
            self.bloco_num += 1

    def processar_bloco(self, y):
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        dur = len(y) / SAMPLE_RATE
        inicio, fim = self.tempo, self.tempo + dur
        path = self.audio_dir / f"bloco_{self.bloco_num:03d}.wav"
        salvar_wav(path, y)

        try:
            r = classificar(y, self.modelos, self.cfg)
        except Exception as exc:  # nao apaga o bloco: registra como incerto
            r = {"status": INCERTO, "pessoa": INCERTO, "conf_pessoa": 0.0,
                 "emocao": INCERTO, "conf_emocao": 0.0, "probs_pessoa": {},
                 "probs_emocao": {}, "erro": str(exc)}

        linha = [format_time(inicio), format_time(fim), r["pessoa"], f"{r['conf_pessoa']:.4f}",
                 r["emocao"], f"{r['conf_emocao']:.4f}", path.name]
        self._w.writerow(linha)
        self._f.flush()

        r.update({"inicio": linha[0], "fim": linha[1], "arquivo": path.name,
                  "inicio_s": inicio, "fim_s": fim, "bloco": self.bloco_num})
        self.tempo = fim
        self.bloco_num += 1
        return r

    def fechar(self):
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()


class CapturaContinua:
    """Grava o microfone sem buracos entre os blocos (InputStream + fila).

    with CapturaContinua(3.0) as cap:
        while True:
            bloco = cap.ler_bloco()
    """

    def __init__(self, duracao_bloco, sr=SAMPLE_RATE, device=None):
        self.n = int(duracao_bloco * sr)
        self.sr = sr
        self.device = device if device is not None else carregar_config().get("dispositivo_entrada")
        self.fila = queue.Queue()
        self.buffer = np.zeros(0, dtype=np.float32)
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        self.fila.put(indata[:, 0].copy())

    def __enter__(self):
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                     callback=self._callback, device=self.device)
        self.stream.start()
        return self

    def ler_bloco(self, timeout=10):
        while self.buffer.size < self.n:
            self.buffer = np.concatenate([self.buffer, self.fila.get(timeout=timeout)])
        bloco, self.buffer = self.buffer[: self.n], self.buffer[self.n:]
        return bloco

    def nivel_atual(self):
        return float(np.sqrt(np.mean(self.buffer ** 2))) if self.buffer.size else 0.0

    def __exit__(self, *exc):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


def gravar_trecho(duracao, sr=SAMPLE_RATE, device=None):
    """Gravacao simples de duracao fixa (datasets e teste)."""
    import sounddevice as sd
    if device is None:
        device = carregar_config().get("dispositivo_entrada")
    audio = sd.rec(int(duracao * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio.reshape(-1)


def listar_microfones():
    """[(indice, nome)] dos dispositivos de entrada disponiveis."""
    import sounddevice as sd
    vistos, lista = set(), []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and d["name"] not in vistos:
            vistos.add(d["name"])
            lista.append((i, d["name"]))
    return lista


def blocos_de_arquivo(fonte, duracao_bloco):
    """Divide um audio gravado (ex.: reuniao.wav) em blocos de N segundos."""
    y = carregar_audio(fonte)
    n = int(duracao_bloco * SAMPLE_RATE)
    for i in range(0, len(y), n):
        bloco = y[i:i + n]
        if len(bloco) >= int(0.5 * SAMPLE_RATE):  # ignora sobra menor que 0,5 s
            yield bloco


def analisar_arquivo(fonte, pasta, modelos, cfg=None, duracao_bloco=None, callback=None):
    cfg = cfg or carregar_config()
    dur = duracao_bloco or cfg["duracao_bloco"]
    resultados = []
    with SessaoReuniao(pasta, modelos, cfg, dur) as s:
        for bloco in blocos_de_arquivo(fonte, dur):
            r = s.processar_bloco(bloco)
            resultados.append(r)
            if callback:
                callback(r)
    return resultados


def descrever(r):
    if r["pessoa"] == SILENCIO:
        return f"[{r['inicio']}-{r['fim']}] (silencio)"
    return (f"[{r['inicio']}-{r['fim']}] Pessoa: {r['pessoa']} ({r['conf_pessoa']:.0%}) | "
            f"Emocao vocal estimada: {r['emocao']} ({r['conf_emocao']:.0%})")
