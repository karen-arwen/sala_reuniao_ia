SALA DE REUNIAO INTELIGENTE
UniMAX - Indaiatuba
Prof. Rogerio Morandi

OBJETIVO
Sistema em Python que:
1. grava e organiza datasets de voz;
2. treina um Random Forest para identificar a pessoa;
3. treina outro Random Forest para classificar a emocao vocal estimada;
4. analisa blocos curtos de uma reuniao;
5. registra previsoes e confiancas em CSV;
6. gera relatorio, graficos e PDF.

IMPORTANTE
- O projeto exige dados reais gravados pelo grupo.
- Nao usa palavra-chave, regras para escolher o nome da pessoa ou porcentagens inventadas.
- A emocao e uma classificacao vocal estimada pelo modelo, nao um diagnostico do sentimento real.
- Grave somente participantes que concordarem e use os audios apenas para a atividade.

ESTRUTURA
sala_reuniao_ia/
  src/
    01_gravar_pessoas.py      grava dataset_pessoas
    02_gravar_emocoes.py      grava dataset_emocoes
    03_treinar_pessoas.py     treina modelo_pessoas.pkl
    04_treinar_emocoes.py     treina modelo_emocoes.pkl
    05_testar_modelos.py      metricas + teste ao vivo / WAV
    06_analisar_reuniao.py    reuniao em blocos -> registros.csv
    07_gerar_relatorio.py     relatorio.txt + graficos/ + relatorio.pdf
    todos_os_programas.py     menu com todos os programas
    app.py                    INTERFACE INTERATIVA (navegador)
    common.py                 features, caminhos, configuracao, classificacao
    treino.py                 treino + validacao cruzada + matriz de confusao
    reuniao.py                gravacao continua em blocos + CSV
    relatorio.py              calculos e graficos do relatorio
  dataset_pessoas/            uma subpasta por pessoa (rogerio/, ana/ ...)
  dataset_emocoes/            alegre/ neutro/ triste/ irritado/
  modelos/                    .pkl, metadados, matrizes de confusao
  reunioes/                   reuniao_01/, reuniao_02/ ...
  config.json                 (criado pela interface) limiares, duracoes e frases
  requirements.txt
  README.txt

INSTALACAO
Python 3.10+ recomendado. No terminal, dentro da pasta sala_reuniao_ia:
    python -m venv .venv
    .venv\Scripts\activate            (Windows)
    python -m pip install --upgrade pip
    pip install -r requirements.txt

JEITO MAIS FACIL (WINDOWS): dois cliques em iniciar.bat
    Ele cria o .venv, instala as bibliotecas e abre a interface no navegador.

INTERFACE INTERATIVA (manual)
    streamlit run src/app.py
Abre no navegador com as abas:
    Painel         checklist da entrega calculado automaticamente
    Datasets       cadastrar pessoas, gravar, ouvir e apagar audios ruins, contagem por classe
    Treinamento    treinar os dois modelos, acuracia, matriz de confusao, importancia das features
    Testar         gravar uma frase e ver pessoa + emocao + probabilidade de cada classe
    Reuniao        reuniao AO VIVO pelo microfone (ou analisar um WAV gravado)
    Relatorio      graficos interativos, trechos de baixa confianca (com audio), download CSV/TXT/PDF
    Configuracoes  limiares de confianca, duracao do bloco, frases, teste de ruido da sala

COMANDOS (o que o README precisa responder)
Qual comando grava pessoas?
    python src/01_gravar_pessoas.py              (todas as pessoas, meta 30 audios cada)
    python src/01_gravar_pessoas.py --pessoa ana --meta 20
Qual comando grava emocoes?
    python src/02_gravar_emocoes.py --pessoa ana --por-emocao 7
    (cada pessoa do grupo roda uma vez; 5 pessoas x 7 = 35 audios por emocao)
Qual comando treina o modelo de pessoas?
    python src/03_treinar_pessoas.py
Qual comando treina o modelo de emocoes?
    python src/04_treinar_emocoes.py
Qual comando testa os modelos?
    python src/05_testar_modelos.py              (ENTER grava pelo microfone)
    python src/05_testar_modelos.py arquivo.wav
Qual comando analisa a reuniao?
    python src/06_analisar_reuniao.py            (nova reuniao ao vivo, Ctrl+C encerra)
    python src/06_analisar_reuniao.py --reuniao reuniao_01 --bloco 4
    python src/06_analisar_reuniao.py --arquivo gravacao.wav
Qual comando gera o relatorio?
    python src/07_gerar_relatorio.py             (reuniao mais recente)
    python src/07_gerar_relatorio.py --reuniao reuniao_01
Menu com tudo:
    python src/todos_os_programas.py

QUANTIDADES MINIMAS
Pessoas:  5 pessoas, minimo 20 audios cada (ideal 30), 3-4 s por audio.
Emocoes:  4 emocoes, minimo 25 audios cada, varias pessoas gravando.
Os programas de treino imprimem a contagem por pasta e avisam se estiver desbalanceado.

FEATURES (o que o Random Forest recebe)
- MFCC media e desvio (13 + 13)  -> timbre da voz
- delta-MFCC desvio (13)          -> ritmo / variacao
- energia RMS media e desvio
- pitch (YIN) media, desvio e fracao de trechos vozeados
- ZCR media e desvio
- centroide, bandwidth e rolloff espectrais (media e desvio)
- flatness e contraste espectral (7 bandas)
Antes da extracao: remove silencio das pontas e normaliza o volume.

MODELO E VALIDACAO
RandomForestClassifier (300 arvores, class_weight="balanced").
Validacao: StratifiedKFold (5 folds). Cada audio e previsto por um modelo que
NAO o viu no treino, entao a acuracia e a matriz de confusao usam todos os
audios de forma honesta. Depois o modelo final e treinado com tudo.
Saidas: modelos/metadados_*.json, matriz_confusao_*.png, importancia_features_*.png

REUNIAO
- Blocos de 3 s (ou 4/5 s), gravacao continua sem buracos entre blocos.
- Uma pessoa fala por vez.
- Confianca da pessoa < 45%  -> "desconhecido".
- Bloco sem fala              -> "silencio" (registrado, nunca apagado).
- Erro ao classificar         -> "incerto".
- Pessoa desconhecida (bonus): alem do limiar de confianca, o grupo pode criar
  dataset_pessoas/desconhecido/ com vozes de pessoas de fora do grupo. O Random
  Forest aprende essa classe e passa a reconhecer quem nao e cadastrado.

CSV OBRIGATORIO (reunioes/reuniao_XX/registros.csv)
inicio,fim,pessoa,conf_pessoa,emocao,conf_emocao,arquivo

RELATORIO (reunioes/reuniao_XX/)
relatorio.txt, relatorio.pdf e graficos/:
    tempo_fala_por_participante.png, emocoes_por_participante.png,
    distribuicao_geral_emocoes.png, linha_do_tempo.png, baixa_confianca.png,
    matriz_confusao_pessoas.png, matriz_confusao_emocoes.png

EVIDENCIAS DE TESTE
Os testes feitos na aba "Testar" podem ser registrados em modelos/testes_ao_vivo.csv
(quem falou de verdade x o que o modelo previu).

NAO INVENTAR RESULTADOS
As metricas e percentuais apresentados na entrega vem dos dados realmente
gravados e dos testes realmente executados.
