# MVP — Dry Bean Classification

Projeto de MVP da disciplina **Engenharia de Software para Sistemas Inteligentes**, com desenvolvimento de um modelo de **classificação multiclasse** usando o **Dry Bean Dataset** e integração do modelo a uma aplicação **full stack** simples.

---

## Sumário

- [Objetivo](#objetivo)
- [Dataset](#dataset)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Notebook no Google Colab](#notebook-no-google-colab)
- [Tecnologias utilizadas](#tecnologias-utilizadas)
- [Arquitetura da solução](#arquitetura-da-solução)
- [Como executar localmente](#como-executar-localmente)
- [Como treinar o modelo](#como-treinar-o-modelo)
- [Como executar a aplicação](#como-executar-a-aplicação)
- [Como executar os testes](#como-executar-os-testes)
- [Teste automatizado de desempenho do modelo](#teste-automatizado-de-desempenho-do-modelo)
- [Qualidade do código](#qualidade-do-código)
- [Segurança e privacidade](#segurança-e-privacidade)
- [Autor](#autor)
- [Observação final](#observação-final)

---

## Objetivo

O objetivo deste projeto é prever a **classe de um grão** com base em atributos geométricos extraídos por visão computacional. O trabalho contempla:

- construção e avaliação de modelos clássicos de machine learning com Scikit-learn;
- comparação entre **KNN**, **Árvore de Classificação**, **Gaussian Naive Bayes** e **SVM**;
- exportação do melhor modelo;
- integração do modelo a uma aplicação web com **back-end Flask** e **front-end HTML/CSS/JavaScript**;
- implementação de **testes automatizados com PyTest** para proteger o desempenho mínimo do modelo.

---

## Dataset

O projeto utiliza o **Dry Bean Dataset**, um problema nativo de classificação com:

- **13.611 instâncias**
- **16 atributos preditores numéricos**
- **7 classes**

O dataset está versionado neste repositório em formato `.xlsx` e também é carregado por **URL** no notebook do Google Colab.

---

## Estrutura do repositório

```text
mvp-dry-bean-classification/
├── data/
│   └── Dry_Bean_Dataset.xlsx
├── notebooks/
│   └── dry_bean_mvp_colab.ipynb
├── models/
│   ├── model.joblib
│   ├── metrics.json
│   └── metadata.json
├── src/
│   ├── app.py
│   ├── modeling.py
│   ├── predictor.py
│   ├── schema.py
│   ├── training.py
│   ├── static/
│   │   ├── app.js
│   │   └── style.css
│   └── templates/
│       └── index.html
├── tests/
│   ├── conftest.py
│   ├── test_app.py
│   ├── test_model_performance.py
│   ├── test_modeling.py
│   └── test_predictor.py
├── .flake8
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Notebook no Google Colab

O notebook com o processo completo de criação do modelo está em:

- `notebooks/dry_bean_mvp_colab.ipynb`

### Links importantes

- **Repositório GitHub:** `https://github.com/rafaelmgsilva/mvp-dry-bean-classification`
- **Link do Colab:** `https://colab.research.google.com/github/rafaelmgsilva/mvp-dry-bean-classification/blob/main/dry_bean_mvp_colab.ipynb`

### O notebook inclui

- carga do dataset por URL;
- inspeção e validação inicial dos dados;
- análise exploratória;
- separação entre treino e teste com **holdout estratificado**;
- baseline com `DummyClassifier`;
- pipelines com transformação de dados;
- otimização de hiperparâmetros com validação cruzada;
- comparação entre **KNN**, **Árvore de Classificação**, **Gaussian Naive Bayes** e **SVM**;
- análise de overfitting e underfitting;
- exportação do modelo final e metadados;
- reflexão sobre segurança e privacidade.

---

## Tecnologias utilizadas

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

### Aplicação full stack
- Flask
- HTML
- CSS
- JavaScript

### Qualidade e testes
- PyTest
- Flake8
- Black
- isort

---

## Arquitetura da solução

A solução utiliza o modelo de implantação com **modelo embarcado no back-end**.

Fluxo da aplicação:

1. o modelo treinado é exportado para arquivo;
2. a aplicação Flask carrega esse arquivo no back-end;
3. o usuário informa os atributos do grão no front-end;
4. o back-end realiza a predição;
5. a aplicação retorna a **classe prevista**, a **probabilidade estimada** e eventuais **warnings** quando os valores informados estão fora da faixa observada no treinamento.

---

## Como executar localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/rafaelmgsilva/mvp-dry-bean-classification.git
cd mvp-dry-bean-classification
```

### 2. Criar e ativar o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

## Como treinar o modelo

Depois de configurar o ambiente, execute o processo de treinamento para gerar os artefatos do modelo:

```bash
python - <<'PY'
from src.training import train_and_save_best_model

metrics = train_and_save_best_model(
    dataset_path="data/Dry_Bean_Dataset.xlsx",
    model_output_path="models/model.joblib",
    metrics_output_path="models/metrics.json",
    metadata_output_path="models/metadata.json",
)
print(metrics)
PY
```

Arquivos gerados:

- `models/model.joblib`
- `models/metrics.json`
- `models/metadata.json`

---

## Como executar a aplicação

Depois de treinar o modelo, execute:

```bash
flask --app src.app run --debug
```

A aplicação ficará disponível em:

```text
http://127.0.0.1:5000
```

---

## Como executar os testes

Para rodar todos os testes:

```bash
python -m pytest
```

---

## Teste automatizado de desempenho do modelo

O projeto inclui testes automatizados para assegurar que o modelo atenda aos requisitos mínimos de desempenho definidos no trabalho.

Atualmente, os thresholds adotados são:

- `accuracy >= 0.90`
- `macro_f1 >= 0.92`

Esses testes ajudam a evitar a substituição do modelo por uma versão com desempenho inferior.

---

## Qualidade do código

O projeto também utiliza ferramentas de análise estática e formatação:

```bash
black src tests
isort src tests
flake8 src tests
```

---

## Segurança e privacidade

Embora o Dry Bean Dataset não contenha dados pessoais, o projeto considera boas práticas de desenvolvimento de software seguro, como:

- validação rigorosa das entradas da aplicação;
- verificação de faixas observadas no treinamento;
- testes automatizados para reduzir regressões;
- organização dos artefatos do modelo;
- possibilidade de uso de comunicação segura em ambiente real.

Em cenários reais com dados sensíveis, seriam relevantes técnicas adicionais como:

- anonimização;
- pseudoanonimização;
- controle de acesso;
- proteção da integridade dos artefatos;
- uso de HTTPS/TLS.


---

## Autor

**Rafael Medeiros**

---

## Observação final

Este repositório foi estruturado para atender aos requisitos do MVP, incluindo:

- notebook executável no Colab;
- aplicação full stack simples;
- modelo embarcado no back-end;
- teste automatizado de desempenho;
- arquivos auxiliares necessários para execução.