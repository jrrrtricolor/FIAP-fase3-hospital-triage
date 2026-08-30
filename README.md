**Repositório público:**
[github.com/jrrrtricolor/FIAP-fase3-hospital-triage](https://github.com/jrrrtricolor/FIAP-fase3-hospital-triage)

**Versão candidata à avaliação:**
[`develop`](https://github.com/jrrrtricolor/FIAP-fase3-hospital-triage/tree/develop) —
a tag imutável será informada após as integrações finais

**Vídeo STAR:** **PENDENTE — falta fornecer a URL pública**

# Hospital Triage — FIAP Tech Challenge Fase 3

[![CI - Dados e treinamento](https://github.com/jrrrtricolor/FIAP-fase3-hospital-triage/actions/workflows/ml-pipeline.yml/badge.svg?branch=develop)](https://github.com/jrrrtricolor/FIAP-fase3-hospital-triage/actions/workflows/ml-pipeline.yml)

Sistema acadêmico de apoio à triagem hospitalar que recebe um texto clínico em
inglês e retorna uma das classes `normal`, `atencao` ou `urgente`. A solução
combina NLP, FastAPI, ONNX Runtime, Docker, GitHub Actions e MLflow. Airflow e a
stack Prometheus/Grafana estão em integração paralela pelo grupo.

> [!IMPORTANT]
> **Vídeo STAR (até cinco minutos): PENDENTE — inserir aqui a URL pública e
> testá-la em uma janela anônima antes da entrega.** Um placeholder não atende à
> rubrica; esta versão do README não deve ser marcada como release final sem o
> link válido.

## Informações da entrega

| Item | Referência |
|---|---|
| Repositório | [jrrrtricolor/FIAP-fase3-hospital-triage](https://github.com/jrrrtricolor/FIAP-fase3-hospital-triage) |
| Branch de integração | [`develop`](https://github.com/jrrrtricolor/FIAP-fase3-hospital-triage/tree/develop) |
| Versão avaliada | Candidata em `develop`; a tag final deve ser criada após Airflow, monitoramento e vídeo |
| CI/CD | [Workflow e histórico de execuções](https://github.com/jrrrtricolor/FIAP-fase3-hospital-triage/actions/workflows/ml-pipeline.yml) |
| Model Card | [docs/model_card.md](docs/model_card.md) |
| Benchmark ONNX | [model/onnx_benchmark.json](model/onnx_benchmark.json) |
| Comparação dos modelos | [model/mlflow_model_comparison.json](model/mlflow_model_comparison.json) |
| Vídeo STAR | **Pendente de URL pública** |
| API pública AWS | Bônus não implementado |

## Roteiro rápido para o avaliador

Com Docker 24+ instalado, estes comandos validam o principal fluxo de
inferência sem baixar o dataset nem configurar o MLflow:

```bash
git clone https://github.com/jrrrtricolor/FIAP-fase3-hospital-triage.git
cd FIAP-fase3-hospital-triage
git switch develop
docker build -t hospital-triage:local .
docker run --rm --name hospital-triage -p 8000:8000 hospital-triage:local
```

Em outro terminal:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/ready
curl --fail \
  --header 'Content-Type: application/json' \
  --data '{"clinical_text":"Severe chest pain and shortness of breath."}' \
  http://localhost:8000/predict
```

O contrato OpenAPI fica disponível em
[`http://localhost:8000/docs`](http://localhost:8000/docs). O container carrega
uma única vez o artefato ONNX versionado em
`model/hospital_triage_model.onnx`; ele não depende de um banco MLflow dentro
da imagem.

Para encerrar, use `Ctrl+C` no terminal do container. Caso tenha iniciado em
segundo plano, execute:

```bash
docker stop hospital-triage
```

## Situação dos requisitos oficiais

| Critério | Peso | Estado atual | Evidência |
|---|---:|---|---|
| Modelagem e otimização | 20% | Atendido | [Treino](src/hospital_triage/training.py), [comparação de modelos](model/mlflow_model_comparison.json), [benchmark ONNX](model/onnx_benchmark.json) e [Model Card](docs/model_card.md) |
| CI/CD | 15% | Atendido | [Workflow](.github/workflows/ml-pipeline.yml) com lint, testes, dados, treino e build Docker |
| Airflow | 15% | Em integração | A DAG precisa ser versionada em `airflow/dags/` e executada com sucesso |
| Monitoramento | 20% | Em integração | A API, Compose, Prometheus, Grafana e dashboard precisam ser versionados juntos |
| Documentação | 15% | Parcial | README e Model Card presentes; vídeo, Airflow e monitoramento ainda precisam de links/evidências finais |
| Vídeo STAR | 15% | Não verificável | URL pública ainda não fornecida |

Esta tabela separa o que já está versionado do que ainda está em integração;
trabalho apenas anunciado não é apresentado como evidência concluída.

## Problema e uso pretendido

O tempo de priorização é relevante em um departamento de emergência. Este
projeto demonstra como um classificador textual leve pode apoiar a organização
da fila ao estimar uma classe de urgência a partir de informações disponíveis
na triagem.

A aplicação é um **demonstrador acadêmico de apoio à decisão**. Ela não realiza
diagnóstico, não define conduta clínica e não substitui profissionais de saúde.
O modelo não foi validado prospectivamente nem aprovado para uso assistencial.
Em especial, um falso negativo de `urgente` pode atrasar atendimento e deve ser
tratado como risco crítico.

## Arquitetura

Foi adotado um monólito modular: a API de inferência permanece pequena e sem
estado, enquanto dados, treino e retreino são processos batch separados. Essa
separação evita incluir Airflow na imagem de produção.

```mermaid
flowchart LR
    CDC[CDC / NHAMCS-ED 2021] --> Download[Download + checksum]
    Download --> Prepare[Preparação e validação]
    Prepare --> DB[(training_data.db)]
    Airflow[Airflow batch] --> Download
    Airflow --> Prepare
    Airflow --> Train[Treino + avaliação]
    DB --> Train
    Train --> MLflow[(MLflow Registry)]
    Train --> ONNX[Modelo ONNX imutável]
    ONNX --> API[FastAPI real-time]
    Client[Cliente] --> API
    API --> Metrics[/metrics]
    Metrics --> Prometheus
    Prometheus --> Grafana
    GitHub[GitHub Actions] --> Train
    GitHub --> Docker[Build Docker]
```

Airflow, `/metrics`, Prometheus e Grafana aparecem no desenho como arquitetura
obrigatória da entrega, mas continuam marcados como integração em andamento
até que os respectivos arquivos e testes estejam no repositório.

### Decisão de nuvem

A estratégia escolhida para uma eventual publicação é **Amazon ECR + AWS App
Runner**:

- a inferência é real-time porque a triagem precisa responder durante a
  interação do profissional, e não em um lote posterior;
- o App Runner oferece HTTPS, health check e escalabilidade com menor carga
  operacional para o demonstrador;
- cada promoção de modelo produz uma imagem imutável no ECR;
- treino e retreino continuam batch, orquestrados pelo Airflow, e não fazem
  parte do processo da API;
- textos clínicos e credenciais não devem ser gravados na imagem, em labels de
  métricas ou em logs.

O deploy AWS é bônus e ainda não está implementado. Uma implantação hospitalar
real exigiria autenticação, rede privada, auditoria, governança clínica e
controles de privacidade fora do escopo acadêmico.

## Dados e construção do texto clínico

O projeto usa o **National Hospital Ambulatory Medical Care Survey — Emergency
Department 2021 (NHAMCS-ED 2021)**, publicado pelo NCHS/CDC:

- [página oficial do NHAMCS 2021](https://www.cdc.gov/nchs/nhamcs/documentation/about-the-data-2021.html);
- [documentação técnica](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/doc21-ed-508.pdf);
- [arquivo Stata oficial](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/stata/ed2021-stata.zip).

O download é público e o uso deve respeitar o NCHS Data User Agreement. Não é
permitida tentativa de reidentificação de pacientes ou instituições. Arquivos
brutos e bancos preparados permanecem fora do Git.

O NHAMCS não fornece narrativa clínica livre. O pipeline materializa
deterministicamente `clinical_text`, em inglês, com descrições oficiais dos
motivos da visita (`RFV1` a `RFV5`), idade, sexo, dor e sinais vitais disponíveis
na triagem. Diagnósticos, exames, medicamentos e disposição não entram como
features.

O target vem de `IMMEDR`, a prioridade registrada na triagem:

| IMMEDR | Classe do projeto |
|---|---|
| 1 — Immediate; 2 — Emergent | `urgente` |
| 3 — Urgent | `atencao` |
| 4 — Semi-urgent; 5 — Nonurgent | `normal` |

Depois dos filtros há 10.495 registros e 10.492 textos únicos. A divisão usa
`StratifiedGroupKFold`, seed 42 e agrupamento pelo hash do texto para impedir o
mesmo texto em mais de um split:

| Split | Registros |
|---|---:|
| Treino | 7.350 |
| Validação | 1.573 |
| Teste reservado | 1.572 |

Checksums, contagens esperadas, sentinelas e regras de preparação estão em
[data/download_dataset.py](data/download_dataset.py) e
[src/hospital_triage/data_preparation.py](src/hospital_triage/data_preparation.py).

## Modelagem e resultados

O baseline selecionado é um pipeline Scikit-Learn com TF-IDF de unigramas e
bigramas, até 10.000 features, seguido de regressão logística balanceada. Ele
foi comparado com um classificador textual PyTorch leve na mesma divisão. O
baseline venceu por macro F1 (`0,5582` contra `0,5349`) e possui operação mais
simples.

Resultados no split de validação:

| Métrica | Resultado |
|---|---:|
| Acurácia | 0,5709 |
| Macro precision | 0,5510 |
| Macro recall | 0,5833 |
| Macro F1 | 0,5582 |
| Weighted F1 | 0,5744 |
| Recall de `urgente` | 0,5780 |

O modelo foi convertido para ONNX Runtime com concordância de predições igual
a 100%. O benchmark compara os dois formatos sobre os mesmos 1.573 registros
no mesmo processo:

| Medida por registro | Scikit-Learn | ONNX Runtime |
|---|---:|---:|
| Latência observada | 0,01860 ms | 0,01295 ms |
| Tamanho do artefato | 405.604 bytes | 252.634 bytes |

Nesta execução local, o ONNX apresentou ganho observado de `1,44x` e redução de
tamanho de `37,71%`. Latências absolutas variam conforme hardware, sistema
operacional e carga; o resultado relevante é a comparação controlada dentro da
mesma execução. Consulte o [relatório completo](model/onnx_benchmark.json) e o
[Model Card](docs/model_card.md).

## API REST

| Método | Endpoint | Finalidade |
|---|---|---|
| `GET` | `/health` | Confirma que o processo está ativo |
| `GET` | `/ready` | Confirma que o modelo foi carregado e informa sua versão |
| `POST` | `/predict` | Classifica um texto clínico e retorna probabilidades |
| `GET` | `/docs` | Interface OpenAPI/Swagger |
| `GET` | `/metrics` | **Em integração** com o trabalho de monitoramento |

Contrato do `POST /predict`:

```json
{
  "clinical_text": "Severe chest pain and shortness of breath."
}
```

Resposta contém classe, probabilidade por classe, versão e latência da chamada:

```json
{
  "target": "urgente",
  "probabilities": {
    "atencao": 0.2112,
    "normal": 0.0156,
    "urgente": 0.7732
  },
  "model_version": "onnx-nhamcs-2021-v1",
  "inference_time_ms": 1.45
}
```

O texto é obrigatório, tem espaços removidos nas extremidades e aceita no
máximo 5.000 caracteres. O código não persiste nem escreve o texto clínico em
logs.

## Instalação local com Poetry

Pré-requisitos:

- Git;
- Python 3.12 — faixa suportada: `>=3.11,<3.13`;
- Poetry 2.4+;
- Docker 24+ para validação da imagem;
- aproximadamente 2 GB livres caso todos os grupos opcionais sejam instalados.

Instalação completa para dados, notebooks, DVC, otimização, PyTorch e testes:

```bash
poetry env use 3.12
poetry install --with dev,notebooks,pipeline,optimization,training
```

Para executar somente a API com o artefato ONNX versionado:

```bash
poetry install
poetry run uvicorn hospital_triage.api:app --host 0.0.0.0 --port 8000
```

## Reprodução de dados, treino e MLflow

Baixar e validar a fonte oficial:

```bash
poetry run python data/download_dataset.py
```

Preparar a base SQLite e validar o contrato:

```bash
poetry run python -m src.hospital_triage.data_preparation
poetry run python -m src.hospital_triage.data_preparation --validate-only
```

Treinar, avaliar, converter para ONNX e registrar no MLflow:

```bash
poetry run python -m src.hospital_triage.training --git-sha local
```

Abrir a interface local do MLflow:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Depois, acesse [`http://localhost:5000`](http://localhost:5000). O banco
`mlflow.db`, `mlruns/`, dados brutos e bancos preparados são locais e ignorados
pelo Git.

O grupo de dependências `pipeline` instala DVC, mas `dvc.yaml` e `dvc.lock`
ainda não estão versionados. Até essa integração ser concluída, os três
comandos acima são a reprodução oficial disponível e DVC permanece como item
não atendido nesta versão.

## Qualidade e testes

```bash
poetry run ruff check src/hospital_triage tests
poetry run pytest tests ml_prep_kit/tests -v
```

Os testes cobrem preparação e validação dos dados, treino, gates mínimos,
registro no MLflow, API, componentes reutilizáveis e uma integração que carrega
o artefato ONNX real usado no Docker.

## CI/CD

O workflow [`.github/workflows/ml-pipeline.yml`](.github/workflows/ml-pipeline.yml)
é executado em pushes e pull requests para `develop` e `main`. Ele fixa por SHA
imutável um reusable workflow público e interrompe o pipeline em qualquer
falha:

1. instala dependências, executa Ruff e Pytest;
2. baixa, prepara e valida o dataset;
3. recebe o artefato de dados, treina, avalia e registra o modelo;
4. compartilha Joblib, ONNX, métricas e benchmark;
5. constrói a imagem Docker sem publicá-la.

Uma [execução verde de referência](https://github.com/jrrrtricolor/FIAP-fase3-hospital-triage/actions/runs/33260221722)
comprovou os quatro jobs. Uma nova execução deve permanecer verde após cada
integração do grupo.

## Airflow e monitoramento

Esses blocos estão sendo desenvolvidos em paralelo pela integrante Júlia. Para
serem considerados atendidos, a integração final precisa entregar, no mínimo:

### Airflow

- DAG importável em `airflow/dags/`;
- tarefas explícitas de ingestão/validação, preparação, treino, avaliação e
  registro;
- dependências corretas, idempotência e reutilização dos módulos existentes;
- comando documentado e evidência de uma execução concluída.

### Prometheus e Grafana

- `/metrics` usando `prometheus-client`;
- contador por rota, método e status, histograma de duração, contador de erros e
  versão do modelo, sem texto clínico em labels;
- `docker-compose.yml` com API, Prometheus e Grafana;
- scrape configurado e healthchecks coerentes;
- provisioning reproduzível do Grafana;
- dashboard com requisições, latência p95 e erros;
- teste da instrumentação e evidência do dashboard com dados.

O README deve ser revisado novamente depois do merge para substituir esta seção
pelos comandos e links reais. Trabalho apenas anunciado não é contabilizado na
avaliação.

## Estrutura principal

```text
.
├── .github/workflows/          # CI/CD
├── data/                       # aquisição; dados locais ficam fora do Git
├── docs/                       # enunciado e Model Card
├── ml_prep_kit/                # componentes reutilizáveis
├── model/                      # ONNX final e relatórios versionados
├── notebooks/                  # qualidade, EDA, análise e comparação
├── src/hospital_triage/        # preparação, treino e API específicos
├── tests/                      # testes do produto
├── Dockerfile
├── pyproject.toml
└── README.md
```

Entradas futuras esperadas após as integrações paralelas:
`airflow/dags/`, `monitoring/`, `docker-compose.yml`, `dvc.yaml` e `dvc.lock`.

## Limitações e segurança clínica

- o texto é materializado de campos estruturados e não representa laudo livre;
- o modelo aceita somente inglês nesta versão;
- as métricas são de validação retrospectiva, não validação clínica;
- macro F1 e recall de `urgente` ainda são modestos;
- não há avaliação por subgrupos, calibração clínica ou teste prospectivo;
- o dataset representa atendimentos dos Estados Unidos em 2021;
- mudanças de população, protocolos ou prevalência podem causar drift;
- falsos negativos de `urgente` exigem atenção especial;
- nenhuma decisão de atendimento deve ser automatizada com este protótipo;
- não envie nomes, documentos, endereços ou outros identificadores pessoais;
- a API não armazena o texto recebido por padrão.

Consulte o [Model Card](docs/model_card.md) para detalhes de uso pretendido,
riscos e métricas por classe.

## Notebooks e evidências

| Evidência | Arquivo |
|---|---|
| Qualidade dos dados | [notebooks/00_data_quality_nhamcs_2021.ipynb](notebooks/00_data_quality_nhamcs_2021.ipynb) |
| Análise exploratória | [notebooks/01_eda_nhamcs_2021.ipynb](notebooks/01_eda_nhamcs_2021.ipynb) |
| Análise da base de treino | [notebooks/02_training_data_analysis_nhamcs_2021.ipynb](notebooks/02_training_data_analysis_nhamcs_2021.ipynb) |
| Comparação Scikit-Learn × PyTorch | [notebooks/03_model_comparison_nhamcs_2021.ipynb](notebooks/03_model_comparison_nhamcs_2021.ipynb) |
| Relatório estruturado da comparação | [model/mlflow_model_comparison.json](model/mlflow_model_comparison.json) |
| Métricas por classe | [notebooks/03_model_comparison_nhamcs_2021.ipynb](notebooks/03_model_comparison_nhamcs_2021.ipynb) |
| Benchmark ONNX | [model/onnx_benchmark.json](model/onnx_benchmark.json) |

## Equipe

- Cássio
- Júlia

Antes da entrega, incluir nomes completos e identificadores acadêmicos exigidos
pela FIAP, validar o vídeo sem login e criar uma tag imutável da versão avaliada.
