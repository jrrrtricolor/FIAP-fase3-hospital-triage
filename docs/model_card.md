# Model Card — Hospital Triage Text Classifier

## Resumo

| Campo | Valor |
|---|---|
| Nome | `hospital-triage-text-classifier` |
| Versão do dataset | `nhamcs-ed-2021-training-v1` |
| Modelo selecionado | TF-IDF + regressão logística balanceada |
| Formato de inferência | ONNX, opset 18 |
| Idioma | Inglês |
| Classes | `normal`, `atencao`, `urgente` |
| Métrica de seleção | Macro F1 |
| Artefato servido | `model/hospital_triage_model.onnx` |
| Versão servida | `onnx-nhamcs-2021-v1` |
| SHA-256 do ONNX | `dec190039662b20dcfacf9b7583fa8b9c3c9e2ed50efce3c19be7fa08911a6f7` |
| Tamanho do ONNX | 252.634 bytes |
| Data desta Model Card | 2 de setembro de 2026 |

O modelo estima uma classe de prioridade a partir de texto clínico em inglês.
Ele é um demonstrador acadêmico de apoio à triagem e não um dispositivo médico,
diagnóstico ou substituto de avaliação profissional.

## Uso pretendido

Usos compatíveis:

- demonstração acadêmica de classificação textual multiclasse;
- estudo de pipeline de dados, CI/CD, tracking e otimização ONNX;
- apoio experimental à ordenação inicial de casos, sempre com revisão humana;
- testes com textos sem identificadores pessoais.

Usos fora do escopo:

- decisão autônoma sobre prioridade, diagnóstico ou tratamento;
- uso clínico real ou integração direta a prontuário;
- textos em português ou outros idiomas;
- previsão sobre populações ou protocolos não avaliados;
- entrada contendo dados pessoais ou informação identificável;
- interpretação das probabilidades como risco clínico calibrado.

## Dataset

O modelo usa o National Hospital Ambulatory Medical Care Survey — Emergency
Department 2021 (NHAMCS-ED 2021), levantamento público do NCHS/CDC sobre
atendimentos em departamentos de emergência dos Estados Unidos.

Fontes:

- [página oficial](https://www.cdc.gov/nchs/nhamcs/documentation/about-the-data-2021.html);
- [documentação técnica](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/doc21-ed-508.pdf);
- [download do arquivo Stata](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/stata/ed2021-stata.zip).

O uso deve respeitar o NCHS Data User Agreement, incluindo a proibição de
tentativa de reidentificação. A fonte pública não contém os textos narrativos
originais. O pipeline cria `clinical_text` deterministicamente com:

- descrições oficiais dos motivos da visita `RFV1` a `RFV5`;
- idade e sexo registrados;
- temperatura, frequência cardíaca e respiratória;
- pressão arterial, saturação de oxigênio e escala de dor, quando disponíveis.

Diagnósticos, exames, medicamentos e disposição, informações posteriores à
triagem, não são usados como features.

### Target

O target deriva diretamente de `IMMEDR`, prioridade de atendimento registrada:

- níveis 1 e 2: `urgente`;
- nível 3: `atencao`;
- níveis 4 e 5: `normal`.

Esse mapeamento preserva a ordem clínica original e não transforma categorias
de doença arbitrárias em urgência.

### Volume e splits

| Item | Registros |
|---|---:|
| Total elegível | 10.495 |
| Textos únicos | 10.492 |
| Treino | 7.350 |
| Validação | 1.573 |
| Teste reservado | 1.572 |

Classes no conjunto completo: 3.186 `normal`, 5.429 `atencao` e 1.880
`urgente`. A divisão usa `StratifiedGroupKFold` com seed 42 e `text_hash` como
grupo, impedindo que textos repetidos apareçam em splits diferentes.

## Desenvolvimento do modelo

O baseline combina:

- `TfidfVectorizer` com unigramas e bigramas;
- no máximo 10.000 features;
- `min_df=2`;
- tokenização ASCII compatível com a conversão ONNX;
- regressão logística com `class_weight="balanced"`, `max_iter=1000` e seed 42.

Um classificador textual PyTorch leve também foi avaliado na mesma divisão. O
baseline foi selecionado por macro F1 de validação superior (`0,5582` contra
`0,5349`), apesar de o candidato PyTorch apresentar menor latência naquela
execução. A comparação completa está em
[`model/mlflow_model_comparison.json`](../model/mlflow_model_comparison.json).

## Avaliação preditiva

As métricas abaixo foram calculadas sobre 1.573 registros de validação. O split
de teste permanece reservado e não é apresentado como evidência final nesta
versão.

| Métrica global | Valor |
|---|---:|
| Acurácia | 0,5709 |
| Macro precision | 0,5510 |
| Macro recall | 0,5833 |
| Macro F1 | 0,5582 |
| Weighted F1 | 0,5744 |
| Recall de `urgente` | 0,5780 |

Métricas por classe:

| Classe | Precision | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| `normal` | 0,5616 | 0,6499 | 0,6025 | 477 |
| `atencao` | 0,6778 | 0,5221 | 0,5899 | 814 |
| `urgente` | 0,4137 | 0,5780 | 0,4822 | 282 |

O recall de `urgente` mostra que uma parcela relevante dessa classe não foi
identificada. Esse é o principal risco do modelo e impede qualquer alegação de
segurança clínica.

## Otimização ONNX

O pipeline Scikit-Learn foi convertido com `skl2onnx` para opset 18 e executado
com ONNX Runtime em CPU. A comparação usa a mesma amostra de 1.573 registros e
o mesmo processo, excluindo o carregamento dos modelos.

| Medida por registro | Scikit-Learn | ONNX Runtime |
|---|---:|---:|
| Latência observada | 0,15882 ms | 0,06740 ms |
| Tamanho | 405.604 bytes | 252.634 bytes |

Resultados:

- concordância das classes previstas: `100%`;
- ganho observado: `2,36x`;
- redução do artefato: `37,71%`.

O benchmark foi executado sobre todo o split de validação, no mesmo processo e
ambiente, sem incluir o carregamento dos modelos. Ele registra a média por
registro de uma passagem controlada. Os números absolutos não devem ser
extrapolados para produção. Esta versão ainda não mede warm-up, múltiplas
repetições, p50, p95, desvio-padrão ou memória. O resultado reproduzível está em
[`model/onnx_benchmark.json`](../model/onnx_benchmark.json).

## Operação e versionamento

A imagem Docker copia o ONNX final e define `MODEL_PATH` explicitamente. A API
carrega o modelo uma vez no lifespan e informa a versão por `/ready` e
`POST /predict`. O MLflow permanece no fluxo batch de treinamento e registro;
a imagem de inferência aceita somente o ONNX promovido e empacotado. O endpoint
`/metrics` expõe métricas técnicas para Prometheus e o dashboard Grafana é
provisionado pelo Docker Compose.

A DAG `hospital_triage` reutiliza os módulos do projeto para download,
preparação, validação, treinamento, avaliação e validação do registro no MLflow.
Treinamento e retreinamento são batch; apenas a inferência ocorre em tempo real.

O treinamento registra:

- parâmetros e versão do dataset;
- métricas globais e recall de `urgente`;
- Joblib, ONNX e benchmark;
- versão no Model Registry;
- alias `champion` e SHA do código.

## Limitações

- `clinical_text` é texto templado derivado de campos estruturados, não laudo
  narrativo livre;
- a fonte representa os Estados Unidos e o ano de 2021;
- o modelo foi avaliado retrospectivamente e não prospectivamente;
- não há validação externa, estudo por hospital ou análise temporal;
- não há avaliação por idade, sexo ou outros subgrupos;
- as probabilidades não foram calibradas para decisão clínica;
- o recall de `urgente` é insuficiente para automação;
- a classe `urgente` é minoritária e apresenta o pior F1;
- abreviações, erros de digitação e linguagem fora do padrão do corpus podem
  degradar o resultado;
- não há detecção de drift implementada;
- desempenho em português é desconhecido e não suportado.

## Riscos e mitigação

| Risco | Consequência | Mitigação exigida |
|---|---|---|
| Falso negativo de `urgente` | Atraso potencial no atendimento | Revisão humana obrigatória; nunca automatizar prioridade |
| Drift de população ou protocolo | Queda silenciosa de qualidade | Monitorar distribuição e reavaliar periodicamente |
| Viés por subgrupo | Tratamento desigual | Avaliação estratificada antes de qualquer piloto |
| Texto fora do domínio | Predições instáveis | Validar idioma/formato e rejeitar uso não suportado |
| Vazamento de dados clínicos | Risco de privacidade | Não registrar texto em logs ou labels; usar dados sem identificadores |
| Confiança excessiva nas probabilidades | Decisão clínica incorreta | Apresentar como score não calibrado e manter profissional no circuito |

## Manutenção recomendada

- executar os testes e o benchmark a cada alteração de dados ou dependências;
- registrar nova versão e imagem para cada modelo promovido;
- não sobrescrever silenciosamente o artefato de uma release;
- comparar métricas por classe e não apenas acurácia;
- bloquear promoção se macro F1 ou recall de `urgente` ficarem abaixo dos gates;
- revisar esta Model Card após qualquer mudança de dados, modelo ou contrato da API.

## Evidências

- preparação: [`src/hospital_triage/data_preparation.py`](../src/hospital_triage/data_preparation.py);
- treino e benchmark: [`src/hospital_triage/training.py`](../src/hospital_triage/training.py);
- API: [`src/hospital_triage/api.py`](../src/hospital_triage/api.py);
- métricas de treinamento: [`model/training_metrics.json`](../model/training_metrics.json);
- métricas por classe: [`notebooks/03_model_comparison_nhamcs_2021.ipynb`](../notebooks/03_model_comparison_nhamcs_2021.ipynb);
- benchmark: [`model/onnx_benchmark.json`](../model/onnx_benchmark.json);
- comparação de candidatos: [`model/mlflow_model_comparison.json`](../model/mlflow_model_comparison.json);
- CI/CD: [`.github/workflows/ml-pipeline.yml`](../.github/workflows/ml-pipeline.yml);
- Airflow: [`airflow/dags/hospital_triage_dag.py`](../airflow/dags/hospital_triage_dag.py);
- monitoramento: [`docker-compose.yml`](../docker-compose.yml) e
  [dashboard Grafana](../src/hospital_triage/grafana/dashboards/hospital-triage.json);
- testes: [`tests/`](../tests/) e [`ml_prep_kit/tests/`](../ml_prep_kit/tests/).
