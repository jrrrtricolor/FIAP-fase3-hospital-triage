# Model Card — Hospital Triage Text Classifier

## Identificação

| Campo | Valor |
|---|---|
| Nome registrado | `hospital-triage-text-classifier` |
| Dataset | `nhamcs-ed-2021-training-v1` |
| Modelo | TF-IDF + regressão logística balanceada |
| Inferência | ONNX Runtime, opset 18 |
| Idioma | Inglês |
| Classes | `normal`, `atencao`, `urgente` |
| Seleção | Macro F1 de validação |
| Artefato | [`model/hospital_triage_model.onnx`](../model/hospital_triage_model.onnx) |
| SHA-256 | `dec190039662b20dcfacf9b7583fa8b9c3c9e2ed50efce3c19be7fa08911a6f7` |
| Tamanho | 252.634 bytes |
| Data | 29 de agosto de 2026 |

O modelo estima prioridade a partir de texto clínico em inglês. É um
demonstrador acadêmico de apoio à triagem, não um dispositivo médico ou
substituto de avaliação profissional.

## Uso

Usos pretendidos:

- classificação textual multiclasse em demonstrações acadêmicas;
- estudo de pipeline, tracking, API e otimização ONNX;
- apoio experimental, sempre com revisão humana;
- testes com textos sem identificadores pessoais.

Fora do escopo:

- diagnóstico, tratamento ou priorização autônoma;
- uso clínico real ou integração direta a prontuário;
- textos em português ou fora do domínio avaliado;
- entrada com dados pessoais ou identificáveis;
- interpretar probabilidades como risco clínico calibrado.

## Dados e target

A fonte é o
[NHAMCS Emergency Department 2021](https://www.cdc.gov/nchs/nhamcs/documentation/about-the-data-2021.html),
levantamento público do NCHS/CDC sobre emergências dos Estados Unidos. O uso
deve respeitar o NCHS Data User Agreement e a proibição de reidentificação.

O NHAMCS não fornece narrativa livre. O pipeline constrói `clinical_text` com
descrições oficiais dos motivos da visita, idade, sexo, dor e sinais vitais
disponíveis na triagem. Diagnósticos, exames, medicamentos e disposição não são
features.

O target deriva da prioridade `IMMEDR`:

| IMMEDR | Target |
|---|---|
| 1–2 | `urgente` |
| 3 | `atencao` |
| 4–5 | `normal` |

| Conjunto | Registros |
|---|---:|
| Total elegível | 10.495 |
| Textos únicos | 10.492 |
| Treino | 7.350 |
| Validação | 1.573 |
| Teste reservado | 1.572 |

As classes completas contêm 3.186 `normal`, 5.429 `atencao` e 1.880
`urgente`. A divisão usa `StratifiedGroupKFold`, seed 42 e `text_hash`, evitando
o mesmo texto em splits diferentes.

## Desenvolvimento e avaliação

O pipeline usa `TfidfVectorizer` com unigramas e bigramas, até 10.000 features,
`min_df=2` e regressão logística com `class_weight="balanced"` e
`max_iter=1000`. O baseline superou o candidato PyTorch em macro F1 (`0,5582`
contra `0,5349`) na mesma validação.

Métricas sobre 1.573 registros de validação:

| Métrica global | Valor |
|---|---:|
| Acurácia | 0,5709 |
| Macro precision | 0,5510 |
| Macro recall | 0,5833 |
| Macro F1 | 0,5582 |
| Weighted F1 | 0,5744 |
| Recall de `urgente` | 0,5780 |

| Classe | Precision | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| `normal` | 0,5616 | 0,6499 | 0,6025 | 477 |
| `atencao` | 0,6778 | 0,5221 | 0,5899 | 814 |
| `urgente` | 0,4137 | 0,5780 | 0,4822 | 282 |

Matriz de confusão; linhas são classes reais e colunas, previstas:

| Real \ Prevista | `normal` | `atencao` | `urgente` |
|---|---:|---:|---:|
| `normal` | 310 | 121 | 46 |
| `atencao` | 204 | 425 | 185 |
| `urgente` | 38 | 81 | 163 |

Dos 282 casos `urgente`, 119 foram classificados em prioridade inferior. Esse
é o risco preditivo mais importante e impede qualquer uso autônomo.

## Otimização e inferência

O Scikit-Learn foi convertido com `skl2onnx` e executado em CPU. A comparação
usa os mesmos 1.573 registros no mesmo processo e exclui o carregamento dos
modelos.

| Medida por registro | Scikit-Learn | ONNX Runtime |
|---|---:|---:|
| Latência observada | 0,01860 ms | 0,01295 ms |
| Tamanho | 405.604 bytes | 252.634 bytes |

O ONNX manteve 100% de concordância das classes, apresentou ganho observado de
`1,44x` e reduziu o artefato em `37,71%`. Os números absolutos dependem do
ambiente; consumo de memória ainda não foi medido. O relatório está em
[`model/onnx_benchmark.json`](../model/onnx_benchmark.json).

A imagem Docker inclui o ONNX final, carrega-o uma vez no startup e expõe a
versão em `/ready` e `/predict`. O treinamento registra parâmetros, dataset,
métricas, Joblib, ONNX, benchmark, versão no Model Registry, alias `champion` e
SHA do código.

## Limitações e riscos

| Risco ou limitação | Impacto | Mitigação |
|---|---|---|
| Falso negativo de `urgente` | Possível atraso no atendimento | Revisão humana obrigatória |
| Dados dos EUA em 2021 | Generalização desconhecida | Validação externa antes de qualquer piloto |
| Texto templado e somente em inglês | Queda fora do formato avaliado | Validar idioma e domínio |
| Ausência de análise por subgrupo | Viés não quantificado | Avaliação estratificada |
| Probabilidades não calibradas | Confiança indevida | Não interpretar como risco clínico |
| Drift de população ou protocolo | Degradação silenciosa | Monitoramento e reavaliação periódica |
| Dados identificáveis na entrada | Risco de privacidade | Não registrar texto; usar dados desidentificados |

Também não há validação prospectiva, clínica, temporal ou por hospital. O
desempenho em português é desconhecido. Abreviações, erros e linguagem fora do
corpus podem degradar o resultado.

## Governança e evidências

Uma nova versão exige execução dos testes e benchmark, comparação por classe,
registro no MLflow e nova imagem imutável. A Model Card deve ser revisada quando
dataset, modelo, observabilidade ou política de uso mudarem.

- preparação: [`data_preparation.py`](../src/hospital_triage/data_preparation.py);
- treino: [`training.py`](../src/hospital_triage/training.py);
- API: [`api.py`](../src/hospital_triage/api.py);
- comparação: [`mlflow_model_comparison.json`](../model/mlflow_model_comparison.json);
- métricas por classe: [`notebook 03`](../notebooks/03_model_comparison_nhamcs_2021.ipynb);
- benchmark: [`onnx_benchmark.json`](../model/onnx_benchmark.json);
- testes: [`tests/`](../tests/).
