# Detecção de Anomalias em Transações

Projeto desenvolvido em Python a partir do desafio **Detecção de Anomalias em Transações**, utilizando um cenário educacional de transações bancárias para investigar classificação de eventos raros, desbalanceamento de classes, seleção de threshold e interpretabilidade de modelos.

O projeto também possui uma extensão experimental de laboratório, na qual a investigação é aprofundada por meio de métricas e técnicas adicionais de análise.

---

## Sobre o projeto

Detectar transações fraudulentas não significa apenas alcançar uma alta acurácia.

Em um conjunto de dados extremamente desbalanceado, um modelo pode apresentar uma acurácia elevada e ainda assim deixar passar uma quantidade relevante de eventos da classe minoritária.

Por isso, este projeto investiga uma sequência de decisões:

```text
Dados
  ↓
Evidências
  ↓
Hipótese
  ↓
Experimento
  ↓
Métricas
  ↓
Threshold
  ↓
Classificação
  ↓
Explicabilidade
  ↓
Limitações
  ↓
Evolução
```

A proposta é observar não apenas qual resultado o modelo produz, mas também como a decisão foi construída, quais evidências sustentam sua interpretação e quais limitações permanecem.

---

## Objetivos

- Analisar o desbalanceamento existente no conjunto de dados.
- Estabelecer um modelo de referência com Regressão Logística.
- Avaliar o efeito do balanceamento de classes.
- Experimentar XGBoost com `scale_pos_weight`.
- Comparar Precision, Recall, F1 e ROC-AUC.
- Separar treinamento, validação e teste final.
- Selecionar o threshold utilizando exclusivamente o conjunto de validação.
- Avaliar o modelo final em um conjunto de teste mantido separado.
- Investigar a importância das variáveis.
- Utilizar SHAP para interpretar decisões individuais do modelo.
- Registrar limitações e evitar interpretações causais indevidas.
- Estender a investigação no Lab utilizando Precision-Recall e Average Precision (AP / PR-AUC).

---

# Dataset

O projeto utiliza o dataset `creditcard.csv`, contendo transações de cartão de crédito.

## Dimensões

- **284.807 registros**
- **30 variáveis preditoras**
- Variável-alvo: `Class`

A variável `Class` representa:

```text
0 → transação não fraudulenta
1 → transação fraudulenta
```

## Distribuição das classes

| Classe | Quantidade | Proporção |
|---|---:|---:|
| 0 | 284.315 | 99,8273% |
| 1 | 492 | 0,1727% |

A relação aproximada entre transações normais e fraudulentas é de **578:1**.

Esse desbalanceamento é um dos elementos centrais da investigação.

---

# Metodologia

O experimento foi estruturado para evitar que o conjunto de teste final fosse utilizado durante a escolha do threshold.

## Divisão dos dados

Inicialmente, foi realizada uma divisão estratificada entre treinamento e teste.

O conjunto de treinamento foi posteriormente dividido em treinamento do modelo e validação.

```text
Dataset
│
├── Treinamento
│   ├── Treinamento do modelo
│   └── Validação
│
└── Teste final
```

Dimensões utilizadas no experimento final:

| Conjunto | Registros | Fraudes |
|---|---:|---:|
| Treinamento do modelo | 182.276 | 315 |
| Validação | 45.569 | 79 |
| Teste final | 56.962 | 98 |

O **teste final permaneceu separado durante a seleção do threshold**.

---

# Fluxo de decisão

Uma das premissas centrais do projeto é separar a probabilidade produzida pelo modelo da classificação final.

```text
Feature
   ↓
Modelo
   ↓
Probabilidade
   ↓
Threshold
   ↓
Classificação
   ↓
Explicação
```

O modelo produz uma probabilidade estimada para a classe positiva.

O threshold transforma essa probabilidade em uma decisão de classificação.

Por isso, o threshold é tratado como parte da metodologia experimental e não apenas como um detalhe de implementação.

---

# Experimentos

## Experimento 1 — Regressão Logística

Foi utilizada Regressão Logística com `StandardScaler` como baseline.

Configuração:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])
```

O objetivo foi estabelecer uma referência para comparação com modelos posteriores.

### Resultado no teste

| Métrica | Classe 1 |
|---|---:|
| Precision | 82,67% |
| Recall | 63,27% |
| F1 | 71,68% |
| ROC-AUC | 96,05% |

Matriz de confusão:

```text
[[56851,    13],
 [   36,    62]]
```

---

## Experimento 2 — Regressão Logística com balanceamento

Foi utilizada Regressão Logística com:

```python
class_weight="balanced"
```

O objetivo foi observar como o tratamento explícito do desbalanceamento altera o comportamento do classificador.

### Resultado no teste

| Métrica | Classe 1 |
|---|---:|
| Precision | 6,10% |
| Recall | 91,84% |
| F1 | 11,44% |
| ROC-AUC | 97,21% |

Matriz de confusão:

```text
[[55478, 1386],
 [    8,   90]]
```

O experimento evidenciou um trade-off entre Precision e Recall: o aumento do Recall da classe minoritária veio acompanhado de um aumento expressivo de falsos positivos.

---

## Experimento 3 — XGBoost com `scale_pos_weight`

Foi utilizado XGBoost com ponderação da classe positiva.

O peso foi calculado exclusivamente sobre o conjunto utilizado para treinamento do modelo:

```python
scale_pos_weight = (
    (y_train_model == 0).sum()
    / (y_train_model == 1).sum()
)
```

Valor utilizado no experimento:

```text
577,653968
```

### Configuração

```python
XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
```

---

# Seleção do threshold

O projeto considera importante distinguir:

```text
Probabilidade ≠ Classificação
```

O modelo produz uma probabilidade.

O threshold determina a partir de qual valor essa probabilidade será interpretada como classe positiva.

A análise foi realizada sobre o conjunto de validação.

Entre os thresholds avaliados, o valor de **0,90** apresentou o maior F1 observado na validação.

```python
THRESHOLD_FINAL = 0.90
```

O teste final não foi utilizado para escolher o threshold.

Após a seleção, o modelo foi avaliado no conjunto de teste final.

---

# Resultado final

No conjunto de teste final, utilizando o XGBoost treinado com `scale_pos_weight` e o threshold de **0,90 definido a partir da validação**, foram obtidos:

| Métrica | Resultado |
|---|---:|
| Precision — classe 1 | 91,95% |
| Recall — classe 1 | 81,63% |
| F1 — classe 1 | 86,49% |
| ROC-AUC | 98,31% |
| Accuracy | 99,96% |

## Matriz de confusão

```text
[[56857,     7],
 [   18,    80]]
```

Em termos absolutos no teste final:

- **80** transações fraudulentas foram identificadas;
- **18** transações fraudulentas não foram identificadas;
- **7** transações normais foram classificadas como fraude;
- **56.857** transações normais foram classificadas corretamente.

Esses resultados descrevem o comportamento observado neste dataset, nesta divisão dos dados, nesta configuração do modelo e neste threshold.

Eles não representam garantia de desempenho em outros datasets ou ambientes de produção.

---

# Comparação dos experimentos

| Modelo / Configuração | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression + StandardScaler | 0,50 | 82,67% | 63,27% | 71,68% | 96,05% |
| Logistic Regression + `class_weight="balanced"` | 0,50 | 6,10% | 91,84% | 11,44% | 97,21% |
| XGBoost + `scale_pos_weight` | 0,50 | 82,18% | 84,69% | 83,42% | 98,31% |
| XGBoost + `scale_pos_weight` | **0,90** | **91,95%** | **81,63%** | **86,49%** | **98,31%** |

A tabela representa os resultados observados nas condições experimentais deste projeto.

O objetivo não é declarar um modelo universalmente superior, mas analisar como configuração, balanceamento e threshold modificaram o comportamento do classificador.

---

# H-001 — Impacto do Desbalanceamento

## Hipótese

> Em conjuntos de dados altamente desbalanceados, métricas agregadas como Accuracy podem não representar adequadamente a capacidade de identificar a classe minoritária.

## Evidência experimental

Os experimentos apresentaram comportamentos diferentes conforme o tratamento do desbalanceamento.

A Regressão Logística balanceada aumentou o Recall da classe fraudulenta para **91,84%**, mas também produziu **1.386 falsos positivos**.

O XGBoost com `scale_pos_weight` apresentou outro equilíbrio entre Precision e Recall, especialmente após a definição do threshold utilizando o conjunto de validação.

## Conclusão

Os resultados experimentais **apoiam H-001 dentro das condições deste experimento**.

O principal aprendizado é que a avaliação de um detector de eventos raros não deve depender exclusivamente da Accuracy.

---

# H-DIFF-001 — Decisão orientada por evidências

Além das hipóteses, o projeto adota como princípio metodológico:

> Uma decisão de classificação deve ser analisada considerando probabilidade, threshold, métricas e contexto da avaliação.

A pergunta deixa de ser apenas:

```text
"Qual modelo teve a maior métrica?"
```

e passa a considerar:

```text
Qual foi a probabilidade?
        ↓
Qual threshold foi utilizado?
        ↓
Quantos falsos positivos ocorreram?
        ↓
Quantos falsos negativos permaneceram?
        ↓
Como Precision, Recall e F1 se comportaram?
        ↓
Como podemos interpretar a decisão?
        ↓
Quais limitações permanecem?
```

Esse princípio orientou a separação entre treinamento, validação e teste final e a escolha do threshold de 0,90.

---

# H-002 — Interpretabilidade das decisões

## Pergunta

> Quais características do dataset mais influenciam as decisões do XGBoost?

As variáveis `V1` a `V28` são variáveis anonimizadas.

Por esse motivo, seus valores não recebem interpretações semânticas ou causais neste projeto.

---

## Feature Importance

A importância agregada fornecida pelo XGBoost indicou:

| Feature | Importância |
|---|---:|
| V14 | 45,48% |
| V4 | 6,23% |
| V10 | 5,99% |
| V8 | 4,76% |
| Amount | 3,53% |
| V12 | 3,29% |

A interpretação correta é:

> V14 apresentou a maior importância agregada entre as features segundo a configuração do XGBoost utilizada no experimento.

Isso **não significa** que V14 seja responsável por 45,48% das fraudes.

A importância de uma feature representa o comportamento do modelo, não uma relação causal com o fenômeno real.

---

# SHAP

Foi utilizada a biblioteca SHAP para investigar o comportamento do modelo de forma global e individual.

Versão utilizada:

```text
SHAP 0.52.0
```

A análise global foi realizada sobre uma amostra de 2.000 observações da validação.

```python
X_shap = X_val.sample(
    n=min(2000, len(X_val)),
    random_state=42
)

explainer = shap.TreeExplainer(xgb_validation)
shap_values = explainer(X_shap)
```

A análise individual também foi utilizada para observar como diferentes features contribuíram para o resultado de uma observação específica.

Em uma das observações analisadas, o modelo produziu:

```text
Probabilidade estimada: 0,0266%
Classe real: 0
Predição: 0
Threshold: 0,90
```

A análise SHAP mostrou contribuições que reduziram o output do modelo para essa observação.

A interpretação utilizada é:

> determinada feature contribuiu para aumentar ou reduzir o output do modelo naquela observação.

Não é feita a afirmação de que determinada variável **causou** ou **impediu** uma fraude.

---

# Extensão do Lab

Além da entrega principal do desafio, este projeto possui uma camada experimental de laboratório.

Como parte dessa extensão, será realizada uma análise complementar utilizando:

- Precision-Recall Curve;
- Average Precision (AP / PR-AUC);
- análise complementar do comportamento da classe minoritária;
- investigação adicional da relação entre Precision, Recall e threshold.

Essa análise **não é tratada como requisito adicional da entrega original do desafio**.

Ela pertence ao **Lab**, cujo objetivo é aprofundar a investigação metodológica sobre modelos de classificação em datasets altamente desbalanceados.

## Questões do Lab

- Como a curva Precision-Recall representa o comportamento do modelo diante da classe minoritária?
- Como Average Precision complementa a análise realizada com ROC-AUC?
- Como diferentes thresholds modificam Precision e Recall?
- Quais evidências adicionais podem ser obtidas a partir de PR-AUC/AP?

Os resultados dessa extensão serão mantidos separados da avaliação principal do desafio.

---

# Principais aprendizados

## 1. Accuracy pode esconder o problema

Quando a classe positiva representa apenas **0,1727%** dos registros, uma Accuracy elevada não é suficiente para compreender o comportamento do detector.

## 2. Balanceamento muda o comportamento do modelo

O uso de `class_weight="balanced"` aumentou o Recall da classe fraudulenta, mas também elevou significativamente os falsos positivos.

## 3. O threshold faz parte da decisão

O modelo produz probabilidades.

O threshold transforma essas probabilidades em classes.

Por isso, escolher o threshold é uma decisão metodológica importante.

## 4. Validação e teste possuem funções diferentes

O conjunto de validação foi utilizado para selecionar o threshold.

O conjunto de teste permaneceu separado para avaliar o comportamento final do modelo.

## 5. Métrica não é explicação

Uma métrica pode mostrar **o que aconteceu**.

Feature Importance e SHAP ajudam a investigar **como o modelo utilizou as características disponíveis**.

## 6. Interpretabilidade não é causalidade

As variáveis anonimizadas do dataset não permitem concluir que determinada feature seja uma causa real de fraude.

A análise realizada descreve o comportamento do modelo.

---

# Limitações

Este projeto possui caráter educacional e experimental.

Entre as principais limitações estão:

- o dataset utilizado possui variáveis anonimizadas;
- não é possível atribuir significado de negócio às variáveis `V1`–`V28`;
- os resultados dependem da divisão dos dados e da configuração dos modelos;
- não foram realizados testes em dados reais de uma instituição financeira;
- o threshold de 0,90 foi definido com base no F1 da validação;
- custos de negócio diferentes para falso positivo e falso negativo não foram incorporados ao critério de decisão;
- não há garantia de generalização para outros períodos, instituições ou padrões de fraude;
- SHAP descreve o comportamento do modelo e não estabelece causalidade;
- as análises realizadas não substituem validação operacional, monitoramento ou requisitos de um sistema de produção.

---

# Tecnologias

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib

---

# Estrutura conceitual

```text
bank-transaction-anomaly-detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── anomaly_detection.ipynb
│
├── docs/
│   └── ...
│
└── README.md
```

A estrutura física do repositório pode ser ajustada conforme a organização final dos arquivos.

---

# Fluxo experimental

```text
                         DATASET
                            │
                            ▼
                  Análise exploratória
                            │
                            ▼
                    Desbalanceamento
                         578:1
                            │
                            ▼
                ┌────────────────────┐
                │       H-001        │
                │  Desbalanceamento  │
                └─────────┬──────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       Logistic Regression          XGBoost
              │                       │
              ▼                       ▼
          Métricas             scale_pos_weight
                                      │
                                      ▼
                                  Validação
                                      │
                                      ▼
                            Seleção do threshold
                                      │
                                      ▼
                                Threshold 0,90
                                      │
                                      ▼
                                Teste final
                                      │
                                      ▼
                         Precision / Recall / F1
                                ROC-AUC
                                      │
                                      ▼
                            Feature Importance
                                      │
                                      ▼
                                    SHAP
                                      │
                                      ▼
                                   H-002
                                      │
                                      ▼
                              Extensão do Lab
                                      │
                                      ▼
                            Precision-Recall
                                      │
                                      ▼
                             AP / PR-AUC
```

---

# Conclusão

O projeto mostrou, dentro das condições experimentais utilizadas, que a detecção de eventos raros exige mais do que observar uma única métrica.

O desbalanceamento influencia o comportamento dos classificadores, o threshold modifica diretamente a relação entre falsos positivos e falsos negativos e técnicas de interpretabilidade permitem investigar como o modelo utiliza as características disponíveis.

A principal abordagem metodológica adotada foi manter uma cadeia explícita entre:

```text
Evidência
   ↓
Hipótese
   ↓
Experimento
   ↓
Métrica
   ↓
Decisão
   ↓
Explicação
   ↓
Limitação
   ↓
Evolução
```

A extensão do Lab continuará essa investigação utilizando **Precision-Recall e Average Precision (AP / PR-AUC)** como análise complementar para cenários de forte desbalanceamento.

---

# Contexto

Projeto desenvolvido para fins educacionais e experimentais a partir do desafio de **Detecção de Anomalias em Transações em Python**.

Os resultados apresentados são específicos ao dataset, à metodologia, à divisão dos dados e às configurações utilizadas neste estudo.

Não representam um sistema de detecção de fraude pronto para uso em produção.

