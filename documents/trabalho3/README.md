# Terceiro Trabalho de Reconhecimento de Padrões

Classificação de risco de crédito com o banco **Statlog (German Credit Data)** do repositório UCI.

## Enunciado

Projetar sistemas de reconhecimento de padrões utilizando os seguintes classificadores:

- Mínimos Quadrados Linear
- Vizinho mais Próximo (kNN)
- Rede MLP
- SVM não-linear
- AdaBoost
- Floresta Aleatória (Random Forest)

Os resultados devem ser comparados por **validação cruzada**. O projeto envolve **pré-processamento** dos dados, implementação dos classificadores e metodologia adequada de avaliação.

## Banco de dados

O dataset descreve **1000** pedidos de crédito a um banco alemão. Cada instância tem **20 atributos** (7 numéricos e 13 categóricos) e uma classe binária:

| Classe | Significado |
|--------|-------------|
| 1 | Good — bom pagador |
| 2 | Bad — mau pagador (risco de inadimplência) |

A distribuição é aproximadamente **70% good** e **30% bad**. A documentação original inclui uma **matriz de custos assimétrica**: classificar um mau cliente como bom é penalizado com custo 5, enquanto o erro inverso tem custo 1.

### Arquivos nesta pasta

```
trabalho3/
├── README.md
├── statlog+german+credit+data.zip
└── german_credit_data/
    ├── german.data           # versão original (atributos categóricos simbólicos)
    ├── german.data-numeric   # versão Statlog (24 atributos numéricos)
    ├── german.doc            # descrição completa dos atributos e matriz de custos
    └── Index
```

- **`german.data`**: códigos como `A11`, `A34`, `A201`; exige codificação de variáveis categóricas.
- **`german.data-numeric`**: versão já convertida para algoritmos que exigem entradas numéricas; é a forma usada pelo Statlog.

## Pré-processamento

Pontos a considerar antes de treinar os classificadores:

1. Codificação de variáveis categóricas (se usar `german.data`) ou revisão da versão numérica.
2. Normalização ou padronização (relevante para kNN, SVM e MLP).
3. Desbalanceamento de classes (~70/30) — métricas estratificadas e, se necessário, `class_weight` ou amostragem.
4. Validação cruzada estratificada para comparação justa entre os métodos.

## Referências

- [UCI Machine Learning Repository — Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)
- Descrição local dos atributos: `german_credit_data/german.doc`
