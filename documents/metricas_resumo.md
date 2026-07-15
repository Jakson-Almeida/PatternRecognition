# Resumo das métricas de avaliação (Trabalhos 1–3)

Fontes: `main.tex` (Trabalho 1), `trabalho2.tex` (Trabalho 2) e `trabalho3.tex` (Trabalho 3).

---

## Trabalho 1 — Classificadores bayesianos e k-NN (dados sintéticos Gaussianos)

Protocolo: treino e teste fixos (1000 amostras/classe), sem validação cruzada.

| Métrica | Uso |
|---------|-----|
| **Probabilidade de erro teórica** \(P_e\) | Erro de Bayes (e do Euclidiano com \(\mu\) verdadeiros), via \(\Phi(-\delta/2)\) e distância de Mahalanobis. Referência ótima. |
| **Taxa / probabilidade de erro empírica** \(\hat{P}_e\) | Fração de classificações incorretas no treino (Q3) ou no teste (demais). Métrica principal de comparação. |
| **Matriz de confusão** | Contagens \(C_i \to C_j\); detalhada sobretudo para o k-NN (\(k \in \{1,5,11\}\)). |

Não foram usadas acurácia nomeada, precisão, recall, F1 nem AUC-ROC. A comparação resume-se a erro teórico vs. erro de teste.

---

## Trabalho 2 — Perceptron, mínimos quadrados e SVM linear

Protocolo: mesmo gerador Gaussiano; um conjunto de teste fixo (2000 amostras). Referência: erro teórico de Bayes \(\approx 0{,}1056\).

| Métrica | Uso |
|---------|-----|
| **Erro teórico de Bayes** \(P_e^{\mathrm{(Bayes)}}\) | Limite inferior / referência para o problema gerador. |
| **Erro empírico de teste** \(\hat{P}_e\) | Métrica central para Perceptron, MQ e SVM, com treino completo e com subconjuntos \(50{+}50\). |
| **Dispersão do erro entre blocos** | Comparação qualitativa de **variância** (dispersão de \(\hat{P}_e\) nos dois blocos pequenos) e **polarização/viés** (afastamento do erro de Bayes com treino grande). Sem Monte Carlo formal. |
| **Número de vetores de suporte** | Descriptor auxiliar do SVM (não é métrica de desempenho). |

Não há precisão, recall, F1, AUC-ROC nem matriz de confusão tabulada; o quadro-resumo compara apenas \(\hat{P}_e\).

---

## Trabalho 3 — German Credit (seis classificadores)

Protocolo: validação cruzada estratificada com \(k=5\); SMOTE só no treino de cada fold; classe positiva = **bad**. Reportam-se média ± desvio padrão entre folds. Matriz de confusão e curva ROC montadas com as predições de teste agregadas.

| Métrica | Definição / papel |
|---------|-------------------|
| **Acurácia** | Fração de acertos totais. |
| **Precisão** | \(\mathrm{VP}/(\mathrm{VP}+\mathrm{FP})\). |
| **Revocação (recall / TPR)** | \(\mathrm{VP}/(\mathrm{VP}+\mathrm{FN})\); enfatizada por causa do custo assimétrico de FN. |
| **F1-score** | Média harmônica de precisão e revocação. |
| **AUC-ROC** | Área sob a curva ROC a partir de escore contínuo de confiança. |
| **Matriz de confusão** | Contagens VP/FP/FN/VN agregadas na CV. |
| **Curva ROC** | Visualização do trade-off TPR–FPR. |
| **Custo total** | \(5\cdot\mathrm{FN} + 1\cdot\mathrm{FP}\), conforme a matriz de custos do Statlog German Credit. |

---

## Evolução ao longo dos trabalhos

| Aspecto | T1 | T2 | T3 |
|---------|----|----|-----|
| Métrica dominante | Erro teórico / empírico | Erro empírico de teste (+ viés/variância qualitativos) | Acc, Prec, Rec, F1, AUC + custo |
| Matriz de confusão | Sim (k-NN) | Não | Sim (todos) |
| ROC / AUC | Não | Não | Sim |
| Validação | Hold-out fixo | Hold-out fixo | CV estratificada \(k=5\) |
| Custo de erro | Simétrico (implícito) | Simétrico (implícito) | Assimétrico (\(5\)–\(1\)) |
