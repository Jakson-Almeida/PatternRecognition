# Guia do Trabalho Final — Classificação de FBGs para demodulação de LPFG

Este guia organiza os passos para construir o classificador, avaliar os resultados e escrever o artigo no template IEEEtran (conferência). Todo o trabalho fica sob `paper/`.

---

## 1. Objetivo do trabalho

Desenvolver um **sistema de reconhecimento de padrões** que, a partir das potências ópticas medidas por um array de FBGs, **classifique quais FBGs são as mais relevantes** para localizar o vale ressonante de um sensor LPFG.

Isso corresponde à **Parte 1** da ideia discutida com o Dr. Felipe Barino:

1. **Classificação** — selecionar as FBGs mais próximas do vale da LPG (máscara).
2. **Interrogação / regressão** — estimar \lambda_{res} (já feita no paper do Barino; opcional no seu artigo como discussão).

**Entregas da disciplina**


| Item         | Requisito                               |
| ------------ | --------------------------------------- |
| Artigo       | Template **IEEEtran** para conferências |
| Apresentação | 15 minutos em sala                      |


---

## 2. Estrutura sugerida de pastas

```
paper/
├── guia_paper.md                 ← este arquivo
├── fbg-demodulated-lpfg/         ← dados e código do Barino (fonte)
├── lpfg_demodulation_supplementary/  ← referência / paper TIM (não usar como fonte principal)
├── classification/               ← SEU trabalho experimental
│   ├── notebooks/
│   ├── figures/
│   └── results/
├── ieee/                         ← artigo LaTeX
│   ├── bare_conf.tex             ← (renomear p/ main.tex)
│   ├── references.bib
│   └── figures/
└── scripts/                      ← utilitários compartilhados (opcional)
```

Criar `classification/` e `ieee/` quando for começar cada etapa.

---

## 3. Fonte de dados (oficial)

Usar `**paper/fbg-demodulated-lpfg/**` (mais completo e atualizado).


| Arquivo                         | Uso                                                  |
| ------------------------------- | ---------------------------------------------------- |
| `data/measured.dataset`         | **Principal** — treino/teste do classificador        |
| `data/measured_spectra.dataset` | Espectros brutos (figuras / análise)                 |
| `data/source.npy`               | Fonte óptica (se regenerar sintéticos)               |
| `1 - Data generation.ipynb`     | Regenerar dados sintéticos (opcional, mais amostras) |
| `models/`                       | Modelo de regressão do Barino (comparação opcional)  |


### Formato de cada amostra (`measured.dataset`)


| Campo            | Dimensão | Papel                                  |
| ---------------- | -------- | -------------------------------------- |
| `input_strength` | 13       | **Entrada** X — potências normalizadas |
| `wl_bragg`       | 13       | Posições dos FBGs (nm)                 |
| `target`         | 1        | \lambda_{res} (nm) — gera o **rótulo** |


Pré-processamento de referência (notebook 4 do Barino):

- X \leftarrow X - \min(X); X \leftarrow X / \sum X (por amostra);
- manter apenas 1515 < \lambda_{res} < 1585.

---

## 4. Definição do problema de classificação

### 4.1 Heurística da máscara (rótulo)

Para cada amostra n, com \lambda_{res}^{(n)} e posições \lambda_{\mathrm{FBG},i}^{(n)}:


e_i = \lvert \lambda_{\mathrm{FBG},i} - \lambda_{res} \rvert


Selecionar os **k menores** e_i (no rascunho / conversa: k = 4).

Rótulo multi-rótulo y \in 0,1^{13}:

- y_i = 1 se o FBG i está entre os k mais próximos;
- y_i = 0 caso contrário.

### 4.2 Formulação recomendada


| Opção                            | Saída                      | Quando usar                     |
| -------------------------------- | -------------------------- | ------------------------------- |
| **A — Multi-rótulo (preferida)** | máscara 13 bits            | Alinhada à heurística do Barino |
| B — Multiclasse                  | índice do FBG mais próximo | Baseline simples                |
| C — Top-k como conjunto          | acerto do conjunto         | Métrica complementar            |


Começar pela **opção A** com k=4. Testar k \in 3,4,5 se houver tempo.

### 4.3 Entrada do classificador

- Mínimo: `input_strength` (13).
- Opcional: concatenar posições normalizadas dos FBGs (como no modelo do Barino).

---

## 5. Passos experimentais (classificador)

Ordem sugerida de notebooks em `classification/notebooks/`:

### Passo 0 — Setup

- [x] Ambiente Python (numpy, pandas, scikit-learn, matplotlib; imbalanced-learn se necessário).
- [x] Carregar `measured.dataset` e inspecionar n, distribuição de \lambda_{res}, exemplos de máscara.
- [x] Fixar `random_state` e documentar versões.

> Implementado em `classification/notebooks/0 - Setup.ipynb` (+ `classification/src/data_utils.py`).
> Achado inicial: após filtro restam **7300/8200** amostras; a máscara top-4 **não é uniforme** (FBGs nas bordas do array são raramente selecionadas).

### Passo 1 — Pré-processamento e rótulos

- [x] Aplicar normalização das potências.
- [x] Filtrar faixa 1515–1585 nm.
- [x] Gerar máscaras top-k a partir de `target` e `wl_bragg`.
- [x] Verificar balanceamento: cada FBG aparece como “relevante” com frequência semelhante?
- [x] Salvar X, y (máscara) e metadados em arquivo intermediário (`.npz` / `.pkl`).

> Implementado em `classification/notebooks/1 - Preprocess and labels.ipynb` (+ `scripts/run_passo1.py`).
> Artefato: `classification/results/prepared_measured_k4.npz` (+ `prepared_measured_k4_meta.json`, `passo1_mask_balance.csv`).
>
> **Achados (dados reais `measured.dataset`):**
>
> - Bruto: **8200×13**; após filtro aberto 1515 < \lambda_{res} < 1585: **7300** (removidas **900**: 400 abaixo, 500 acima).
> - \lambda_{res} filtrado fica em **~1516.56–1580.19** nm (não preenche até 1585).
> - `input_strength` bruto já tem **soma=1** por linha, mas **mínimo ≠ 0**; a normalização do Barino (min-subtract + re-soma) ainda altera X e deixa min≈0, soma≈1.
> - Máscara top-4: **exatamente 4 uns/linha**; **0 empates** no limiar do k-ésimo; coincide com `argpartition` e `argsort`.
> - Balanceamento **não uniforme** (esperado se \lambda_{res} concentra no meio): fração positiva de ~0.07 (FBG 0) a ~0.68 (FBG 4); FBG 12 ~0.014. Uniforme seria k/13 \approx 0.308.
> - Coerência espacial: correlação entre posição média do FBG e média de \lambda_{res} quando o FBG é positivo ≈ **0.992**.
>
> **Decisão:** manter a normalização do Barino (min-subtract + soma=1).

### Passo 2 — Metodologia de avaliação

- [x] Validação cruzada estratificada (cuidado: multi-rótulo — estratificar por \lambda_{res} em bins ou por padrão da máscara).
- [x] Alternativa robusta: `RepeatedStratifiedKFold` / split por faixas de \lambda_{res}.
- [x] Separar hold-out final **só** para relatório (não para ajuste fino).
- [ ] SMOTE / balanceamento: só se houver desbalanceamento claro **e** apenas no treino de cada fold.

> Implementado em `classification/notebooks/2 - Evaluation methodology.ipynb` (+ `src/cv_utils.py`, `scripts/run_passo2.py`).
> Artefato: `classification/results/cv_splits_passo2.npz` (+ meta/CSVs/figuras `passo2_*`).
>
> **Configuração (reprodutível, `random_state=42`):**
>
> - Hold-out **20%** → **1460** amostras; desenvolvimento **5840**.
> - Estratificação: **10 quantis de \lambda_{res}** (há também 10 máscaras únicas no dataset; a chave usada foi \lambda_{res}, não o padrão da máscara).
> - **A — `StratifiedKFold`:** 5 folds (1168 teste / 4672 treino cada).
> - **B — `RepeatedStratifiedKFold`:** 5 folds × 5 repetições = **25** avaliações.
>
> **Coerência (dados reais):**
>
> - Hold-out sem interseção com folds; cobertura total 7300.
> - Fração positiva por FBG no hold-out ≈ global (diferença máxima ~0.01).
> - \mathbb{E}[\lambda_{res}] no teste estável (~1541.54 nm); std entre folds: A≈0.069, B≈0.053.
> - FBG 12 (mais raro): ≥12 positivos no teste em A; em B min=10, mediana=17.
> - SMOTE **não** aplicado neste passo (só splits).
>
> **Decisão:** estratégia **A** (`StratifiedKFold`, 5 folds) para os passos seguintes. Hold-out de 20% permanece isolado para o relatório final. Estratégia B fica documentada como alternativa não selecionada.

### Passo 3 — Classificadores

Comparar pelo menos os da disciplina (adaptados a multi-rótulo quando preciso):


| commit e pushMétodo   | Observação                                |
| --------------------- | ----------------------------------------- |
| kNN                   | Distância em 13D; sensível à escala       |
| SVM                   | One-vs-rest ou multi-output               |
| MLP                   | Saída sigmoid 13 + BCE                    |
| Random Forest         | `MultiOutputClassifier` ou RF multi-label |
| AdaBoost              | Via one-vs-rest                           |
| MQ / regressão linear | Baseline linear (limiar por canal)        |


Opcional: baseline “ingênuo” (sempre os 4 FBGs de maior potência).

> Implementado em `classification/notebooks/3 - Classifiers.ipynb` (+ `src/classifiers.py`, `src/metrics_utils.py`, `scripts/run_passo3.py`).
> Artefatos: `results/passo3_summary.csv`, `passo3_fold_metrics.csv`, `figures/passo3_metrics_comparison.png`.
>
> **Protocolo:** CV estratégia A (5 folds); hold-out intacto; predição **top-4 por score**; sem SMOTE; defaults fixos (tuning no Passo 4).
> Escala `StandardScaler` (fit só no treino) para kNN, SVM, MLP, MQ.
>
> **Resultados reais (média ± std, Jaccard samples):**
>
>
> | Método       | Jaccard           | Exact match | F1 micro | Hamming |
> | ------------ | ----------------- | ----------- | -------- | ------- |
> | RandomForest | **0.974 ± 0.002** | 0.934       | 0.983    | 0.010   |
> | AdaBoost     | 0.971 ± 0.004     | 0.928       | 0.982    | 0.011   |
> | MLP          | 0.963 ± 0.005     | 0.908       | 0.977    | 0.014   |
> | kNN          | 0.963 ± 0.003     | 0.907       | 0.977    | 0.014   |
> | SVM          | 0.961 ± 0.002     | 0.902       | 0.975    | 0.015   |
> | MQ           | 0.922 ± 0.005     | 0.805       | 0.951    | 0.030   |
>
>
> Com máscaras de cardinalidade fixa k=4, F1 micro = set recall (esperado).

### Passo 4 — Ajuste de hiperparâmetros

- [x] Grid/Random search **dentro** da CV (nested CV se possível).
- [x] Não usar o conjunto de teste final para escolher k, C, profundidade, etc.

> Implementado em `classification/notebooks/4 - Hyperparameter tuning.ipynb` (+ `src/tuning.py`, `scripts/run_passo4.py`).
> Artefatos: `passo4_summary.csv`, `passo4_best_params.csv`, `figures/passo4_vs_passo3_jaccard.png`.
>
> **Protocolo:** nested CV — outer = estratégia A (5); inner = `GridSearchCV` 3-fold estratificado por bins de \lambda_{res} **apenas no treino**; scorer = Jaccard top-4; hold-out intacto.
> MQ no Passo 4: busca \alpha de **Ridge** (OLS do Passo 3 não tem hiperparâmetro).
>
> **Resultados (Jaccard samples, teste externo):**
>
>
> | Método       | Passo 4           | Passo 3 | Δ          |
> | ------------ | ----------------- | ------- | ---------- |
> | **SVM**      | **0.974 ± 0.003** | 0.961   | **+0.013** |
> | RandomForest | 0.973 ± 0.002     | 0.974   | −0.001     |
> | AdaBoost     | 0.972 ± 0.004     | 0.971   | +0.001     |
> | MLP          | 0.965 ± 0.006     | 0.963   | +0.001     |
> | kNN          | 0.964 ± 0.002     | 0.963   | +0.001     |
> | MQ           | 0.922 ± 0.005     | 0.922   | ~0         |
>
>
> **Consenso estável entre folds:** SVM sempre `C=10`, `gamma=0.1`. AdaBoost tende a `n_estimators=100`, `learning_rate=1.0`. Demais métodos variam mais entre folds.

### Passo 5 — Métricas e análise

Reportar (média ± desvio entre folds):

**Por canal / multi-rótulo**

- [x] Hamming loss
- [x] Precisão, revocação, F1 (micro e macro)
- [x] Exact match ratio (máscara inteira correta)
- [x] Jaccard / IoU do conjunto top-k

**Por amostra**

- [x] Fração dos k FBGs verdadeiros recuperados (recall do conjunto)
- [x] Matriz de confusão agregada por canal (ou heatmap 13×2)

**Análise crítica (importante para o artigo)**

- [x] Erros vs \lambda_{res} (bordas do espectro pioram?).
- [x] Comparar k=3,4,5.
- [x] Com vs sem posições FBG na entrada.
- [ ] (Opcional) Impacto na regressão do Barino: \lambda_{res} só com FBGs mascaradas vs todas.

> Implementado em `classification/notebooks/5 - Metrics and analysis.ipynb` (+ `scripts/run_passo5.py`, `src/tuned_models.py`).
> Hold-out **não** usado. Modelos = consenso do Passo 4.
>
> **k=4, só potências (ranking Jaccard):**
> | Método | Jaccard | Exact | F1 micro | Hamming |
> |--------|---------|-------|----------|---------|
> | **SVM** | **0.974 ± 0.003** | 0.935 | 0.984 | 0.010 |
> | RandomForest | 0.973 ± 0.002 | 0.933 | 0.983 | 0.010 |
> | AdaBoost | 0.973 ± 0.004 | 0.932 | 0.983 | 0.011 |
> | MLP | 0.965 ± 0.006 | 0.912 | 0.978 | 0.014 |
> | kNN | 0.964 ± 0.002 | 0.910 | 0.978 | 0.014 |
> | MQ | 0.922 ± 0.005 | 0.805 | 0.951 | 0.030 |
>
> **Erros vs \(\lambda_{res}\):** para o SVM, o pior bin **não** é a borda — é ~1553–1564 nm (Jaccard ≈ 0.91); bordas e centro ficam ≥ 0.97. Hipótese “bordas pioram” **não** se confirma nestes dados.
>
> **k ∈ {3,4,5}:** SVM lidera em todos; Jaccard mais alto em k=3 (0.985), depois k=5 (0.978), k=4 (0.974). k=4 permanece o alvo do problema (heurística Barino).
>
> **+ `wl_bragg` na entrada:** melhora RF/AdaBoost/MLP/MQ (~+0.008 a +0.015); SVM quase neutro (+0.0004); **kNN piora** (−0.022). Com posições, RF passa a ~0.981.
>
> Impacto na regressão do Barino: **não executado** (opcional).

### Passo 6 — Figuras para o artigo

- [x] Esquema LPG + 13 FBGs + máscara (como no rascunho).
- [x] Exemplo de espectro / potências com máscara verdadeira vs prevista.
- [x] Barras ou tabela de métricas por classificador.
- [x] Heatmap de erros por índice de FBG.
- [ ] (Opcional) Curvas ROC por canal ou precision-recall.

> Implementado em `classification/notebooks/6 - Paper figures.ipynb` (+ `scripts/run_passo6.py`).
> Figuras de artigo em **`paper/ieee/figures/`** (cópias também em `classification/figures/`):
>
> | Arquivo | Uso sugerido no IEEE |
> |---------|----------------------|
> | `fig_schema_lpg_fbg.png` | Fig. 1 — problema / máscara top-4 |
> | `fig_mask_true_vs_pred.png` | Fig. 2 — exemplo real (SVM) |
> | `fig_metrics_classifiers.png` | Fig. 3 — comparação dos 6 métodos |
> | `fig_error_heatmap_fbg.png` | Fig. 4 — erro por canal |
> | `fig_jaccard_vs_lambda.png` | Fig. 5 — desempenho vs \(\lambda_{res}\) |
>
> ROC/PR: **não** gerado (opcional). Próximo passo natural: montar `paper/ieee/main.tex`.

---

## 6. Checklist de qualidade experimental

Antes de escrever o artigo, garantir:

- [ ] Rótulos gerados **só** a partir de \lambda_{res} e \lambda_{\mathrm{FBG}} (sem vazamento do teste).
- [ ] Pré-processamento ajustado **só no treino** de cada fold (scaler, se houver).
- [ ] Mais de uma configuração testada (não um único run “mágico”).
- [ ] Desvios entre folds reportados.
- [ ] Conclusão alinhada à métrica certa (ex.: Jaccard do top-4, não só acurácia exata).
- [ ] Reprodutibilidade: semente, versões, caminho dos dados.

---

## 7. Artigo IEEE (conferência)

### 7.1 Baixar o template

Opções oficiais:

1. **IEEE Template Selector:** [https://template-selector.ieee.org/](https://template-selector.ieee.org/)
  → Conference paper → LaTeX.
2. **CTAN IEEEtran:** [https://ctan.org/pkg/ieeetran](https://ctan.org/pkg/ieeetran)
  → usar `bare_conf.tex` como ponto de partida.
3. **Overleaf:** template “IEEE Conference Paper”.

Classe típica:

```latex
\documentclass[conference]{IEEEtran}
```

Colocar o projeto em `paper/ieee/` (ex.: `main.tex` + `references.bib` + `figures/`).

### 7.2 Estrutura sugerida do artigo

1. **Title / Authors / Abstract / Keywords**
2. **Introduction** — LPFG, custo do OSA, array FBG, contribuição (classificação das FBGs relevantes).
3. **Related work** — Barino (demodulação / self-attention); interrogação óptica; classificação multi-rótulo.
4. **Problem ~~~~formulation** — sinal 13D; heurística da máscara top-k; notação.
5. **Methodology** — pré-processamento; classificadores; CV; métricas.
6. **Experiments and results** — dataset; tabelas; figuras; análise de erros.
7. **Discussion** — ligação com a Parte 2 (interrogação); limitações.
8. **Conclusion**
9. **References**

Tamanho típico de conferência: ~4–6 páginas (confirmar com o professor se houver limite).

### 7.3 Apresentação (15 min)

Roteiro sugerido (~1–2 min por bloco):

1. Motivação e problema físico
2. Ideia em duas partes (classificar → interrogar)
3. Dataset e geração da máscara
4. Métodos e métricas
5. Resultados principais (1–2 figuras fortes)
6. Conclusão e próximos passos

---

## 8. Bibliografia (a revisar e expandir)

### Obrigatórias / centrais

- Barino, F. O.; Dos Santos, A. B. — paper TIM / demodulação LPFG + FBG + self-attention  
DOI: [https://doi.org/10.1109/TIM.2025.3573014](https://doi.org/10.1109/TIM.2025.3573014)  
- Material / tese: repositório UFJF e Zenodo do suplementar (citar o que for usado).
- Código/dados: `fbg-demodulated-lpfg` (autorização de uso — mencionar no artigo).

### Temas a complementar (revisar e escolher 6–12 boas referências)


| Tema                            | Exemplos de busca                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------- |
| Sensores LPFG / FBG             | demodulação, interrogação, sensor networks                                            |
| Aprendizado em sensores ópticos | neural networks fiber Bragg, machine learning optical sensing                         |
| Classificação multi-rótulo      | multi-label classification survey (Tsoumakas, Zhang & Zhou, etc.)                     |
| Atenção / feature selection     | attention mechanisms, feature selection for classification                            |
| Avaliação                       | cross-validation, Hamming loss, multi-label metrics                                   |
| Métodos clássicos               | kNN, SVM, Random Forest (livros/papers canônicos ou scikit-learn docs com parcimônia) |


### Boas práticas BibTeX

- [ ] Um arquivo `references.bib` único.
- [ ] DOIs em todas as entradas possíveis.
- [ ] Estilo IEEE (`\bibliographystyle{IEEEtran}`).
- [ ] Citar o que foi **lido e usado**; evitar lista inflada.
- [ ] Revisar grafia de nomes (Barino, LPFG, FBG) e anos.

---

## 9. Cronograma sugerido


| Fase  | Atividade                                                          | Saída                   |
| ----- | ------------------------------------------------------------------ | ----------------------- |
| **A** | Explorar `measured.dataset`, gerar máscaras, figuras exploratórias | Notebook 0–1            |
| **B** | Pipeline CV + baselines + 3–6 classificadores                      | Notebooks 2–3 + tabelas |
| **C** | Análise crítica, ablações (k, features), figuras finais            | `results/` + `figures/` |
| **D** | Baixar IEEEtran, esqueleto do artigo, bib inicial                  | `ieee/main.tex`         |
| **E** | Escrever resultados + discussão; revisar bib                       | Draft completo          |
| **F** | Polir artigo + slides 15 min                                       | PDF + apresentação      |


---

## 10. Ordem de execução (resumo)

```
1. Dados (fbg-demodulated-lpfg) → máscaras top-k
2. Classificadores + CV + métricas multi-rótulo
3. Análise / figuras
4. Template IEEE + bib (Barino + referências revisadas)
5. Artigo + apresentação 15 min
```

---

## 11. Referências rápidas do repositório

- Dados e código do Barino: `paper/fbg-demodulated-lpfg/`
- Suplementar TIM (contexto): `paper/lpfg_demodulation_supplementary/`
- Heurística: máscara dos k FBGs com menor |\lambda_{\mathrm{FBG},i}-\lambda_{res}|
- Enunciado: sistema de RP + artigo IEEEtran + 15 min

---

*Documento vivo: atualizar este guia conforme decisões experimentais (valor de k, métrica principal, título do artigo) forem fechadas.*