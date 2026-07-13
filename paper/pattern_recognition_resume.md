# Resumo — Reconhecimento de Padrões (Trabalho Final)

Resumo do que foi feito no experimento de classificação de FBGs relevantes para demodulação de LPFG. Números abaixo vêm dos artefatos reais em `paper/classification/results/` (não inventados).

---

## 1. Objetivo

Construir um sistema de reconhecimento de padrões que, a partir das **potências ópticas de 13 FBGs**, classifique **quais FBGs são as mais relevantes** para localizar o vale ressonante \(\lambda_{res}\) de um sensor LPFG.

Isso é a **Parte 1** da ideia discutida com o Dr. Felipe Barino:

1. **Classificação** — máscara das FBGs próximas de \(\lambda_{res}\) (foco deste trabalho).
2. **Interrogação / regressão** — estimar \(\lambda_{res}\) (já feita no paper do Barino; não reimplementada aqui).

**Formulação:** classificação **multi-rótulo** com saída \(y \in \{0,1\}^{13}\).

---

## 2. Banco de dados

### Fonte

| Item | Valor |
|------|--------|
| Arquivo | `paper/fbg-demodulated-lpfg/data/measured.dataset` |
| Origem | Dados reais autorizados (trabalho do Barino / TIM) |
| Formato | dicionário pickle |

### Campos por amostra

| Campo | Shape | Papel |
|-------|-------|--------|
| `input_strength` | 13 | Entrada \(X\) — potências ópticas |
| `wl_bragg` | 13 | Posições dos FBGs (nm) — gera o rótulo |
| `target` | 1 | \(\lambda_{res}\) (nm) — gera o rótulo |

### Tamanho e filtro

| Etapa | \(n\) |
|-------|------|
| Bruto | **8200** |
| Após filtro \(1515 < \lambda_{res} < 1585\) nm | **7300** |
| Removidas | **900** (400 abaixo, 500 acima) |
| \(\lambda_{res}\) filtrado (faixa observada) | ~1516,56 – 1580,19 nm |

### Pré-processamento de \(X\)

Normalização do notebook 4 do Barino (mantida por decisão experimental):

1. subtrair o mínimo por amostra;
2. dividir pela soma (soma ≈ 1).

Observação: o bruto já tinha soma = 1, mas mínimo ≠ 0; a normalização ainda altera \(X\).

### Rótulo (máscara top-\(k\))

Para cada amostra:

\[
e_i = \lvert \lambda_{\mathrm{FBG},i} - \lambda_{res} \rvert
\]

Selecionar os \(k\) menores \(e_i\) → \(y_i = 1\) nesses canais.

- \(k = 4\) (configuração principal).
- Sempre exatamente 4 uns por linha; 0 empates no limiar.
- **10** padrões de máscara únicos no dataset filtrado.

### Balanceamento da máscara (\(k=4\))

Não é uniforme (esperado se \(\lambda_{res}\) concentra no meio). Fração positiva por FBG:

| FBG | Fração ≈ |
|-----|----------|
| 0 | 0,071 |
| 4 (pico) | 0,676 |
| 12 | 0,014 |
| Uniforme seria \(4/13\) | ≈ 0,308 |

Coerência espacial: correlação entre posição média do FBG e média de \(\lambda_{res}\) quando o FBG é positivo ≈ **0,992**.

Artefato: `classification/results/prepared_measured_k4.npz`.

---

## 3. Metodologia de avaliação

| Item | Escolha |
|------|---------|
| Semente | `random_state = 42` |
| Hold-out | **20%** estratificado por quantis de \(\lambda_{res}\) → **1460** amostras (só relatório final) |
| Desenvolvimento | **5840** amostras |
| Estratificação | 10 quantis de \(\lambda_{res}\) |
| CV escolhida | **Estratégia A** — `StratifiedKFold`, **5** folds |
| Alternativa testada (não usada depois) | Estratégia B — `RepeatedStratifiedKFold` (5×5) |
| Predição | **top-\(k\) por score** (sempre \(k\) positivos) |
| Métrica principal | Jaccard por amostra |
| SMOTE | **Não** aplicado |

Hold-out nunca entrou em ajuste de hiperparâmetros.

Artefato de splits: `classification/results/cv_splits_passo2.npz`.

---

## 4. Classificadores

Seis métodos da disciplina, adaptados a multi-rótulo:

| Método | Adaptação |
|--------|-----------|
| **kNN** | multilabel nativo + `StandardScaler` |
| **SVM** | One-vs-Rest, kernel RBF + scaler |
| **MLP** | multilabel + scaler, early stopping |
| **Random Forest** | multilabel nativo |
| **AdaBoost** | One-vs-Rest |
| **MQ** | regressão linear / Ridge multi-saída + top-\(k\) nos valores |

### Hiperparâmetros (Passo 4 — nested CV)

Busca **dentro** do treino de cada fold externo (`GridSearchCV`, 3 folds internos, scorer = Jaccard top-\(k\)).

Consenso estável usado depois:

| Método | Configuração de consenso |
|--------|--------------------------|
| SVM | \(C=10\), \(\gamma=0{,}1\) (**todos** os folds) |
| kNN | \(k_{\mathrm{NN}}=11\), `weights=distance` |
| MLP | hidden \((128,64)\), \(\alpha=10^{-4}\) |
| Random Forest | \(n_{\mathrm{estimators}}=200\), `max_depth=None` |
| AdaBoost | \(n_{\mathrm{estimators}}=100\), `learning_rate=1{,}0` |
| MQ | Ridge com \(\alpha=0{,}01\) (no Passo 3 era OLS) |

Ganho principal do tuning: **SVM** Jaccard \(0{,}961 \rightarrow 0{,}974\) (+0,013). Demais métodos mudaram pouco.

---

## 5. Resultados

### 5.1 CV estratégia A — modelos afinados (\(k=4\), só potências)

Média ± desvio entre 5 folds:

| Método | Jaccard | Exact match | F1 micro | Hamming |
|--------|---------|-------------|----------|---------|
| **SVM** | **0,974 ± 0,003** | 0,935 | 0,984 | 0,010 |
| Random Forest | 0,973 ± 0,002 | 0,933 | 0,983 | 0,010 |
| AdaBoost | 0,973 ± 0,004 | 0,932 | 0,983 | 0,011 |
| MLP | 0,965 ± 0,006 | 0,912 | 0,978 | 0,014 |
| kNN | 0,964 ± 0,002 | 0,910 | 0,978 | 0,014 |
| MQ | 0,922 ± 0,005 | 0,805 | 0,951 | 0,030 |

Com cardinalidade fixa \(k=4\), F1 micro = set recall (esperado).

### 5.2 Hold-out (avaliação única, pós-tuning)

Treino no desenvolvimento completo (5840); teste no hold-out (1460):

| Método | Jaccard | Exact match | F1 micro |
|--------|---------|-------------|----------|
| **SVM** | **0,975** | 0,938 | 0,985 |
| Random Forest | 0,974 | 0,936 | 0,984 |
| MLP | 0,974 | 0,934 | 0,984 |
| AdaBoost | 0,973 | 0,934 | 0,983 |
| kNN | 0,967 | 0,918 | 0,979 |
| MQ | 0,933 | 0,832 | 0,958 |

A ordem do ranking se mantém: SVM lidera também no hold-out.

### 5.3 Análises críticas (Passo 5)

**Erros vs \(\lambda_{res}\)**  
A hipótese “bordas do espectro pioram” **não** se confirma. Para o SVM, o pior bin fica em torno de **1553–1564 nm** (Jaccard ≈ 0,91); bordas e centro ficam ≥ 0,97.

**Comparação \(k \in \{3,4,5\}\)**  
SVM lidera em todos. Jaccard mais alto em \(k=3\) (≈ 0,985) — problema mais fácil. Mantém-se \(k=4\) pela heurística do problema.

**Entrada: só potências vs potências + `wl_bragg`** (\(k=4\)):

| Método | Δ Jaccard ao adicionar posições |
|--------|----------------------------------|
| MQ | +0,015 |
| MLP | +0,013 |
| AdaBoost | +0,008 |
| Random Forest | +0,008 |
| SVM | +0,0004 (quase neutro) |
| kNN | **−0,022** (piora) |

Com posições, RF sobe a ≈ 0,981 na CV.

**Impacto na regressão do Barino:** não executado (opcional).

---

## 6. Artigo e figuras

- Rascunho IEEE em português: `paper/ieee/main.tex` (≈ **3 páginas**; limite ≤ 6).
- Figuras principais em `paper/ieee/figures/`:
  - esquema LPFG + 13 FBGs + máscara;
  - exemplo real máscara verdadeira vs SVM;
  - barras de métricas;
  - heatmap de erro por FBG;
  - Jaccard vs \(\lambda_{res}\).

---

## 7. Conclusão experimental (em uma frase)

Com dados reais de 13 FBGs, máscara top-4 e avaliação cuidadosa (CV + hold-out isolado), o **SVM RBF one-vs-rest** foi o melhor classificador (Jaccard ≈ **0,975** no hold-out); o MQ ficou como baseline linear mais fraco.

---

## 8. Onde achar os artefatos

| Conteúdo | Caminho |
|----------|---------|
| Dados fonte | `paper/fbg-demodulated-lpfg/data/measured.dataset` |
| Dataset preparado | `paper/classification/results/prepared_measured_k4.npz` |
| Splits CV / hold-out | `paper/classification/results/cv_splits_passo2.npz` |
| Resumo CV afinada | `paper/classification/results/passo5_summary_k4.csv` |
| Hold-out | `paper/classification/results/passo7_holdout_metrics.csv` |
| Notebooks | `paper/classification/notebooks/` |
| Artigo | `paper/ieee/main.tex` / `main.pdf` |
| Guia detalhado | `paper/guia_paper.md` |

---

*Documento de resumo do experimento de reconhecimento de padrões. Atualizar se novos resultados (bibliografia final, impacto na regressão, etc.) forem incorporados.*
