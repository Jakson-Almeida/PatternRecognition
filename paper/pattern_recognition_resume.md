# Resumo — Reconhecimento de Padrões (Trabalho Final)

Formulação oficial: **multiclasse com 10 classes** (janelas contíguas de 4 FBGs). Números de `paper/classification/results/`.

---

## 1. Objetivo

A partir das potências de 13 FBGs, classificar a **janela** \(C_s=\{s,s+1,s+2,s+3\}\), \(s\in\{0,\ldots,9\}\), alinhada às FBGs mais próximas de \(\lambda_{res}\).

Parte 1 (este trabalho): classificação. Parte 2 (Barino): regressão de \(\lambda_{res}\) (não reimplementada).

---

## 2. Dados

| Item | Valor |
|------|--------|
| Fonte | `measured.dataset` (autorizado) |
| Bruto → filtrado | 8200 → **7300** (\(1515<\lambda_{res}<1585\)) |
| \(X\) | 13 potências (min-subtract + soma=1) |
| \(y\) | `y_class` ∈ {0…9}; `y_mask` equivalente |

**Contagens:** C0=520, C1=1257, C2=2089, C3=89, C4=1501, C5=542, C6=236, C7=696, C8=270, C9=100.

---

## 3. Avaliação

- Hold-out 20% (**1460**), estratificado por **classe**
- Dev **5840**; StratifiedKFold 5; nested CV (accuracy)
- Métricas: acurácia, F1 weighted/macro, CM \(10\times10\)

---

## 4. Resultados (afinados)

**CV (média):** SVM 0,934 · RF 0,929 · MLP 0,919 · AdaBoost 0,924 · kNN 0,907 · MQ 0,779

**Hold-out:** SVM/RF **0,938** · MLP 0,933 · AdaBoost **0,923** · kNN 0,919 · MQ 0,789

**+ wl_bragg:** ajuda MLP/RF/MQ; prejudica kNN (−0,05); SVM quase neutro.

---

## 5. Conclusão

Problema bem posto como 10 classes; SVM/RF melhores. Próximo passo: impacto na regressão de \(\lambda_{res}\).
