# Plano experimental — otimização de hiperparâmetros e teste cego (TCC II)

**Registrado em 2026-10-29, antes de qualquer execução do teste cego.** O commit que introduz este arquivo é o
pré-registro: tudo o que está aqui foi decidido sem conhecer nenhum resultado do conjunto cego. Depois do teste
cego do braço 0 (passo 2 da seção 10), este plano não muda — qualquer alteração vai para o
[Registro de desvios](#12-registro-de-desvios), com data e motivo, e só é aceita se não tiver sido motivada por
resultado do teste cego.

## 1. Objetivo

Responder duas perguntas, medindo a generalização para **proteínas novas** (teste cego):

1. A otimização de hiperparâmetros melhora o desempenho em relação à linha de base?
2. Escolher os hiperparâmetros com uma validação cruzada **separada por gene** (homólogos nunca divididos entre
   treino e validação) generaliza melhor do que com a validação atual, de **genes misturados**?

A validação cruzada atual mede principalmente o reconhecimento de homólogos de genes do treino (CLAUDE.md §4.1);
o teste cego é separado por gene por construção. A pergunta 2 testa se otimizar para "reconhecer o que já viu"
leva a escolhas piores para proteínas novas.

## 2. O que está fixo

| Item | Valor |
|---|---|
| Treino | 632 sequências: 316 positivas (96 genes, `data/raw_v2/`) e 316 negativas (111 genes, `data/raw_negativos_v2/`) |
| Teste cego | 294 sequências: 147 positivas (46 genes dos loci *Newly reported* de Dand et al., 2025, `data/teste_cego_v2/`) e 147 negativas (56 genes, `data/teste_cego_negativos_v2/`). Nunca usado em nenhuma escolha |
| Representação | k=3 contínuo, projeção aleatória gaussiana com **N=4000** (decisão de 2026-10-29, CLAUDE.md §6 "Decisão de representação") |
| Sequências idênticas entre espécies | Mantidas no treino (decisão do orientador, CLAUDE.md §6 Pendências 4) |
| Semente | `RANDOM_STATE = 42` |
| Versões | `requirements.txt` (alinhado em 2026-10-29 às versões que geraram todos os resultados) |
| Métrica principal | **MCC**; ROC-AUC como secundária (saturou em ~99,7 na validação cruzada e não diferencia os modelos) |

## 3. Os três braços

| Braço | Tag no `08_blind_test.py` | Como os hiperparâmetros são escolhidos | Modelos |
|---|---|---|---|
| **0 — Linha de base** | `antes` | `GridSearchCV` com os grids da linha de base, `StratifiedKFold` 10 folds, refit por ROC-AUC (os `results_*_n4000.json` já existentes) | LR, SVM, RF, GB, NB, kNN |
| **A — Otimizado, genes misturados** | `otimizado_misturado` | Busca bayesiana (seção 5) com a validação da seção 4-A | LR, SVM, árvores, boosting, MLP e comitê |
| **B — Otimizado, separado por gene** | `otimizado_por_gene` | A mesma busca, com a validação da seção 4-B | idem |

Os braços A e B são **idênticos** — espaço de busca, orçamento, sementes, modelos, limiar, comitê — exceto pela
forma de dividir os folds. Qualquer diferença entre eles no teste cego vem dessa escolha.

NB e kNN não são otimizados (ficam como linha de base, como no plano de 2026-10-20) e aparecem só no braço 0.

## 4. Validação cruzada dos braços A e B

- **A — genes misturados:** `RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)`.
- **B — separado por gene:** 5 repetições de `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42+r)`,
  r = 0…4. Grupo = gene: nome do arquivo FASTA para os positivos; para os negativos, a parte antes do primeiro `_`
  (ex.: `ACTB_housekeeping_facil` → `ACTB`). Todos os homólogos de um gene — inclusive os ortólogos idênticos —
  caem sempre no mesmo fold.
- **Função objetivo:** MCC médio nas 50 avaliações (5 repetições × 10 folds), com limiar 0,5 durante a busca.
  Também são registrados ROC-AUC, acurácia, precisão, sensibilidade, especificidade e F1.

## 5. Busca de hiperparâmetros

- **Método:** otimização bayesiana com o algoritmo TPE (biblioteca Optuna), `TPESampler(seed=42)`.
- **Sem poda** (*pruning*): toda tentativa é avaliada nas 50 divisões completas. As tentativas rodam em sequência
  (para a busca ser reprodutível) e as 50 avaliações de cada tentativa rodam em paralelo.
- **Orçamento (número de tentativas por modelo e por braço):** LR 50 · SVM 150 · árvores 150 · boosting 100 · MLP 100.
- **Regra de borda:** se, ao fim do orçamento, o melhor valor de um hiperparâmetro numérico estiver nos 5% extremos
  da faixa (medidos em escala logarítmica quando a faixa é logarítmica), a faixa é ampliada naquela direção — 10×
  para faixas logarítmicas, até o limite válido para as demais — e a busca continua com mais 50 tentativas.
  No máximo 2 ampliações por modelo e braço.

### Espaços de busca (p = 4000 variáveis)

| Modelo | Escolhas estruturais | Faixas numéricas |
|---|---|---|
| **LR** (`StandardScaler` + `LogisticRegression`, solver saga, `max_iter=5000`) | regularização ∈ {L2, L1, elastic net} | C ∈ [10⁻⁵, 10³] (log); proporção L1 do elastic net ∈ [0,05; 0,95] |
| **SVM** (`StandardScaler` + `SVC`, `probability=True`) | kernel ∈ {RBF, linear} | C ∈ [2⁻⁵, 2¹⁵] (log); gamma ∈ [10⁻³/p, 10²/p] (log, só RBF) |
| **Árvores** | tipo ∈ {Random Forest, Extra Trees}; 1000 árvores (fixo); `bootstrap=True` | variáveis por divisão ∈ [1, 2000] (log, inteiro); amostras mínimas por folha ∈ [1, 10]; fração da amostra por árvore ∈ [0,2; 1,0] |
| **Boosting** | tipo ∈ {Gradient Boosting, HistGradientBoosting} | **GB:** estágios ∈ [100, 2000]; taxa de aprendizado ∈ [0,005; 0,5] (log); profundidade ∈ [1, 6]; subamostragem ∈ [0,5; 1,0]. **HistGB:** iterações ∈ [100, 2000]; taxa ∈ [0,005; 0,5] (log); folhas máximas ∈ [4, 64] (log); amostras mínimas por folha ∈ [5, 50]; regularização L2 ∈ [10⁻⁶, 10] (log); fração de variáveis ∈ [0,05; 1,0]; sem parada antecipada |
| **MLP** (`StandardScaler` + `MLPClassifier`, ReLU, Adam, parada antecipada com 10% de validação, `max_iter=500`) | camadas ocultas ∈ {1, 2}; lote ∈ {32, 64, 128} | neurônios por camada ∈ [32, 1024] (log, inteiro); regularização alpha ∈ [10⁻⁶, 10⁻¹] (log); taxa de aprendizado inicial ∈ [10⁻⁴, 10⁻²] (log) |

**De onde vêm as faixas:** valores padrão do scikit-learn, ampliados em ordens de grandeza para os dois lados;
recomendações da literatura — Hsu, Chang & Lin, *A Practical Guide to Support Vector Classification* (C e gamma em
escala exponencial); Probst, Wright & Boulesteix (2019), *Hyperparameters and tuning strategies for random forest*
(número de árvores não é otimizado, basta ser alto; variáveis por divisão é o hiperparâmetro mais importante);
Friedman (2001), *Greedy function approximation: a gradient boosting machine* (troca entre taxa de aprendizado e
número de estágios) — e os vencedores da linha de base com N=4000, que ficam dentro das faixas.
Referências do método: Akiba et al. (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework*
(KDD); Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization* (TPE). Métrica: Chicco & Jurman (2020),
*BMC Genomics*, sobre o MCC. **Conferir as referências antes de citar.**

## 6. Comitê (*ensemble*)

Em cada braço, média simples das probabilidades (*soft voting*, pesos iguais) dos cinco modelos otimizados do
braço: LR, SVM, árvores, boosting e MLP. Sem otimização própria além do limiar (seção 7).

## 7. Limiar de decisão

Fixados os hiperparâmetros, o limiar t ∈ {0,05; 0,06; …; 0,95} é escolhido para maximizar o MCC sobre as
probabilidades fora do fold (*out-of-fold*) da mesma validação 5×10 do braço (média do MCC entre as 5 repetições).
No teste cego, os braços A e B são reportados com o **limiar ajustado (principal)** e também com **0,5**, para
comparação direta com o braço 0.

## 8. Modelo principal de cada braço

Regra aprovada em 2026-10-29: **maior MCC médio na validação cruzada do próprio braço; desempate pelo ROC-AUC.**
No braço 0, a validação é a `StratifiedKFold` 10 folds da linha de base (hoje: SVM, MCC 95,3). O comitê concorre
como mais um modelo nos braços A e B.

## 9. Teste cego e regra de reporte

- `08_blind_test.py --tag {antes | otimizado_misturado | otimizado_por_gene} --n-components 4000`, **uma vez por braço**.
- Métricas por sequência e por gene (média das probabilidades entre os homólogos do gene); MCC principal, ROC-AUC,
  sensibilidade, especificidade; sensibilidade por nível de evidência do locus (A–D) e especificidade por nível
  de dificuldade do negativo (fácil, intermediário, difícil).
- **Os três braços e todos os modelos são reportados.** O resultado do TCC é a **comparação** entre os braços.
  Não se escolhe, depois do teste cego, o melhor braço ou modelo para apresentá-lo como estimativa não enviesada
  do desempenho; se ele for citado, vem com a ressalva de que foi escolhido olhando o teste.

## 10. Ordem de execução

1. Commit deste plano (pré-registro).
2. Braço 0: `06_train_final.py` para os 6 modelos com k=3, N=4000 → `08_blind_test.py --tag antes`.
   **A partir daqui o plano está travado.**
3. Implementação da otimização dos braços A e B; testes locais sem tocar no conjunto cego.
4. Rodada-piloto na VM (~10 tentativas por modelo e braço) → custo projetado → aprovação do grupo.
5. Rodada completa dos braços A e B → modelos finais → teste cego de cada braço.
6. Tabelas, figuras (ex.: MCC ao longo das tentativas) e texto.

## 11. Infraestrutura e reprodutibilidade

- **VM:** Azure (assinatura Sponsorship), East US, `Standard_F32als_v7` — 32 núcleos físicos AMD, 64 GB, US$ 1,94/h —,
  num grupo de recursos próprio (`tcc-otimizacao`), apagado ao fim. Estimativa preliminar para os dois braços:
  ~US$ 100–200; a rodada-piloto refina o número antes da rodada completa.
- Cada resultado registra o commit do código, as versões das bibliotecas, as sementes e o histórico completo de
  tentativas da busca.

**Terminologia:** o que se faz aqui é **otimização de hiperparâmetros**, não *fine tuning*. *Fine tuning* é
continuar o treino de um modelo pré-treinado; a metodologia usa uma representação fixa (k-mers e projeção
aleatória) e classificadores treinados do zero, então não há modelo pré-treinado a ajustar.

## 12. Registro de desvios

| Data | Desvio | Motivo |
|---|---|---|
| 2026-10-29 | **Regressão Logística:** o parâmetro `penalty` está obsoleto no scikit-learn 1.8; os três tipos de regularização passam a ser expressos por `l1_ratio` (L2 = 0, L1 = 1, elastic net ∈ [0,05; 0,95]), com solver saga. Mesmo modelo, mesma faixa de C. | Mudança de API da biblioteca. Esclarecimento de implementação, antes de qualquer resultado dos braços A e B. |
| 2026-10-29 | **Regra de borda em faixas lineares:** "até o limite válido" implementado como ampliar uma largura da faixa naquela direção, sem passar do limite válido do parâmetro (ex.: subamostragem ≤ 1; amostras por folha ≥ 1). Faixas logarítmicas: 10×, como no plano. | O texto do plano não definia o passo para parâmetros sem limite natural (ex.: número de estágios). Esclarecimento, antes de qualquer resultado. |
| 2026-10-29 | **Modelo principal (seção 8):** "maior MCC médio na validação do braço" = o MCC médio das 50 avaliações com limiar 0,5, que é a própria função objetivo da busca; desempate pelo ROC-AUC médio. | Esclarecimento, antes de qualquer resultado. |
| 2026-10-29 | **Rodada-piloto em diretório separado** (`otimizacao_piloto/`), sem regra de borda e sem treino final; o piloto não entra no histórico da busca da rodada completa. Implementação: `src/09_optimize_hyperparams.py` (Optuna 5.0.0); teste cego dos braços A e B pelo `08_blind_test.py`, com o limiar de cada modelo lido do `_meta.json`. | Separar medição de custo da busca de verdade. Esclarecimento, antes de qualquer resultado. |
