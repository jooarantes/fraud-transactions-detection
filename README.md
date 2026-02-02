
# Projeto: Detecção de Anomalias em Transações de Cartão de Crédito

Este projeto tem como objetivo o desenvolvimento de um sistema de detecção de fraudes em transações financeiras, utilizando técnicas de Machine Learning supervisionado e não supervisionado, com foco em robustez estatística, interpretabilidade e impacto no negócio.

A solução foi construída de forma incremental, partindo de um modelo âncora interpretável, evoluindo para modelos baseados em árvores e ensembles, e explorando abordagens de detecção de anomalias. Todo o pipeline foi desenhado para refletir cenários reais de produção, incluindo desbalanceamento extremo, definição de limiar de decisão (threshold) orientado a custo e testes de estresse de métricas de negócio.

## 📸 Destaques Visuais

Os gráficos abaixo representam os principais achados do projeto:

### **Distribuição dos valores das Fraudes**
![fraude-amount-dist](https://github.com/jooarantes/fraud-transactions-detection/blob/main/reports/graphs/distplot-fraudlent-transactions.png)

### **Desempenho do Modelo em Dados de Teste (holdout)**
![performance-holdout](https://github.com/jooarantes/fraud-transactions-detection/blob/main/reports/graphs/test-logit-with-best-thr.png)


##  🎯 Objetivos do Projeto

- Desenvolver um modelo de classificação binária para atuar como sistema anti-fraude;
- Comparar diferentes famílias de modelos como modelos lineares, baseados em árvores e ensembles e, modelos não supervisionados como IsolationForest;


## 🧠 Principais Aprendizados

- Como lidar com desbalanceamento extremo da base e tomar decisões que vão além do modelo;
- Exploração da otimização do threshold com base em trade-offs entre falsos positivos e falsos negativos;
- Modelos mais complexos nem sempre dominam o baseline quando avaliados sob métricas de estabilidade e generalização;
- Utilização de metadados para evitar vazamento de dados e garantir avaliação justa entre os modelos;


## 📂 Conteúdo do Repositório

O repositório está organizado para facilitar a navegação entre análises, resultados e implementação, permitindo que diferentes perfis de leitores explorem o projeto conforme seu interesse.

### 📓 Notebooks Analíticos

Toda análise está concentrada em um único notebook:

**[01_logit_anchor_model.ipynb](https://github.com/jooarantes/fraud-transactions-detection/blob/main/notebooks/01_logit_anchor_model.ipynb)**  

**[02_based_tree_models.ipynb](https://github.com/jooarantes/fraud-transactions-detection/blob/main/notebooks/02_based_tree_models.ipynb)**  

**[03_isolation_forest.ipynb](https://github.com/jooarantes/fraud-transactions-detection/blob/main/notebooks/03_isolation_forest.ipynb)**  

---

### 📊 Reports e Resultados

A pasta `reports/` contém os principais artefatos gerados ao longo do projeto, permitindo acesso direto a resultados sem a necessidade de executar os notebooks:

- **[Gráficos](https://github.com/jooarantes/fraud-transactions-detection/tree/main/reports/graphs)** utilizados na análise final;
- **[Tabelas](https://github.com/jooarantes/fraud-transactions-detection/tree/main/reports/Tables)** resumo;
- **[Figuras](https://github.com/jooarantes/fraud-transactions-detection/tree/main/reports/Figures)** consolidadas para comunicação dos resultados.

---

### 🧠 Código Fonte (`src/`)

A pasta `src/` contém a implementação modular utilizada nos notebooks:

- **`evaluation/`**
  Métrica econômica usada na avaliação dos modelos.

- **`utils/`**  
  Funções auxiliares reutilizáveis ao longo do projeto (pré-processamento, visualizações e helpers).


## ▶️ Como Reproduzir as Análises

### 1. Clonar o repositório
Clone o repositório para sua máquina local:

```bash
git clone https://github.com/jooarantes/fraud-transactions-detection.git
cd fraud-transactions-detection
```
### 2. Criar e Ativar o ambiente virtual
```bash
conda env create -f environment.yml
conda activate fraud-transactions-detection
```
### 3. Baixar a Base de Dados
Acesse https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud para baixar o dataset

### 4. Criar a pasta Data para conter o Dataset
Crie uma pasta no diretório do repositório chamada Data e dentro dela crie uma pasta chamada Raw e coloque o arquivo csv lá.

Agora é só rodar o Notebook =)
## 📖 Contexto do Problema de Negócio

Instituições Financeiras enfrentam o desafio de identificar fraudes em tempo quase real, equilibrando dois riscos principais: fraudes não detectadas, que geram perdas financeiras diretas; E, transações legítimas bloqueadas que impactam a experiência do cliente e aumentam custos operacionais.

Este projeto busca simular esse contexto propondo uma abordagem orientada a decisão, onde o modelo é apenas uma parte do sistema, e não um fim em si mesmo.
## ⚙️ Metodologia

O projeto segue a lógica do CRISP-DM
- Entendimento do problema
- Entendimento dos dados
- Preparação dos dados
- Análise exploratória inicial (EDA)
- Modelagem Preditiva
- Validação
- Deploy
## 📐 Métricas de Avaliação

O projeto utiliza dois grupos de métricas:

**Métricas Estatísticas**

- ROC AUC
- Precision-Recall
- KS

Utilizadas principalmente para **diagnóstico e comparação técnica.**

**Métricas Econômicas**

- Métrica Personalizada de Ganhos

A decisão final **não é baseada exclusivamente em métricas estatísticas.**

## 📊 Principais Resultados

- O modelo âncora (regressão logística) mitigou cerca de 70% do impacto econômico gerado pelas fraudes, resultando em um ganho financeiro de, aproximadamente, 12.4%;
- O modelo de regressão logística ficou muito perto do desempenho dos modelos baseados em árvore de decisão (RandomForest e AdaBoost), tido como benchmark de performance;
- O aumento da complexidade do modelo oferecido pelos modelos tree-based não compensaram o aumento do tempo de execução do algoritmo;
- O IsolationForest teve desempenho intermediário não superando o modelo de regressão logística.

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas, Numpy
- Scikit-Learn
- RandomForest e AdaBoost
- Matplotlib e Seaborn
- Jupyter Notebook

  
## 👤 Autores

**João Arantes**

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/joao-arantes-ds/)

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherineoelsner.com/)
## 🔗 Conteúdos Relacionados

- Artigo no Medium: ***[Além do AUC - Construindo um Sistema de Detecção de Fraudes Orientado ao Negócio](https://medium.com/@jooaarantes/al%C3%A9m-do-auc-construindo-um-sistema-de-detec%C3%A7%C3%A3o-de-fraudes-orientado-ao-neg%C3%B3cio-9ed99b06208b)***


## Licença

[MIT](https://github.com/jooarantes/fraud-transactions-detection/blob/main/LICENSE)

