# Manipulação de Dados
import pandas as pd
import numpy as np

# Pacotes Gráficos
import seaborn as sns
import matplotlib.pyplot as plt

# Pacotes Matemáticos
from scipy import stats
from scipy.stats import shapiro
import math

# Pacotes de Modelagem
import statsmodels as sms
import statsmodels.api as sm

from IPython.display import display
from sklearn.feature_selection import f_classif

# Métricas de Desempenho
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from scipy.stats import ks_2samp





# Classificar variaveis numéricas
def classificar_numericas(df, num_vars, limite_discreta=15, proporcao_max=0.05):
    """
    Classifica variáveis numéricas em discretas e contínuas de forma automatizada.
    
    Parâmetros
    ----------
    df : pandas.DataFrame
        Base de dados.
    num_vars : list
        Lista de variáveis numéricas a serem avaliadas.
    limite_discreta : int, default=15
        Número máximo de valores únicos para considerar uma variável como discreta.
    proporcao_max : float, default=0.05
        Proporção máxima entre valores únicos e tamanho do dataset para ser discreta.

    Retorno
    -------
    disc_vars : list
        Lista de variáveis discretas.
    cont_vars : list
        Lista de variáveis contínuas.
    """

    disc_vars = []
    cont_vars = []

    n = len(df)

    for col in num_vars:
        # Remove NaN temporariamente para avaliar valores únicos
        n_unique = df[col].dropna().nunique()
        proporcao_unicos = n_unique / n

        if (n_unique <= limite_discreta) or (proporcao_unicos <= proporcao_max):
            disc_vars.append(col)
        else:
            cont_vars.append(col)

    print("📊 Classificação das variáveis numéricas:")
    print(f"→ Discretas ({len(disc_vars)}): {disc_vars}")
    print(f"→ Contínuas ({len(cont_vars)}): {cont_vars}")

    return disc_vars, cont_vars
    

## SANITY CHECK FUNCTIONS
# Percentual de Outliers
def perc_outliers(df, num_vars):
    """
    Calcula o percentual de outliers para cada variável numérica com base no método do IQR.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados.
    num_vars : list
        Lista com os nomes das variáveis numéricas.

    Retorna:
    --------
    pandas.DataFrame
        DataFrame com duas colunas:
        - 'Variável': nome da variável numérica
        - '%_Outliers': percentual de outliers na variável
    """
    resultados = []

    for col in num_vars:
        # Quartis
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Limites inferior e superior
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Contagem de outliers
        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        perc = 100 * n_outliers / df[col].count()
        
        resultados.append({'Variável': col, '%_Outliers': round(perc, 2)})

    return pd.DataFrame(resultados).sort_values(by='%_Outliers', ascending=False).reset_index(drop=True)


## UNIVARIATE FUNCTIONS
# Histograma + QQPlots
def plot_univariate(df, col):
    """
    Gera um subplot com histograma + KDE e QQplot lado a lado para uma variável numérica.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados.
    col : str
        Nome da variável numérica a ser analisada.
    """


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Histograma com KDE ---
    sns.histplot(df[col], kde=True, color='skyblue', ax=axes[0])
    axes[0].set_title(f'Histograma e KDE - {col}', fontsize=12)
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequência')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- QQ Plot ---
    stats.probplot(df[col].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title(f'QQ Plot - {col}', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# Teste normalidade Shapiro
def test_normality_shapiro(df, num_vars, alpha=0.05):
    """
    Aplica o teste de Shapiro-Wilk para verificar normalidade em variáveis numéricas.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com as variáveis.
    num_vars : list
        Lista com os nomes das variáveis numéricas a serem testadas.
    alpha : float, opcional
        Nível de significância para interpretação do p-valor (default=0.05).
    
    Retorna
    -------
    pd.DataFrame com:
        - Variável
        - Estatística W
        - p-valor
        - Interpretação ('Segue distribuição normal' ou 'Não segue distribuição normal')
    """
    resultados = []

    for var in num_vars:
        if var not in df.columns:
            print(f"Atenção: '{var}' não encontrada no DataFrame. Ignorando.")
            continue
        
        # Remove valores ausentes antes do teste
        data = df[var].dropna()
        
        # Verifica se há amostras suficientes (Shapiro exige 3 ≤ n ≤ 5000)
        if len(data) < 3:
            resultados.append({
                'Variável': var,
                'W': None,
                'p-valor': None,
                'Interpretação': 'Amostra insuficiente'
            })
            continue
        elif len(data) > 5000:
            data = data.sample(5000, random_state=42)
        
        # Teste de Shapiro-Wilk
        stat, p_value = shapiro(data)
        
        interpretacao = (
            'Segue distribuição normal'
            if p_value > alpha else
            'Não segue distribuição normal'
        )
        
        resultados.append({
            'Variável': var,
            'W': stat,
            'p-valor': p_value,
            'Interpretação': interpretacao
        })
    
    return pd.DataFrame(resultados)




# Tabelas de Frequência
def freq_table(df, var):
    """
    Retorna uma tabela com frequência absoluta, relativa (%) e acumulada (%) 
    para uma variável categórica.
    """

    freq_abs = df[var].value_counts(dropna=False)
    freq_rel = df[var].value_counts(normalize=True, dropna=False) * 100
    freq_cum = freq_rel.cumsum()

    table = pd.DataFrame({
        "Frequência Absoluta": freq_abs,
        "Frequência Relativa (%)": freq_rel.round(2),
        "Frequência Acumulada (%)": freq_cum.round(2)
    }).reset_index().rename(columns={"index": var})

    return table

# grafico de barras freq abs e freq relativa
def grid_freq_bars(df, cat_vars, n_cols=2):
    """
    Gera um grid de gráficos de barras com frequências absolutas (barras)
    e relativas (linha com rótulos) para uma lista de variáveis categóricas.

    Parâmetros
    ----------
    df : pandas.DataFrame
        Base de dados contendo as variáveis.
    cat_vars : list
        Lista de nomes das variáveis categóricas a serem analisadas.
    n_cols : int, opcional
        Número de colunas no grid de subplots (default = 2).
    """
    n_vars = len(cat_vars)
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axes = axes.flatten() if n_vars > 1 else [axes]

    sns.set_theme(style="whitegrid")

    for i, var in enumerate(cat_vars):
        ax1 = axes[i]

        # --- Cálculo das frequências ---
        freq_abs = df[var].value_counts(dropna=False)
        freq_rel = (freq_abs / freq_abs.sum()) * 100

        # --- DataFrame auxiliar ---
        data_plot = pd.DataFrame({
            var: freq_abs.index.astype(str),
            "Frequência Absoluta": freq_abs.values,
            "Frequência Relativa (%)": freq_rel.values
        })

        # --- Barras (frequência absoluta) ---
        sns.barplot(
            x=var, 
            y="Frequência Absoluta", 
            data=data_plot, 
            color="skyblue", 
            ax=ax1
        )

        # --- Linha (frequência relativa) ---
        ax2 = ax1.twinx()
        sns.lineplot(
            x=var, 
            y="Frequência Relativa (%)", 
            data=data_plot, 
            color="darkorange", 
            marker="o", 
            linewidth=2, 
            ax=ax2, 
            label="Frequência Relativa (%)"
        )

        # --- Rótulos nos pontos da linha ---
        for j, row in data_plot.iterrows():
            ax2.text(
                x=j,
                y=row["Frequência Relativa (%)"] + 5,
                s=f"{row['Frequência Relativa (%)']:.1f}%", 
                color="darkorange", 
                fontsize=12, 
                ha="center"
            )

        # --- Estética ---
        ax1.set_title(f"Distribuição de {var}", fontsize=11, pad=10)
        #ax1.set_xlabel(var, fontsize=10)
        ax1.set_ylabel("Frequência Absoluta", fontsize=9)
        ax2.set_ylabel("Frequência Relativa (%)", fontsize=9)

        ax1.grid(False)
        ax2.grid(False)
        ax2.set_ylim(0, 100)
        ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

        # --- Legenda única (somente linha) ---
        lines, labels = ax2.get_legend_handles_labels()
        ax1.legend(lines, labels, loc="upper right", frameon=False)

    # Remover subplots vazios (caso sobrem espaços)
    for k in range(i + 1, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    plt.show()




## BIVARIATE FUNCTIONS

# analise IV (cat_vars x Binária Target)
class analise_iv:
        
    # função private
    def __get_tab_bivariada(self, var_escolhida):
     
        # Cria a contagem de Target_1 e Target_0
        df_aux = self.df.copy() 
        df_aux['target2'] = self.df[self.target]
        df2 = df_aux.pivot_table(values='target2',
                                 index=var_escolhida,
                                 columns=self.target,
                                 aggfunc='count')
        
        df2 = df2.rename(columns={0:'#Target_0',
                                  1:'#Target_1'})
        df2.fillna(0, inplace=True)

        # Cria as demais colunas da tabela bivariada
        df2['Total'] = (df2['#Target_0'] + df2['#Target_1'])
        df2['%Freq'] = (df2['Total'] / (df2['Total'].sum()) * 100).round(decimals=2)
        df2['%Target_1'] = (df2['#Target_1'] / (df2['#Target_1'].sum()) * 100).round(decimals=2)
        df2['%Target_0'] = (df2['#Target_0'] / (df2['#Target_0'].sum()) * 100).round(decimals=2)
        df2['%Target_0'] = df2['%Target_0'].apply(lambda x: 0.01 if x == 0 else x) #corrige problema do log indeterminado
        df2['%Taxa_de_Target_1'] = (df2['#Target_1'] / df2['Total'] * 100).round(decimals=2)
        df2['Odds'] = (df2['%Target_1'] / df2['%Target_0']).round(decimals=2)
        df2['Odds'] = df2.Odds.apply(lambda x: 0.01 if x == 0 else x) #corrige problema do log indeterminado
        df2['LN(Odds)'] = np.log(df2['Odds']).round(decimals=2)
        df2['IV'] = (((df2['%Target_1'] / 100 - df2['%Target_0'] / 100) * df2['LN(Odds)'])).round(decimals=2)
        df2['IV'] = np.where(df2['Odds'] == 0.01, 0 , df2['IV']) 

        df2 = df2.reset_index()
        df2['Variavel'] = var_escolhida
        df2 = df2.rename(columns={var_escolhida: 'Var_Range'})
        df2 = df2[['Variavel','Var_Range', '#Target_1','#Target_0', 'Total', '%Freq', '%Target_1', '%Target_0',
       '%Taxa_de_Target_1', 'Odds', 'LN(Odds)', 'IV']]
        
        # Guarda uma cópia da tabela no histórico
        self.df_tabs_iv = pd.concat([self.df_tabs_iv, df2], axis = 0)
        
        return df2
        
    def get_bivariada(self, var_escolhida='all_vars'):
        
        if var_escolhida == 'all_vars':
                       
            #vars = self.df.drop(self.target,axis = 1).columns
            vars = self.get_lista_iv().index
            for var in vars:
                tabela = self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var]
                print('==> "{}" tem IV de {}'.format(var,self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var]['IV'].sum().round(decimals=2)))
                # printa a tabela no Jupyter
                display(tabela)
            
            return
        
        else:
            print('==> "{}" tem IV de {}'.format(var_escolhida,self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var_escolhida]['IV'].sum().round(decimals=2)))
            return self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var_escolhida]
                   
            
    def get_lista_iv(self):
        
    
        # agrupa a lista de IV's em ordem descrescente
        lista = (self.df_tabs_iv.groupby('Variavel').agg({'IV':'sum'})).sort_values(by=['IV'],ascending=False)
            
        return lista
    
    

    def __init__(self, df, target, nbins=10):

        self.df = df.copy()
        self.target = target

        #lista de variaveis numericas
        df_num = self.df.loc[:,((self.df.dtypes == 'int32') | 
                                (self.df.dtypes == 'int64') | 
                                (self.df.dtypes == 'float64')
                               )
                            ]

        vars = df_num.drop(target,axis = 1).columns

        for var in vars:
            nome_var = 'fx_' + var 
            df_num[nome_var] = pd.qcut(df_num[var], 
                                       q=nbins, 
                                       precision=2,
                                       duplicates='drop')
            df_num = df_num.drop(var, axis = 1)
            df_num = df_num.rename(columns={nome_var: var})

        #lista de variaveis qualitativas
        df_str = self.df.loc[:,((self.df.dtypes == 'object') | 
                                (self.df.dtypes == 'category') |
                                (self.df.dtypes == 'bool'))]


        self.df = pd.concat([df_num,df_str],axis = 1)


         # inicializa tab historica
        self.df_tabs_iv = pd.DataFrame()

        vars = self.df.drop(self.target,axis = 1).columns
        for var in vars:
            self.__get_tab_bivariada(var);

        # remove tabs de iv duplicadas
        self.df_tabs_iv = self.df_tabs_iv.drop_duplicates(subset=['Variavel','Var_Range'], keep='last')


# TESTE F - ANOVA (num_vars x Binária Target)
def calcular_f_anova(df, num_vars, target):
    """
    Calcula o teste F (ANOVA) para variáveis numéricas em relação à variável alvo.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame com os dados.
    num_vars : list
        Lista com os nomes das variáveis numéricas a testar.
    target : str
        Nome da variável alvo (coluna categórica binária ou multicategórica).

    Retorna
    -------
    resultados_df : pandas.DataFrame
        DataFrame com colunas ['feature', 'F_statistic', 'p_value'],
        ordenado pelo p_value (crescente).
    """

    # Define X e y
    X = df[num_vars]
    y = df[target]

    # Calcula ANOVA F-test
    f_values, p_values = f_classif(X, y)

    # Monta DataFrame com os resultados
    resultados_df = pd.DataFrame({
        'feature': num_vars,
        'F_statistic': f_values,
        'p_value': p_values
    })

    # Ordena pelo p-value (do menor para o maior)
    resultados_df = resultados_df.sort_values(by='p_value', ascending=True).reset_index(drop=True)

    return resultados_df



# Grid com Graficos com variaveis discretas
def grid_disc_binario(df, target, disc_vars):
    """
    Cria um grid de gráficos de linha mostrando a proporção da classe positiva
    para variáveis discretas ou ordinais em um problema de classificação binária.
    
    Parâmetros
    ----------
    df : pandas.DataFrame
        Base de dados.
    target : str
        Nome da variável alvo binária (0/1 ou duas classes).
    disc_vars : list
        Lista de variáveis discretas/ordinais.
    """

    if not disc_vars:
        print("⚠️ Nenhuma variável discreta/ordinal informada.")
        return

    sns.set(style="whitegrid", palette="Set2", font_scale=1.0)

    n_disc = len(disc_vars)
    ncols = 3
    nrows = int(np.ceil(n_disc / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).reshape(-1)
    
    media_global = df[target].mean()

    for i, col in enumerate(disc_vars):
        prop = df.groupby(col, dropna=False)[target].mean().reset_index()

        sns.lineplot(data=prop, x=col, y=target, marker='o', ax=axes[i], color='tab:blue')
        axes[i].axhline(y=media_global, color='red', linestyle='--', label=f'Média global = {media_global:.2f}')
        axes[i].set_title(f"{col}")
        axes[i].set_ylabel("Proporção da classe positiva")
        axes[i].set_xlabel(col)
        axes[i].legend()

    # Remove eixos extras
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle("📊 Variáveis Discretas / Ordinais — Proporção da Classe Positiva", fontsize=14, y=1.02)
    plt.show()


# Grid com variáveis Continuas
def grid_cont_binario(df, target, cont_vars, bins=20):
    """
    Cria um grid de histogramas normalizados (densidades) para variáveis contínuas
    em um problema de classificação binária.
    
    Parâmetros
    ----------
    df : pandas.DataFrame
        Base de dados.
    target : str
        Nome da variável alvo binária (0/1 ou duas classes).
    cont_vars : list
        Lista de variáveis contínuas.
    bins : int, default=20
        Número de bins dos histogramas.
    """

    if not cont_vars:
        print("⚠️ Nenhuma variável contínua informada.")
        return

    sns.set(style="whitegrid", palette="Set2", font_scale=1.0)

    n_cont = len(cont_vars)
    ncols = 3
    nrows = int(np.ceil(n_cont / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).reshape(-1)

    pos_mask = df[target] == 1
    neg_mask = df[target] == 0

    for i, col in enumerate(cont_vars):
        x_pos = df.loc[pos_mask, col].dropna()
        x_neg = df.loc[neg_mask, col].dropna()

        # Normalização empírica: densidades que somam 1
        if len(x_pos) > 0 and len(x_neg) > 0:
            width_pos = (x_pos.max() - x_pos.min()) / bins
            width_neg = (x_neg.max() - x_neg.min()) / bins

            weights_pos = np.ones_like(x_pos) / (len(x_pos) * width_pos)
            weights_neg = np.ones_like(x_neg) / (len(x_neg) * width_neg)

            sns.histplot(x=x_pos, bins=bins, color='tab:blue', alpha=0.6, ax=axes[i],
                         stat="density", weights=weights_pos, label="Classe Positiva")
            sns.histplot(x=x_neg, bins=bins, color='tab:orange', alpha=0.6, ax=axes[i],
                         stat="density", weights=weights_neg, label="Classe Negativa")

            axes[i].set_title(f"{col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Proporção / Densidade Normalizada")
            axes[i].legend()

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle("📈 Variáveis Contínuas — Distribuições Normalizadas", fontsize=14, y=1.02)
    plt.show()

## plota graficos de barras empilhadas
def grid_stacked_bar(df, cat_vars, target, n_cols=2):
    n_vars = len(cat_vars)
    n_rows = math.ceil(n_vars / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")

    custom_palette = [
        "#1E3A8A", "#3B82F6", "#60A5FA",
        "#A5B4FC", "#94A3B8", "#CBD5E1"
    ]

    for i, var in enumerate(cat_vars):
        ax = axes[i]
        crosstab = pd.crosstab(df[target], df[var], normalize='index') * 100
        unique_cats = crosstab.columns
        palette = custom_palette[:len(unique_cats)]

        crosstab.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            width=0.7,
            edgecolor='white',
            color=palette
        )

        ax.set_title(f"{var} por {target}", fontsize=12, pad=10, weight="bold")
        ax.set_xlabel(target, fontsize=10)
        ax.set_ylabel("Frequência Relativa (%)", fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        ax.tick_params(axis='x', rotation=0)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="center",
                         fontsize=8, color="white", weight="bold")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# boxplot target x binaria
def boxplots_target_binaria(df, target, num_vars, n_cols=3):
    """
    Gera boxplots de uma variável alvo binária para várias variáveis numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    target : str
        Nome da variável alvo binária (ex: 'Target', 'Y', etc.).
    num_vars : list
        Lista com os nomes das variáveis numéricas.
    n_cols : int, optional
        Número de colunas do grid de subplots. Default é 3.
    """
    
    n_vars = len(num_vars)
    n_rows = int(np.ceil(n_vars / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    sns.set_style("whitegrid")
    plt.suptitle("Boxplots por variável numérica (Target binária)", fontsize=16, fontweight="bold")

    for i, var in enumerate(num_vars):
        sns.boxplot(
            x=target,
            y=var,
            data=df,
            hue=target,          # ✅ necessário para evitar o warning
            palette="Set2",
            legend=False,        # ✅ não queremos legendas repetidas
            showfliers=True,
            boxprops=dict(alpha=0.7),
            ax=axes[i]
        )
        axes[i].set_title(f"{var} x {target}", fontsize=12)
        axes[i].set_xlabel("")
        axes[i].set_ylabel(var)
    
    # Remover eixos vazios
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



################### Modelagem ###########################




##### INFERENCIA #######
##### STATSMODEL #######
##### REGRESSAO LINEAR MULTIPLA ######

# Residual plots (grid2x2)
def diagnostico_residuos(result):
    """
    Gera um painel 2x2 de diagnóstico dos resíduos de um modelo statsmodels:
    1. Resíduos padronizados vs número de observações
    2. Resíduos padronizados vs valores preditos (com LOESS)
    3. Histograma + KDE dos resíduos padronizados
    4. QQPlot dos resíduos padronizados
    """

    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    # Extrair resíduos padronizados e valores preditos
    resid = result.resid_pearson
    fitted = result.fittedvalues
    n = len(resid)

    # Criar figura e eixos 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Diagnóstico de Resíduos", fontsize=16, fontweight='bold')

    # -------------------------------
    # 1️⃣ Resíduos padronizados vs número de observações
    sns.scatterplot(x=range(n), y=resid, ax=axes[0, 0], color='steelblue', alpha=0.7)
    axes[0, 0].axhline(0, color='black', linestyle='--')
    axes[0, 0].axhline(2, color='red', linestyle='--')
    axes[0, 0].axhline(-2, color='red', linestyle='--')
    axes[0, 0].set_title("Resíduos Padronizados vs Observações", fontsize=12)
    axes[0, 0].set_xlabel("Índice da Observação")
    axes[0, 0].set_ylabel("Resíduos Padronizados")
    axes[0, 0].grid(alpha=0.3)

    # -------------------------------
    # 2️⃣ Resíduos padronizados vs valores preditos com LOESS
    sns.scatterplot(x=fitted, y=resid, ax=axes[0, 1], color='teal', alpha=0.6)
    loess_fit = lowess(resid, fitted, frac=0.3)
    axes[0, 1].plot(loess_fit[:, 0], loess_fit[:, 1], color='red', linewidth=2, label='LOESS')
    axes[0, 1].axhline(0, color='black', linestyle='--')
    axes[0, 1].axhline(2, color='red', linestyle='--')
    axes[0, 1].axhline(-2, color='red', linestyle='--')
    axes[0, 1].set_title("Resíduos Padronizados vs Valores Preditos", fontsize=12)
    axes[0, 1].set_xlabel("Valores Preditos")
    axes[0, 1].set_ylabel("Resíduos Padronizados")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # -------------------------------
    # 3️⃣ Histograma + KDE dos resíduos padronizados
    sns.histplot(resid, kde=True, ax=axes[1, 0], color='royalblue', bins=25)
    axes[1, 0].set_title("Distribuição dos Resíduos Padronizados", fontsize=12)
    axes[1, 0].set_xlabel("Resíduos Padronizados")
    axes[1, 0].set_ylabel("Frequência")
    axes[1, 0].grid(alpha=0.3)

    # -------------------------------
    # 4️⃣ QQPlot dos resíduos padronizados
    sm.qqplot(resid, line='45', fit=True, ax=axes[1, 1], alpha=0.7)
    axes[1, 1].set_title("QQPlot dos Resíduos Padronizados", fontsize=12)
    axes[1, 1].set_xlabel("Quantis Teóricos")
    axes[1, 1].set_ylabel("Quantis dos Resíduos")

    # -------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()    


# residuos parciais via statsmodels
def partial_res(result, data, target_name):
    """
    Gera partial regression plots para cada preditor do modelo,
    mostrando a reta da modelagem e a suavização LOWESS pontilhada vermelha.
    
    Parameters
    ----------
    result : statsmodels RegressionResultsWrapper
        Modelo ajustado
    data : pd.DataFrame
        DataFrame contendo todas as variáveis
    target_name : str
        Nome da variável dependente no DataFrame
    """

    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    exog_names = [v for v in result.model.exog_names if v != 'const']
    n_vars = len(exog_names)
    
    n_cols = 2
    n_rows = int(np.ceil(n_vars / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
    axes = axes.flatten()
    
    # Título geral do grid
    fig.suptitle("Resíduos Parciais", fontsize=16, fontweight='bold', y=1.02)
    
    for i, var in enumerate(exog_names):
        # Plot parcial padrão
        sm.graphics.plot_partregress(endog=target_name,
                                     exog_i=var,
                                     exog_others=[v for v in exog_names if v != var],
                                     data=data,
                                     obs_labels=False,
                                     ax=axes[i])
        
        # Captura pontos do scatter (resíduos parciais)
        ax = axes[i]
        x = ax.lines[0].get_xdata()
        y = ax.lines[0].get_ydata()
        
        # LOWESS
        lowess_fit = lowess(y, x, frac=0.3)
        ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], 'r--', linewidth=2, label='LOWESS')
        
        # Reta do modelo (linear)
        coef = np.polyfit(x, y, 1)
        x_vals = np.array([x.min(), x.max()])
        y_vals = coef[0]*x_vals + coef[1]
        ax.plot(x_vals, y_vals, 'k-', linewidth=2, label='Reta do modelo')
        
        # Título do subplot e legendas
        ax.set_title(f'{var}', fontsize=12)
        ax.legend()

        # Títulos individuais opcionais
        axes[i].set_title(f'{var}', fontsize=12)
        axes[i].set_xlabel(f'Resíduos parciais de {var}')
        axes[i].set_ylabel(f'Resíduos parciais de {target_name}')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    # Remove eixos extras caso existam
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

######################################################
######################################################

# Função para cálculo do KS
def ks_stat(y, y_pred):
    return ks_2samp(y_pred[y==1], y_pred[y!=1]).statistic

# Função para cálculo do desempenho de modelos
def calcula_desempenho(modelo, x_train, y_train, x_test, y_test, thr=0.5):
    # Probabilidades preditas
    ypred_proba_train = modelo.predict_proba(x_train)[:, 1]
    ypred_proba_test  = modelo.predict_proba(x_test)[:, 1]

    # Converte probabilidades em classes usando o threshold informado
    ypred_train = (ypred_proba_train >= thr).astype(int)
    ypred_test  = (ypred_proba_test  >= thr).astype(int)

    # Métricas de Desempenho
    acc_train = accuracy_score(y_train, ypred_train)
    acc_test = accuracy_score(y_test, ypred_test)
    
    roc_train = roc_auc_score(y_train, ypred_proba_train)
    roc_test  = roc_auc_score(y_test, ypred_proba_test)
    
    ks_train = ks_stat(y_train, ypred_proba_train)
    ks_test  = ks_stat(y_test, ypred_proba_test)
    
    prec_train = precision_score(y_train, ypred_train, zero_division=0)
    prec_test  = precision_score(y_test, ypred_test, zero_division=0)
    
    recl_train = recall_score(y_train, ypred_train)
    recl_test  = recall_score(y_test, ypred_test)
    
    f1_train = f1_score(y_train, ypred_train)
    f1_test  = f1_score(y_test, ypred_test)

    df_desemp = pd.DataFrame({
        'Treino': [acc_train, roc_train, ks_train, prec_train, recl_train, f1_train],
        'Teste':  [acc_test, roc_test, ks_test, prec_test, recl_test, f1_test]
    }, index=['Acurácia','AUCROC','KS','Precision','Recall','F1'])
    
    df_desemp['Variação'] = round(df_desemp['Teste'] / df_desemp['Treino'] - 1, 2)
    
    return df_desemp
