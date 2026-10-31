# Diretrizes do Agente: Análise Exploratória de Risco (Behavior Score)

## 1. Missão do Agente
Atuar como um **Cientista de Dados Sênior Especialista em Risco de Crédito Quantitativo**. O objetivo é executar uma Análise Exploratória de Dados (EDA) orientada a comportamento financeiro (*Behavior Score*), discretização de variáveis, mensuração univariada de risco e geração de diagnósticos estratégicos para a mesa de crédito e políticas de risco.

---

## 2. Padrão Estético e Visual
- **Tema Visual:** Clean corporativo / Executivo.
- **Paleta de Cores:** Tonalidades estritas de verde (*Forest, Emerald, Jade, Mint e Sage*), garantindo legibilidade e contraste profissional.
- **Gráficos esperados:**
  1. Estabilidade Temporal de Safras (Bad Rate vs. Volume).
  2. Screening de Poder Discriminativo por Information Value (IV).
  3. Curvas de Monotonicidade e WoE por Faixa de Risco.
  4. Matriz de Correlação Monotônica de Spearman.

---

## 3. Código Python do Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# ==============================================================================
# CONFIGURAÇÃO DE PALETA CORPORATIVA: TONALIDADES DE VERDE
# ==============================================================================
GREEN_DARK   = "#1B4D3E"  # Verde Floresta Profundo (Ênfase/Linhas críticas)
GREEN_MID    = "#2E8B57"  # SeaGreen / Esmeralda (Barras principais)
GREEN_LIGHT  = "#66BB6A"  # Verde Claro Médio
GREEN_PALE   = "#A3E4D7"  # Menta Pálido (Fundos/Volumes secundários)
GREEN_ACCENT = "#0E6251"  # Verde Petróleo (Destaques executivos)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 11,
    'axes.edgecolor': '#D5D8DC',
    'axes.linewidth': 0.8
})

# ==============================================================================
# 1. FUNÇÃO CORE: CÁLCULO DE WEIGHT OF EVIDENCE (WOE) E INFORMATION VALUE (IV)
# ==============================================================================
def calcular_woe_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10, is_continuous: bool = True):
    """
    Discretiza atributos contínuos em quantis, preserva valores nulos em categoria dedicada
    e calcula a tabela de contingência com WoE e IV.
    """
    tmp = df[[feature, target]].copy()
    tmp['bin'] = tmp[feature].fillna("MISSING")
    
    if is_continuous:
        mask_valid = tmp[feature].notnull()
        try:
            tmp.loc[mask_valid, 'bin'] = pd.qcut(
                tmp.loc[mask_valid, feature], 
                q=bins, 
                duplicates='drop'
            ).astype(str)
        except Exception:
            tmp.loc[mask_valid, 'bin'] = pd.cut(
                tmp.loc[mask_valid, feature], 
                bins=bins, 
                duplicates='drop'
            ).astype(str)

    total_goods = (tmp[target] == 0).sum()
    total_bads = (tmp[target] == 1).sum()
    
    agg = tmp.groupby('bin', observed=False).agg(
        total=(target, 'count'),
        bads=(target, 'sum'),
        goods=(target, lambda x: (x == 0).sum()),
        bad_rate=(target, 'mean')
    ).reset_index()

    # Laplace smoothing contra log(0) e divisões indefinidas
    agg['dist_good'] = (agg['goods'] + 0.5) / (total_goods + 1.0)
    agg['dist_bad'] = (agg['bads'] + 0.5) / (total_bads + 1.0)

    # Convenção clássica de crédito: ln(Good% / Bad%)
    agg['woe'] = np.log(agg['dist_good'] / agg['dist_bad'])
    agg['iv_bin'] = (agg['dist_good'] - agg['dist_bad']) * agg['woe']
    
    total_iv = agg['iv_bin'].sum()
    return agg, total_iv

# ==============================================================================
# 2. ESTABILIDADE DE SAFRA (VOLUME ATIVO VS. BAD RATE TEMPORAL)
# ==============================================================================
def plot_estabilidade_safra(df: pd.DataFrame, col_safra: str, target: str):
    """
    Avalia se variações na taxa de inadimplência decorrem de choques macroeconômicos
    ou relaxamento na política de concessão/expansão de carteira.
    """
    safra_summary = df.groupby(col_safra).agg(
        volume=(target, 'count'),
        bad_rate=(target, 'mean')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    # Volume (Barras em tom verde pálido)
    ax1.bar(
        safra_summary[col_safra], 
        safra_summary['volume'], 
        color=GREEN_PALE, 
        edgecolor=GREEN_MID,
        alpha=0.7, 
        label='Volume Ativo'
    )
    
    # Bad Rate (Linha em verde floresta escuro)
    ax2.plot(
        safra_summary[col_safra], 
        safra_summary['bad_rate'] * 100, 
        color=GREEN_DARK, 
        marker='o', 
        linewidth=2.5, 
        label='Bad Rate (%)'
    )

    ax1.set_xlabel('Safra de Referência', fontweight='bold', color=GREEN_DARK)
    ax1.set_ylabel('Volume de Clientes Ativos', color=GREEN_MID, fontweight='bold')
    ax2.set_ylabel('Taxa de Default (%)', color=GREEN_DARK, fontweight='bold')
    ax1.grid(False)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.title('Estabilidade de Risco no Tempo: Volume vs. Taxa de Default', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3. SCREENING DE PODER DISCRIMINATIVO (RANKING DE IV EM VERDE MONOCROMÁTICO)
# ==============================================================================
def plot_ranking_iv(df: pd.DataFrame, features: list, target: str):
    """
    Ordena variáveis pelo Information Value em gradientes de verde institucional.
    """
    iv_dict = {col: calcular_woe_iv(df, col, target)[1] for col in features}
    iv_df = pd.DataFrame(list(iv_dict.items()), columns=['Feature', 'IV']).sort_values('IV', ascending=True)

    # Gradiente em tons de verde de acordo com a força preditiva
    cores = []
    for val in iv_df['IV']:
        if val < 0.02:
            cores.append('#D4EFDF')  # Inexpressivo (<0.02)
        elif val < 0.10:
            cores.append('#A9DFBF')  # Fraco (0.02 - 0.10)
        elif val < 0.30:
            cores.append('#52BE80')  # Médio (0.10 - 0.30)
        elif val <= 0.50:
            cores.append('#1E8449')  # Forte Ideal (0.30 - 0.50)
        else:
            cores.append('#114B27')  # Suspeito/Alerta de Leakage (>0.50)

    fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.45)))
    ax.barh(iv_df['Feature'], iv_df['IV'], color=cores, edgecolor=GREEN_DARK, linewidth=0.6)

    # Linhas de corte canônicas
    ax.axvline(0.02, color=GREEN_LIGHT, linestyle=':', label='Mínimo Relevante (0.02)')
    ax.axvline(0.10, color=GREEN_MID, linestyle='--', label='Preditores Médios (0.10)')
    ax.axvline(0.30, color=GREEN_DARK, linestyle='--', label='Preditores Fortes (0.30)')
    ax.axvline(0.50, color=GREEN_ACCENT, linestyle='-', label='Alerta de Vazamento (>0.50)')

    ax.set_xlabel('Information Value (IV)', fontweight='bold', color=GREEN_DARK)
    ax.set_title('Screening Univariado: Information Value (IV)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.show()
    return iv_df.sort_values('IV', ascending=False)

# ==============================================================================
# 4. CURVA DE MONOTONICIDADE E ESTRATIFICAÇÃO POR FAIXA (WOE & BAD RATE)
# ==============================================================================
def plot_curva_monotonicidade(tabela_woe: pd.DataFrame, feature_name: str, total_iv: float):
    """
    Exibe a concordância monotônica entre faixas de comportamento, inadimplência e WoE.
    """
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(tabela_woe))

    # Barras de Bad Rate (Verde Médio)
    ax1.bar(
        x, 
        tabela_woe['bad_rate'] * 100, 
        color=GREEN_MID, 
        alpha=0.65, 
        width=0.45, 
        label='Bad Rate (%)'
    )
    
    # Curva WoE (Verde Floresta Escuro)
    ax2.plot(
        x, 
        tabela_woe['woe'], 
        color=GREEN_DARK, 
        marker='s', 
        markersize=6,
        linewidth=2.5, 
        label='WoE [ln(Good%/Bad%)]'
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(tabela_woe['bin'], rotation=35, ha='right')
    ax1.set_ylabel('Taxa de Default (% Bad)', color=GREEN_MID, fontweight='bold')
    ax2.set_ylabel('Weight of Evidence (WoE)', color=GREEN_DARK, fontweight='bold')
    ax1.set_title(f'Monotonicidade e Risco: {feature_name} (IV: {total_iv:.3f})', fontsize=12, fontweight='bold', pad=15)
    
    ax1.grid(False)
    ax2.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 5. MATRIZ DE CORRELAÇÃO DE SPEARMAN (COMBATE À MULTICOLINEARIDADE)
# ==============================================================================
def plot_matriz_spearman(df: pd.DataFrame, top_features: list):
    """
    Matriz de correlação não-linear para evitar atributos comportamentais redundantes.
    """
    corr, _ = spearmanr(df[top_features].dropna())
    corr_df = pd.DataFrame(corr, index=top_features, columns=top_features)

    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    
    # Paleta customizada de verdes contínuos
    cmap_verdes = sns.light_palette(GREEN_DARK, as_cmap=True)

    sns.heatmap(
        corr_df, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap_verdes, 
        vmin=0, 
        vmax=1, 
        linewidths=0.6,
        cbar_kws={"shrink": .8}
    )
    plt.title('Matriz de Correlação de Spearman (Verificação de Redundância)', fontsize=12, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
```

---

## 4. Estrutura Esperada do Notebook de EDA

O notebook deve ser organizado **por variável analisada**, priorizando legibilidade e leitura executiva em vez de automação por loops.

### 4.1 Regras de Estrutura

* Antes de cada análise, incluir uma célula **markdown** com o nome da coluna e seu significado de negócio.
* Chamar as funções **explicitamente para cada coluna**, sem `for` para a análise individual.
* Após o gráfico e a tabela de WoE/IV, incluir uma célula **markdown** com a descrição detalhada da variável.
* Loops são permitidos apenas dentro de funções utilitárias (ex.: cálculo de IV no screening), nunca para substituir a análise individual das colunas.
* Ao final das análises individuais, consolidar uma síntese executiva com as recomendações de modelagem.

### 4.2 Template de Análise por Variável

```python
# ---------------------------------------------------------------------------
# PAY_0 — Status de pagamento mais recente
# ---------------------------------------------------------------------------
tabela_pay_0, iv_pay_0 = calcular_woe_iv(df, "PAY_0", TARGET, is_continuous=False)
plot_curva_monotonicidade(tabela_pay_0, "PAY_0", iv_pay_0)
display(tabela_pay_0[["bin", "total", "bads", "goods", "bad_rate", "woe", "iv_bin"]].round(4))
```

### 4.3 Conteúdo Obrigatório da Descrição da Variável

Cada descrição deve cobrir, de forma objetiva:

1. **O que a variável mede** e como suas faixas devem ser interpretadas.
2. **Evidência estatística:** IV, bad rate por faixa, monotonicidade e WoE.
3. **Correlação** com outras variáveis do bloco (Spearman), quando aplicável.
4. **Recomendação para modelagem:**
   * Manter como preditor central.
   * Transformar/agrupar categorias (ex.: `0 / 1 / 2 / 3+`).
   * Descartar por redundância ou baixo valor incremental.
   * Verificar possível vazamento (*leakage*) quando o IV for superior a `0.50`.
5. **Cuidados operacionais:** categorias raras, nulos com significado de negócio e estabilidade temporal.

### 4.4 Variáveis Ordinais e Categóricas

Para variáveis de comportamento já discretizadas (`PAY_0` a `PAY_6`, `FREQ_ATRASO`, `MAX_ATRASO`, `SEX`, `EDUCATION`, `MARRIAGE`), usar `is_continuous=False`. Reservar discretização por quantis (`is_continuous=True`) para variáveis monetárias e contínuas (`LIMIT_BAL`, `BILL_AMT*`, `PAY_AMT*`, `AGE`, `UTILIZACAO`, etc.).

### 4.5 Recomendação Final de Variáveis para Modelagem

O notebook deve terminar com uma seção explícita de recomendação, contendo:

1. **Tabela de variáveis recomendadas para o modelo**, com colunas de variável, papel no modelo, IV aproximado e justificativa.
2. **Tabela de diagnóstico estatístico das variáveis recomendadas**, contendo: IV, indicação de monotonicidade (com direção e/ou `ρ` de Spearman), **WoE mínimo, máximo e amplitude** e **VIF** calculado somente para o conjunto final, após a transformação WoE.
3. **Tabela de variáveis que não devem entrar**, listando cada variável ou grupo e o motivo da exclusão (redundância, baixo IV, restrição regulatória, identificador, target etc.).
4. **Transformações recomendadas** antes da modelagem (agrupamento de categorias, `log1p`, clip, bin de nulos).
5. **Ressalvas de validação**, incluindo VIF, stepwise, PSI por safra e validação out-of-time.
6. **Critério de decisão:** priorizar equilíbrio entre poder discriminativo (IV), interpretabilidade de negócio e baixa correlação (Spearman), evitando modelos inflados por variáveis redundantes.

**Regras para o diagnóstico estatístico:**

* O **WoE é calculado por faixa**; não apresentar um valor único como resumo. Usar mínimo, máximo e amplitude (ou o WoE das faixas extremas).
* Classificar a monotonicidade como `Sim` (crescente/decrescente), `Parcial` ou `Não`, com base na sequência de bad rate e no `ρ` de Spearman.
* Calcular o **VIF** apenas no conjunto final de variáveis, após transformação WoE, com dados centralizados. Referências: `< 5` aceitável, `5–10` atenção e `> 10` multicolinearidade severa.

---

## 5. Guia de Interpretação e Roteiro de Negócios

Ao consolidar a análise para a diretoria e comitê de crédito, o agente deve seguir este roteiro de diagnóstico:

### 1. Dinâmica da Safra (Vintages)
* **Padrão Encontrado:** Bad Rate estável com picos pontuais.
* **Diagnóstico de Negócio:** Se o aumento na taxa de default coincide com expansões de limite ou relaxamento no corte de concessão, trata-se de **seleção adversa**. Caso ocorra com volume estável, a causa é exógena (estresse macroeconômico regional ou setorial).

### 2. Saturação de Limites (Overextension)
* **Padrão Encontrado:** Bad Rate sobe de forma exponencial acima de $75\%$ de utilização.
* **Ação de Política:** 
  * Clientes com utilização $< 40\%$ são perfis para campanhas de *cross-sell* e ampliação de limite.
  * Clientes na faixa $> 80\%$ devem sofrer travas de aumento automático e receber ofertas estruturadas de consolidação de passivo antes de ingressarem em atrasos formais.

### 3. Frequência no Rotativo e Cheque Especial
* **Padrão Encontrado:** Alto poder preditivo ($IV > 0.35$).
* **Ação de Política:** O uso continuado por mais de 10 dias consecutivos é sintoma de descasamento severo de liquidez. O score deve penalizar essa permanência de forma desproporcional para mitigar perdas por rolagem contínua.

### 4. Tratamento Estratégico de Registros Ausentes (Missings)
* **Padrão Encontrado:** Nulos em variáveis de atraso histórico apresentam os menores índices de default.
* **Ação de Política:** Nulos em *Behavior Score* carregam significado operacional (ex: ausência de atrasos = cliente exemplar). Não imputar pela média ou mediana; manter em bin dedicado com seu respectivo valor de WoE.