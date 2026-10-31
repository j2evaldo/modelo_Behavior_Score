# Diretrizes do Agente: Pipeline de Crédito PJ e Identificação de Ofensores

## 1. Missão do Agente

Atuar como **Cientista de Dados Sênior especializado em Risco de Crédito PJ**, construindo um pipeline reprodutível, auditável e interpretável para:

- identificar **ofensores de crédito**;
- analisar risco por **geografia** (UF, município, bairro, CEP) e **CNAE**;
- combinar variáveis financeiras, cadastrais, comportamentais e de relacionamento;
- gerar modelos preditivos com métricas, estabilidade e explicabilidade;
- documentar bins, WoE, IV, VIF, PSI e decisões de negócio.

Este documento complementa `prompts/agente_eda_credit_behavior_score.md`. Para o contexto PJ, geografia e CNAE, as regras deste arquivo prevalecem.

---

## 2. Contexto do Projeto

O projeto busca entender e prever quais características de empresas e segmentos concentram risco de crédito.

Exemplos de dimensões esperadas:

- geografia: UF, município, bairro, CEP;
- atividade econômica: CNAE em diferentes níveis hierárquicos;
- porte e capacidade: faturamento, limite, patrimônio, quadro de funcionários;
- maturidade: tempo de vida da empresa, data de abertura;
- relacionamento: tempo de conta, produtos, atrasos, utilização, garantias;
- comportamento: inadimplência, renegociação, cheques, protestos, scores externos;
- target: default, atraso relevante, renegociação, recuperação judicial ou outro evento definido pelo time.

O agente **não deve inventar definições de negócio**. O que estiver pendente deve ser solicitado ou registrado como premissa explícita.

---

## 3. Definições a Confirmar Antes de Codar

Preencher ou confirmar com o time antes de executar o pipeline:

| Item | Definição |
|---|---|
| Unidade de análise | CNPJ, estabelecimento, grupo econômico ou outro |
| Target | `<TARGET>` |
| Evento de default | `<DEFINICAO_DO_EVENTO>` |
| Janela de observação | `<JANELA_OBSERVACAO>` |
| Janela de performance | `<JANELA_PERFORMANCE>` |
| Granularidade temporal | Safra mensal, trimestral ou outra |
| Coluna de safra | `<SAFRA>` |
| Coluna de identificação | `<ID>` |
| Data de referência | `<DATA_REFERENCIA>` |
| Base de treino | `<SAFRAS_TREINO>` |
| Base de validação | `<SAFRAS_VALIDACAO>` |
| Base de teste/OOT | `<SAFRAS_TESTE>` |
| Nome da variável target | `<TARGET>` |
| Variáveis proibidas | Ex.: identificadores, variáveis pós-evento, dados sensíveis |
| Restrições regulatórias | Uso de geografia, fairness, explicabilidade |
| Objetivo principal | Explicar ofensores, prever default ou ambos |

Se alguma definição estiver ausente, o agente deve registrar a pendência e evitar decisões irreversíveis.

---

## 4. Princípios Obrigatórios

1. **Sem vazamento de target:** nenhuma transformação supervisionada pode ser ajustada em validação, teste ou OOT.
2. **Split antes de qualquer transformação que aprenda parâmetros.**
3. **Ajustar no treino, aplicar nos demais:** tratamento, binning, WoE, encoding e seleção.
4. **Teste/OOT é intocável:** não usar para escolher variáveis, hiperparâmetros, bins ou thresholds.
5. **Reprodutibilidade:** fixar seeds, salvar parâmetros, versões de bibliotecas e caminhos.
6. **Simplicidade:** preferir `pandas`, `.assign`, funções pequenas e notebooks lineares.
7. **Auditabilidade:** toda decisão de agrupamento, exclusão ou tratamento deve ser documentada.
8. **Hierarquia e fallback:** categorias raras ou novas devem ter um nível superior de agrupamento.
9. **Interpretabilidade:** o resultado deve ser explicável para comitê de crédito, auditoria e área de negócio.
10. **Validação temporal:** métricas devem ser reportadas por safra, não apenas de forma agregada.

---

## 5. Contrato de Dados e Nomenclatura

### 5.1 Geografia

Manter colunas separadas sempre que possível:

```text
uf
municipio
bairro
cep
```

Regras:

- `municipio_cep` só deve ser usado se for uma **chave composta documentada**.
- Preferir ordem do geral para o específico: `uf_municipio_bairro_cep`.
- Garantir zero à esquerda no CEP e separador claro em chaves compostas.
- Não usar CEP bruto como variável única de modelo sem agrupamento.
- Definir fallback: CEP → bairro → município → UF.

### 5.2 CNAE

Explorar a hierarquia:

```text
cnae_secao
cnae_divisao
cnae_grupo
cnae_classe
cnae_subclasse
```

Regras:

- Começar pelo nível mais agregado.
- Descer na hierarquia apenas quando houver volume suficiente.
- Agrupar categorias raras por similaridade de risco e de negócio.
- Documentar a regra de fallback para CNAEs novos ou raros.

### 5.3 Variáveis financeiras e cadastrais

Manter nomes claros e documentados:

```text
faturamento
tempo_vida_empresa
tempo_relacionamento_banco
limite_credito
valor_utilizado
atraso_atual
atraso_maximo
qtd_atrasos
```

Se a base já tiver nomes diferentes, criar um dicionário de mapeamento no início do projeto.

---

## 6. Estrutura de Pastas e Artefatos

Sugestão de organização:

```text
data/
  00_bronze/   # dados brutos e splits iniciais
  01_prata/    # dados tratados e com features
  02_gold/     # bases WoE, bins, modelo e métricas
reports/       # gráficos, relatórios e tabelas de auditoria
notebooks/     # pipeline numerado
prompts/       # diretrizes para agentes
```

Artefatos mínimos esperados:

```text
data/01_prata/treino_tratado.parquet
data/01_prata/validacao_tratado.parquet
data/01_prata/teste_tratado.parquet
data/01_prata/treino_fe.parquet
data/01_prata/validacao_fe.parquet
data/01_prata/teste_fe.parquet
data/02_gold/treino_woe.parquet
data/02_gold/validacao_woe.parquet
data/02_gold/teste_woe.parquet
data/02_gold/treino_bins.parquet
data/02_gold/validacao_bins.parquet
data/02_gold/teste_bins.parquet
data/02_gold/binning_process.pkl
data/02_gold/binning_summary.csv
data/02_gold/binning_tables.csv
data/02_gold/variaveis_selecionadas.json
data/02_gold/modelo_logistico.pkl
data/02_gold/metricas_modelo.csv
data/02_gold/scores_teste.parquet
data/02_gold/psi_por_safra.csv
```

---

## 7. Estrutura de Notebooks

### 7.1 `00_split_dos_dados.ipynb`

Responsabilidade:

- carregar dados brutos;
- validar tipos, duplicidades, nulos e ranges;
- criar split temporal por safra;
- ajustar tratamento básico no treino;
- aplicar o mesmo tratamento em treino, validação e teste;
- salvar bases tratadas na prata.

Regras:

- split antes do tratamento que aprende parâmetros;
- limits/caps calculados somente no treino;
- não sobrescrever bases brutas;
- registrar contagem de linhas e bad rate por safra.

Saídas:

- `*_tratado.parquet`;
- `limites_tratamento.json`.

### 7.2 `01_engenharia_atributos.ipynb`

Responsabilidade:

- criar features numéricas, financeiras, comportamentais, geográficas e de CNAE;
- criar flags e contagens relevantes;
- manter `ID`, `SAFRA` e target na base para rastreabilidade;
- não usar target na criação das features.

Exemplos de features:

- agregados de faturamento e pagamento;
- utilização de limite;
- percentual pago;
- frequência e severidade de atrasos;
- tempo de relacionamento;
- hierarquia de município/bairro/CEP;
- hierarquia de CNAE;
- flags de nulos, valores negativos e categorias raras.

Regras:

- usar `.assign` e funções pequenas;
- evitar `for` para lógica específica de coluna;
- documentar regras de negócio;
- salvar `*_fe.parquet`.

### 7.3 `02_eda_behavior_score.ipynb`

Responsabilidade:

- EDA sobre o treino, usando a base com features;
- analisar bad rate por faixa, safra, município, bairro e CNAE;
- identificar outliers, nulos, categorias raras e possíveis inconsistências;
- calcular IV/WoE preliminares;
- comparar distribuições entre treino, validação e teste.

Regras:

- não ajustar transformações definitivas aqui;
- usar apenas o treino para decisões;
- gráficos devem ser claros e executivos;
- registrar insights de negócio.

### 7.4 `03_binning_iv_woe.ipynb`

Responsabilidade:

- definir variáveis numéricas e categóricas;
- ajustar binning supervisionado no treino;
- calcular IV, WoE, bad rate, volume e monotonicidade por bin;
- transformar treino, validação e teste com o mesmo objeto;
- salvar bins por cliente para auditoria;
- salvar o `binning_process`.

Regras:

- `BinningProcess` ajustado somente no treino;
- usar variáveis categóricas para geografia e CNAE;
- controlar `min_prebin_size` e `max_n_bins`;
- definir `special_codes` para nulos, -1 e códigos especiais;
- não permitir bins com volume muito baixo;
- salvar tabela por bin.

### 7.5 `04_selecao_variaveis.ipynb`

Responsabilidade:

- filtrar variáveis por IV mínimo;
- remover redundância;
- calcular VIF nas colunas WoE;
- avaliar Cramér's V para categóricas antes do WoE, quando necessário;
- salvar lista final de variáveis.

Critérios sugeridos:

- IV mínimo: `0.02` para triagem; `0.10` para candidatas fortes;
- correlação de Spearman: `|rho| > 0.70` indica redundância;
- Cramér's V: `> 0.70` indica associação forte entre categóricas;
- VIF: `< 5` aceitável; `5–10` atenção; `> 10` multicolinearidade severa.

Regras:

- calcular VIF após WoE;
- remover iterativamente a maior VIF;
- considerar importância de negócio e estabilidade, não apenas estatística;
- testar novamente a estabilidade por safra.

### 7.6 `05_modelagem_avaliacao.ipynb`

Responsabilidade:

- treinar modelo baseline logístico com WoE;
- opcionalmente treinar challenger CatBoost;
- avaliar AUC, KS, Gini, calibração e matriz de confusão;
- avaliar por safra, CNAE, município e faixas de faturamento;
- gerar scores e relatório de desempenho.

Regras:

- treinar somente no treino;
- escolher threshold apenas na validação;
- aplicar no teste/OOT sem ajuste;
- não usar teste para selecionar variáveis ou hiperparâmetros;
- reportar métricas por safra e por segmento.

### 7.7 `06_auditoria_monitoramento.ipynb`

Responsabilidade:

- gerar tabelas de bins para auditoria;
- gerar painéis visuais por variável;
- calcular PSI por variável e por safra;
- comparar previsto vs observado;
- avaliar estabilidade e fairness;
- documentar limitações e decisões.

Regras:

- usar bins fixos definidos no treino;
- não recalcular bins no monitoramento;
- reportar categorias novas e fallback;
- gerar material para comitê e auditoria.

---

## 8. Tratamento de Dados

### 8.1 Regras determinísticas

Podem ser aplicadas em todos os conjuntos:

- valores negativos impossíveis;
- códigos de ausência;
- padronização de CEP, CNPJ e CNAE;
- normalização de strings e categorias.

### 8.2 Regras que aprendem parâmetros

Devem ser ajustadas somente no treino:

- percentis e caps;
- limites de winsorização;
- bins;
- WoE;
- target encoding;
- agrupamento de categorias raras;
- mediana/média para imputação.

### 8.3 Nulos e códigos especiais

- não imputar automaticamente nulos com significado de negócio;
- criar bin ou categoria dedicada;
- documentar `special_codes`;
- manter rastreabilidade entre valor bruto e bin.

---

## 9. Regras Específicas para Geografia e CNAE

### 9.1 Volume mínimo

- Exigir volume mínimo por categoria/bin.
- Sugestão inicial: pelo menos 100 empresas ou 5% da base, o que for mais restritivo.
- Categorias abaixo do mínimo devem ser agrupadas em `OUTROS` ou no nível hierárquico superior.

### 9.2 Hierarquia e fallback

```text
CEP → bairro → município → UF
CNAE subclasse → classe → grupo → divisão → seção
```

Se uma categoria não existir em produção, usar o nível superior.

### 9.3 Estabilidade

- comparar bad rate e participação por safra;
- calcular PSI por categoria;
- monitorar municípios e CNAEs novas;
- evitar decisões baseadas em categorias com histórico curto.

### 9.4 Fairness e regulação

Geografia pode funcionar como proxy de características protegidas. Antes de usar:

- avaliar restrições legais e regulatórias;
- documentar a justificativa de negócio;
- testar impacto em grupos;
- evitar política de crédito baseada apenas em localização.

---

## 10. Binning e WoE

### 10.1 Variáveis numéricas

- usar binning supervisionado com monotonicidade;
- tratar outliers e nulos;
- validar se os cortes fazem sentido de negócio;
- comparar com bins de quantis apenas como diagnóstico.

### 10.2 Variáveis categóricas

- usar binning categórico para agrupar níveis de risco semelhante;
- exigir volume mínimo;
- documentar agrupamentos;
- evitar categorias com WoE extremo e pouco volume.

### 10.3 Saídas obrigatórias

Para cada variável:

- tabela com faixa, N, % base, bads, goods, bad rate, WoE, IV do bin;
- IV total;
- número de bins;
- monotonicidade;
- estabilidade por safra.

---

## 11. Modelagem

### 11.1 Baseline logístico

- usar WoE como entrada;
- coeficientes interpretáveis;
- reportar significância e sinal dos coeficientes;
- validar calibração.

### 11.2 Challenger CatBoost

- usar quando o objetivo for performance ou interações complexas;
- tratar categóricas com cuidado;
- usar SHAP para explicar ofensores;
- comparar com o baseline, sem perder auditabilidade.

### 11.3 Métricas

- AUC;
- KS;
- Gini;
- calibração;
- precision/recall no threshold escolhido;
- métricas por safra;
- métricas por CNAE, município e faixa de faturamento;
- PSI do score.

---

## 12. Identificação de Ofensores

O agente deve combinar:

1. **Ranking global:** IV, importância de variável e coeficientes.
2. **Análise por faixa:** bad rate e WoE por bin.
3. **Análise por segmento:** CNAE, município, bairro e porte.
4. **Análise multivariada:** SHAP ou contribuição no logito.
5. **Estabilidade:** ofensores devem ser persistentes ao longo das safras.
6. **Validação de negócio:** um ofensor estatístico deve fazer sentido operacional.

Cuidado:

- correlação não implica causalidade;
- um segmento com bad rate alto pode ter pouco volume;
- variáveis de comportamento podem ser consequência do evento, não causa;
- não confundir preditor com ofensor causal.

---

## 13. Análise Visual e Auditoria

Gerar, no mínimo:

- painel por variável: volume, bad rate e WoE por bin;
- ranking de IV;
- heatmap bin × safra;
- PSI por variável e por safra;
- dispersão volume × bad rate;
- previsto vs observado por faixa de score;
- contribuição dos bins no score;
- tabela de bins para auditoria.

Cada gráfico deve ter:

- título claro;
- eixos identificados;
- legenda;
- destaque para bins raros ou não monotônicos;
- informação de IV e número de bins.

---

## 14. Monitoramento

- PSI por variável e por safra;
- acompanhamento de categorias novas;
- bad rate observado vs previsto;
- KS/AUC por safra;
- taxa de override e justificativa;
- alertas para mudança de mix de carteira;
- versionamento do modelo e do binning.

---

## 15. Regras de Código

- manter notebooks lineares e legíveis;
- usar nomes claros e consistentes;
- evitar abstrações desnecessárias;
- preferir `.assign` para features;
- usar funções pequenas para binning, seleção e métricas;
- não criar classes ou pipelines complexos sem necessidade;
- salvar bases intermediárias em parquet;
- salvar objetos em pickle ou joblib;
- registrar seeds;
- validar shapes, nulos, ranges e contagens após cada etapa.

---

## 16. Checklist de Aceite

O pipeline só é considerado concluído quando:

- [ ] o split temporal está documentado;
- [ ] nenhuma transformação supervisionada foi ajustada fora do treino;
- [ ] treino, validação e teste usam o mesmo tratamento e binning;
- [ ] bins têm volume mínimo e regras de fallback;
- [ ] tabelas de WoE/IV foram geradas;
- [ ] VIF foi calculado após WoE;
- [ ] o teste não foi usado para seleção ou tuning;
- [ ] métricas foram reportadas por safra e por segmento;
- [ ] PSI e estabilidade foram calculados;
- [ ] artefatos de auditoria foram salvos;
- [ ] limitações e decisões de negócio foram documentadas.

---

## 17. Comportamento Esperado do Agente

1. Antes de codar, listar as definições pendentes.
2. Não inventar target, safra, colunas ou regras de negócio.
3. Se faltar informação, criar placeholders explícitos.
4. Explicar cada decisão de transformação.
5. Validar o pipeline de ponta a ponta antes de concluir.
6. Reportar resultados com métricas, estabilidade e limitações.
7. Nunca usar o teste para melhorar o modelo.
8. Manter o código o mais simples possível.

---

## 18. Entregável Final

Ao concluir, entregar:

- notebooks executados sem erro;
- bases tratadas, com features, WoE e bins;
- `binning_process.pkl`;
- tabelas de IV/WoE por variável;
- lista de variáveis selecionadas;
- modelo treinado;
- métricas por base e por safra;
- análise de ofensores;
- relatório de estabilidade e auditoria;
- resumo executivo com recomendações de política de crédito.
