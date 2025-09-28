# 🤰 MVP Machine Learning: Classificação de Risco Gestacional

## 🎯 Objetivo do Projeto

Desenvolver um sistema preditivo robusto, utilizando Machine Learning, para classificar o nível de risco gestacional de pacientes (em **Baixo Risco**, **Risco Médio** e **Alto Risco**) com base em indicadores de saúde fisiológicos. O objetivo primário é fornecer uma ferramenta de triagem para auxiliar na tomada de decisão clínica.

## ⚙️ Metodologia e Pipeline de Desenvolvimento

O projeto seguiu um ciclo rigoroso de Machine Learning, que culminou na serialização do modelo de produção:

1.  ### Pré-processamento e Engenharia de Features

      * **Feature Domain:** Criação da variável binária `Idade_35+` baseada em conhecimento de domínio médico (idade gestacional $\ge 35$ anos como fator de risco).
      * **Codificação:** Transformação da variável alvo categórica para numérica (`0`, `1`, `2`) para ser utilizada pelos algoritmos.
      * **Padronização:** Aplicação do `StandardScaler` para avaliar a sensibilidade dos modelos à escala dos dados.

2.  ### Seleção e Comparação de Modelos

      * Seis algoritmos de classificação (KNN, CART, NB, SVM, XGBoost, RNA) foram avaliados via **Cross-Validation (K-Fold = 10)** em três cenários de dados (originais, padronizados e normalizados).
      * O algoritmo **XGBoost** demonstrou a melhor *performance* inicial consistente ($\approx 0.8025$ de acurácia).

3.  ### Otimização e Validação Final

      * O modelo XGBoost foi submetido à otimização de hiperparâmetros (`GridSearchCV`), elevando a acurácia média para $\mathbf{0.8139}$.
      * O modelo final foi validado em um conjunto de teste não visto, focando em métricas de segurança clínica.

## 📊 Principais Descobertas e Resultados

### Achados Clínicos Chave

  * **Preditores Mais Influentes:** A análise de correlação e o *Feature Importance* convergiram, confirmando que o **Nível de Glicose** e a **Pressão Arterial (Sistólica e Diastólica)** são os preditores mais fortes para o risco gestacional.
  * **Idade como Potencializador:** O grupo de gestantes com **Idade $\ge 35$ anos** possui maior prevalência de **Alto Risco**, confirmando a idade como um fator que potencializa o risco em combinação com outras métricas fisiológicas.
  * **Impacto do Pré-processamento:** Modelos sensíveis à escala (SVM, RNA) tiveram um aumento de performance de $\approx \mathbf{10}$ pontos percentuais de acurácia com a Padronização, validando a importância do pré-processamento.

### Performance do Modelo de Produção

O modelo escolhido e exportado foi o **XGBoost Otimizado**.

| Classe (Risco) | Acurácia Média Otimizada | Recall para Alto Risco |
| :--- | :---: | :---: |
| **XGBoost** | $\mathbf{0.8139}$ | $\mathbf{0.95}$ |

**Segurança Clínica (Recall):**
A métrica mais crítica em um contexto de saúde, o **$\mathbf{Recall}$ de $\mathbf{0.95}$ (95%)** para a classe **Alto Risco (Classe 2)**, demonstra que o modelo é altamente eficaz em **identificar corretamente os casos mais graves**, minimizando o perigoso erro de Falso Negativo.

## 🚀 Como Utilizar o Modelo (Prontidão para Produção)

O modelo final de Machine Learning está serializado e pronto para ser carregado e utilizado em um ambiente de produção.

### Arquivo Exportado

  * **`modelo_xgb_otimizado.pkl`**: Pipeline completo contendo o modelo XGBoost treinado no *dataset* integral.

### Exemplo de Uso

Para fazer uma previsão, basta carregar o arquivo `.pkl` e fornecer novos dados com as *features* na ordem correta, como demonstrado no *notebook*:

```python
import joblib
import pandas as pd

# 1. Carregar o modelo serializado
loaded_pipeline = joblib.load('modelo_xgb_otimizado.pkl')

# 2. Novos dados de uma gestante (na mesma ordem das features de treino)
new_data = pd.DataFrame({
    'Idade': [37],
    'Nivel_Glicose': [9.0],
    'Pressao_Sistolica': [112],
    'Pressao_Diastolica': [80],
    'Temperatura_Corporal': [98.5],
    'Frequencia_Cardiaca': [78],
    'Idade_35+': [1] # 1 indica >= 35 anos
})

# 3. Fazer a previsão
prediction = loaded_pipeline.predict(new_data)

# 4. Exibir Resultado
mapeamento_risco = {0: 'Baixo Risco', 1: 'Risco Médio', 2: 'Alto Risco'}
resultado = mapeamento_risco[prediction[0]]

print(f"Previsão de risco: {resultado}")
# Saída: Previsão de risco: Risco Médio
```

## 📚 Dataset e Ferramentas

| Recurso | Descrição | Link |
| :--- | :--- | :--- |
| **Dataset** | Maternal Health Risk Data | [Kaggle](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data?resource=download) |
| **Ferramentas** | Python, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn, Joblib | - |
| **Notebook** | Todo o processo de EDA, ML e Otimização | `Template_Machine_Learning_Pregnancy-5.ipynb` |
