import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importar os dados de 'medical_examination.csv' e atribuir ao df
df = pd.read_csv('medical_examination.csv')

# 2. Criar a coluna 'overweight'
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3. Normalizar colesterol e gluc (0 = bom, 1 = ruim)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Desenhar o gráfico categórico na função draw_cat_plot
def draw_cat_plot():
    # 5. Criar o DataFrame para o gráfico categórico usando pd.melt
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Agrupar e formatar os dados para contar cada valor categórico por cardio
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Criar o gráfico categórico usando sns.catplot()
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', 
                      data=df_cat, kind='bar', height=5, aspect=1).fig

    # 8. Salvar a figura como 'catplot.png'
    fig.savefig('catplot.png')
    return fig

# 10. Desenhar o mapa de calor na função draw_heat_map
def draw_heat_map():
    # 11. Limpar os dados, filtrando segmentos incorretos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calcular a matriz de correlação
    corr = df_heat.corr()

    # 13. Gerar a máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Configurar a figura do matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15. Plotar o mapa de calor
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, 
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.5}, ax=ax)

    # 16. Salvar a figura como 'heatmap.png'
    fig.savefig('heatmap.png')
    return fig
