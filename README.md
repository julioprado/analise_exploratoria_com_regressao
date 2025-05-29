# Projeto de Análise de Dados de Vendas e Salários

[](https://www.google.com/search?q=%5Bhttps://www.google.com/search%3Fq%3DLINK_PARA_SEU_NOTEBOOK_COLAB%5D\(https://www.google.com/search%3Fq%3DLINK_PARA_SEU_NOTEBOOK_COLAB\))

Este projeto realiza uma análise exploratória de dados de vendas ao longo do tempo e uma análise de regressão linear para prever salários com base em anos de experiência. O código foi desenvolvido no Google Colaboratory, facilitando a execução e o compartilhamento da análise.

## Índice

1.  **Descrição do Projeto**
2.  **Conjunto de Dados**
      * Descrição do Dataset de Vendas (`sales data-set.csv`)
      * Descrição do Dataset de Salários (`Salary_dataset.csv`)
3.  **Como Clonar o Repositório**
4.  **Instalação das Bibliotecas**
5.  **Explicação do Código**
      * Importação de Bibliotecas
      * Upload do Arquivo de Vendas
      * Leitura do Arquivo CSV de Vendas
      * Visualização Inicial dos Dados de Vendas
      * Conversão da Coluna de Data para o Formato Datetime (Vendas)
      * Obtenção das Datas Mínima e Máxima (Vendas)
      * Extração do Mês e Ano (Vendas)
      * Cálculo das Vendas Mensais (Vendas)
      * Visualização das Vendas Mensais ao Longo do Tempo
      * Filtragem de Dados de Vendas para Fevereiro de 2010
      * Cálculo da Soma das Vendas Semanais Filtradas
      * Leitura do Arquivo CSV de Salários
      * Cálculo da Correlação entre Anos de Experiência e Salário
      * Visualização da Linearidade entre Anos de Experiência e Salário
      * Detecção de Outliers na Coluna de Salário (Boxplot)
      * Detecção de Outliers na Coluna de Anos de Experiência (Boxplot)
      * Identificação de Outliers na Coluna de Salário (IQR)
      * Identificação de Outliers na Coluna de Anos de Experiência (IQR)
      * Preparação dos Dados para Regressão Linear
      * Divisão dos Dados em Conjuntos de Treino e Teste
      * Criação e Treinamento do Modelo de Regressão Linear
      * Realização de Previsões no Conjunto de Teste
      * Avaliação do Modelo de Regressão Linear
      * Previsão de Salário para 22 Anos de Experiência
      * Código Refatorado (Organizado em Funções)
6.  **Glossário**
7.  **Próximos Passos**
8.  **Contribuição**
9.  **Licença**

## 1\. Descrição do Projeto

Este projeto aborda duas análises distintas utilizando Python e bibliotecas populares de ciência de dados. A primeira parte foca na análise de dados de vendas, explorando a tendência das vendas semanais ao longo do tempo, com um olhar específico para um período selecionado. A segunda parte concentra-se na construção de um modelo de regressão linear para prever o salário com base nos anos de experiência, além de realizar uma análise de correlação e detecção de outliers no dataset de salários. O objetivo é demonstrar o uso de Pandas para manipulação de dados, Matplotlib e Seaborn para visualização, e Scikit-learn para modelagem preditiva.

## 2\. Conjunto de Dados

Este projeto utiliza dois arquivos de dados distintos:

### Descrição do Dataset de Vendas (`sales data-set.csv`)

Este dataset contém informações sobre vendas semanais ao longo do tempo. As colunas relevantes para a análise incluem:

  * **Date:** A data da venda semanal.
  * **Weekly\_Sales:** O valor das vendas naquela semana.
  * **Outras colunas:** O dataset pode conter outras colunas não diretamente utilizadas nesta análise (como Store, Temperature, Fuel\_Price, etc.).

### Descrição do Dataset de Salários (`Salary_dataset.csv`)

Este dataset contém informações sobre salários e anos de experiência. As colunas relevantes são:

  * **YearsExperience:** O número de anos de experiência do indivíduo.
  * **Salary:** O salário correspondente aos anos de experiência.

## 3\. Como Clonar o Repositório

Para obter uma cópia local deste projeto, você pode clonar o repositório do GitHub utilizando o seguinte comando no seu terminal:

```bash
git clone [URL_DO_SEU_REPOSITÓRIO]
```

Certifique-se de substituir `[URL_DO_SEU_REPOSITÓRIO]` pelo link real do seu repositório no GitHub.

## 4\. Instalação das Bibliotecas

As bibliotecas necessárias para executar este projeto são:

  * **pandas:** Para manipulação e análise de dados tabulares.
  * **matplotlib:** Para criação de gráficos e visualizações básicas.
  * **seaborn:** Para criação de gráficos estatísticos mais avançados.
  * **scikit-learn (sklearn):** Para tarefas de aprendizado de máquina, como regressão linear e divisão de dados.
  * **google-colab:** Para funcionalidades específicas do Google Colaboratory (como o upload de arquivos).

Você pode instalar todas essas bibliotecas utilizando o `pip`, o gerenciador de pacotes do Python. Execute o seguinte comando no seu terminal:

```bash
pip install pandas matplotlib seaborn scikit-learn google-colab
```

Se você estiver utilizando um ambiente virtual (recomendado), certifique-se de que o ambiente esteja ativado antes de executar o comando.

## 5\. Explicação do Código

O código presente no notebook do Google Colab é dividido em blocos lógicos, cada um responsável por uma etapa da análise. Abaixo, uma explicação detalhada de cada bloco:

### Importação de Bibliotecas

Este bloco importa as bibliotecas necessárias para o projeto.

```python
from google.colab import files
import pandas as pd
```

> `from google.colab import files`: Importa o módulo `files` do Google Colaboratory, que permite interagir com o sistema de arquivos do ambiente Colab, como fazer upload de arquivos.
>
> `import pandas as pd`: Importa a biblioteca Pandas e a atribui ao alias `pd`. Pandas é fundamental para trabalhar com DataFrames, que são estruturas de dados tabulares.

### Upload do Arquivo de Vendas

Este bloco permite que o usuário faça o upload do arquivo de dados de vendas diretamente no ambiente do Google Colab.

```python
# Fazer o upload do arquivo
uploaded = files.upload()
```

> `uploaded = files.upload()`: Chama a função `upload()` do módulo `files`. Ao executar esta linha, uma interface gráfica será exibida para que o usuário possa selecionar e fazer o upload do arquivo `sales data-set.csv`. A variável `uploaded` armazenará informações sobre os arquivos carregados.

### Leitura do Arquivo CSV de Vendas

Este bloco carrega o arquivo CSV de vendas para um DataFrame do Pandas.

```python
# Ler o arquivo CSV
df_sales = pd.read_csv('sales data-set.csv')
```

> `df_sales = pd.read_csv('sales data-set.csv')`: Utiliza a função `read_csv()` do Pandas para ler o conteúdo do arquivo `sales data-set.csv` e armazená-lo em um DataFrame chamado `df_sales`.

```python
data = df_sales
```

> `data = df_sales`: Atribui o DataFrame `df_sales` a uma nova variável chamada `data`. Isso é feito para simplificar o uso do DataFrame nas etapas seguintes.

### Visualização Inicial dos Dados de Vendas

Este bloco exibe as primeiras linhas e os nomes das colunas do DataFrame de vendas para uma inspeção inicial.

```python
# Display the first 5 rows and the column names
data_head = data.head()
data_columns = data.columns.tolist()

data_columns

data_head
```

> `data_head = data.head()`: A função `head()` exibe as 5 primeiras linhas do DataFrame `data`, permitindo visualizar as colunas e alguns exemplos de dados.
>
> `data_columns = data.columns.tolist()`: O atributo `columns` retorna um objeto Index contendo os nomes das colunas. `.tolist()` converte esse objeto para uma lista.
>
> As linhas subsequentes (`data_columns` e `data_head`) simplesmente exibem essas variáveis na saída do Colab.

### Conversão da Coluna de Data para o Formato Datetime (Vendas)

Este bloco converte a coluna 'Date' para o formato datetime do Pandas, o que é essencial para realizar operações temporais.

```python
# Converty the Date column to datetime format for acurate comparison
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
```

> `data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')`: Utiliza a função `to_datetime()` do Pandas para converter os valores da coluna 'Date' para o tipo datetime. O argumento `format='%d/%m/%Y'` especifica o formato em que as datas estão originalmente no arquivo CSV (dia/mês/ano).

### Obtenção das Datas Mínima e Máxima (Vendas)

Este bloco encontra a primeira e a última data presentes no dataset de vendas.

```python
# Get the minimum and maximum of the Date column
date_min = data['Date'].min()
date_max = data['Date'].max()

date_min, date_max
```

> `date_min = data['Date'].min()`: A função `min()` retorna a data mais antiga presente na coluna 'Date'.
>
> `date_max = data['Date'].max()`: A função `max()` retorna a data mais recente presente na coluna 'Date'.
>
> A linha seguinte exibe os valores de `date_min` e `date_max`.

### Extração do Mês e Ano (Vendas)

Este bloco cria novas colunas 'Month' e 'Year' no DataFrame, extraindo essas informações da coluna 'Date'.

```python
# Extract the month and year for each date
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
```

> `data['Month'] = data['Date'].dt.month`: Utiliza o acessor `.dt` para acessar propriedades específicas de datetime da coluna 'Date' e extrai o mês, armazenando-o na nova coluna 'Month'.
>
> `data['Year'] = data['Date'].dt.year`: Similarmente, extrai o ano da coluna 'Date' e o armazena na nova coluna 'Year'.

### Cálculo das Vendas Mensais (Vendas)

Este bloco agrupa os dados por ano e mês e calcula a soma das vendas semanais para cada período.

```python
monthly_sales = data.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
```

> `data.groupby(['Year', 'Month'])`: Agrupa o DataFrame `data` pelas colunas 'Year' e 'Month', criando grupos para cada combinação única de ano e mês.
>
> `['Weekly_Sales'].sum()`: Para cada grupo, calcula a soma dos valores na coluna 'Weekly\_Sales'.
>
> `.reset_index()`: Converte o resultado (que é uma Series com um MultiIndex) em um novo DataFrame `monthly_sales` com 'Year' e 'Month' como colunas regulares.

```python
monthly_sales
```

> Exibe o DataFrame `monthly_sales`.

### Visualização das Vendas Mensais ao Longo do Tempo

Este bloco cria um gráfico de linhas para visualizar a tendência das vendas mensais ao longo do tempo.

```python
import matplotlib.pyplot as plt

# Create a datetime column for the montlhy_sales dataset
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['Date'], monthly_sales['Weekly_Sales'], marker='o')
plt.title('Vendas Mensais ao Longo do Tempo')
plt.ylabel('Soma das Vendas Mensais (Weekly_Sales)')
plt.xlabel('Data')
plt.grid(True)
plt.tight_layout()
plt.show()
```

> `import matplotlib.pyplot as plt`: Importa a biblioteca Matplotlib para criar gráficos.
>
> `monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))`: Cria uma nova coluna 'Date' no DataFrame `monthly_sales` combinando as colunas 'Year' e 'Month'. `.assign(day=1)` adiciona um dia (o primeiro do mês) para criar um objeto datetime completo.
>
> `plt.figure(figsize=(12, 6))`: Cria uma nova figura com um tamanho específico (largura 12 polegadas, altura 6 polegadas).
>
> `plt.plot(monthly_sales['Date'], monthly_sales['Weekly_Sales'], marker='o')`: Plota um gráfico de linhas com a coluna 'Date' no eixo x e a coluna 'Weekly\_Sales' no eixo y. `marker='o'` adiciona marcadores circulares aos pontos de dados.
>
> `plt.title('Vendas Mensais ao Longo do Tempo')`: Define o título do gráfico.
>
> `plt.ylabel('Soma das Vendas Mensais (Weekly_Sales)')`: Define o rótulo do eixo y.
>
> `plt.xlabel('Data')`: Define o rótulo do eixo x.
>
> `plt.grid(True)`: Adiciona uma grade ao gráfico.
>
> `plt.tight_layout()`: Ajusta o layout para evitar sobreposição de rótulos.
>
> `plt.show()`: Exibe o gráfico.

### Filtragem de Dados de Vendas para Fevereiro de 2010

Este bloco filtra o DataFrame de vendas para selecionar apenas as entradas correspondentes ao mês de fevereiro do ano de 2010.

```python
filtered_data = data[(data['Month'] == 2) & (data['Year'] == 2010)]
```

> `filtered_data = data[(data['Month'] == 2) & (data['Year'] == 2010)]`: Cria um novo DataFrame `filtered_data` contendo apenas as linhas do DataFrame `data` onde a coluna 'Month' é igual a 2 (fevereiro) e a coluna 'Year' é igual a 2010.

### Cálculo da Soma das Vendas Semanais Filtradas

Este bloco calcula a soma das vendas semanais para os dados filtrados (fevereiro de 2010).

```python
# Calculate the sum of Weekly_Sales for the filtered data
sum_weekly_sales_filtered = filtered_data['Weekly_Sales'].sum()
sum_weekly_sales_filtered
```

> `sum_weekly_sales_filtered = filtered_data['Weekly_Sales'].sum()`: Calcula a soma dos valores na coluna 'Weekly\_Sales' do DataFrame `filtered_data`.
>
> A linha seguinte exibe o valor da soma calculada.

### Leitura do Arquivo CSV de Salários

Este bloco carrega o arquivo CSV de salários para um DataFrame do Pandas.

```python
# Calculating correlation
df_salaries = pd.read_csv('Salary_dataset.csv')
salary_data = df_salaries
```

> `df_salaries = pd.read_csv('Salary_dataset.csv')`: Utiliza a função `read_csv()` do Pandas para ler o conteúdo do arquivo `Salary_dataset.csv` e armazená-lo em um DataFrame chamado `df_salaries`.
>
> `salary_data = df_salaries`: Atribui o DataFrame `df_salaries` a uma nova variável chamada `salary_data`.

### Cálculo da Correlação entre Anos de Experiência e Salário

Este bloco calcula o coeficiente de correlação de Pearson entre as colunas 'YearsExperience' e 'Salary'.

```python
correlation = salary_data['YearsExperience'].corr(salary_data['Salary'])
```

> `correlation = salary_data['YearsExperience'].corr(salary_data['Salary'])`: Utiliza a função `corr()` para calcular a correlação entre as duas colunas especificadas. O resultado é um valor entre -1 e 1, indicando a força e a direção da relação linear.

### Visualização da Linearidade entre Anos de Experiência e Salário

Este bloco cria um gráfico de dispersão para visualizar a relação entre anos de experiência e salário, incluindo o valor da correlação no título.

```python
# Scatter plt to visualize linearity
plt.figure(figsize=(8, 6))
plt.scatter(salary_data['YearsExperience'], salary_data['Salary'], alpha=0.7)
plt.title(f'Sactter Plot: YearsExperience vs. Salary (Correlation: {correlation:.2f})')
plt.ylabel('Salary')
plt.xlabel('Anos de Experiência')
plt.grid(True)
plt.show()
```

> `plt.scatter(salary_data['YearsExperience'], salary_data['Salary'], alpha=0.7)`: Cria um gráfico de dispersão com 'YearsExperience' no eixo x e 'Salary' no eixo y. `alpha=0.7` define a transparência dos pontos.
>
> `plt.title(f'Sactter Plot: YearsExperience vs. Salary (Correlation: {correlation:.2f})')`: Define o título do gráfico, incluindo o valor da correlação formatado para duas casas decimais.

### Detecção de Outliers na Coluna de Salário (Boxplot)

Este bloco cria um boxplot para identificar possíveis outliers na distribuição da coluna 'Salary'.

```python
import seaborn as sns

# Boxplot to identify outliers in Salary
plt.figure(figsize=(8, 6))
sns.boxplot(x=salary_data['Salary
```
