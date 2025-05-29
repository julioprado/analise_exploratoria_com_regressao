# Projeto de Regressão Linear para Previsão de Salários

[](https://www.google.com/search?q=%5Bhttps://www.google.com/search%3Fq%3DLINK_PARA_SEU_NOTEBOOK_COLAB%5D\(https://www.google.com/search%3Fq%3DLINK_PARA_SEU_NOTEBOOK_COLAB\))

Este projeto realiza uma análise de regressão linear simples para prever salários com base nos anos de experiência. O código foi desenvolvido no Google Colaboratory, facilitando a execução e o compartilhamento da análise.

## Índice

1.  **Descrição do Projeto**
2.  **Conjunto de Dados**
      * Descrição do Dataset (`Salary_dataset.csv`)
3.  **Como Clonar o Repositório**
4.  **Instalação das Bibliotecas**
5.  **Explicação do Código**
      * Importação de Bibliotecas
      * Função `load_data()`
      * Função `calculate_correlation()`
      * Função `plot_scatter()`
      * Função `plot_boxplot()`
      * Função `detect_outliers_iqr()`
      * Função `train_linear_regression()`
      * Função `evaluate_model()`
      * Função `predict_salary()`
      * Função `main()`
6.  **Glossário**
7.  **Próximos Passos**
8.  **Contribuição**
9.  **Licença**

## 1\. Descrição do Projeto

Este projeto tem como objetivo construir um modelo de regressão linear simples para prever o salário de um indivíduo com base em seus anos de experiência. Além da modelagem, o projeto inclui etapas de análise exploratória de dados, como cálculo de correlação, visualização da relação entre as variáveis e detecção de outliers. O código é modularizado em funções para melhor organização e reutilização.

## 2\. Conjunto de Dados

### Descrição do Dataset (`Salary_dataset.csv`)

O dataset utilizado neste projeto é o `Salary_dataset.csv`. Ele contém duas colunas relevantes para a análise:

  * **YearsExperience:** O número de anos de experiência do indivíduo (variável independente).
  * **Salary:** O salário correspondente aos anos de experiência (variável dependente).

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
  * **scikit-learn (sklearn):** Para tarefas de aprendizado de máquina, como regressão linear, divisão de dados e métricas de avaliação.

Você pode instalar todas essas bibliotecas utilizando o `pip`, o gerenciador de pacotes do Python. Execute o seguinte comando no seu terminal:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

Se você estiver utilizando um ambiente virtual (recomendado), certifique-se de que o ambiente esteja ativado antes de executar o comando.

## 5\. Explicação do Código

O código deste projeto é organizado em diversas funções para realizar tarefas específicas. Abaixo, uma explicação detalhada de cada função e do bloco principal (`if __name__ == "__main__":`).

### Importação de Bibliotecas

Este bloco importa as bibliotecas necessárias para o projeto.

```python
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

> `from sklearn.metrics import mean_squared_error, r2_score`: Importa as métricas de avaliação de modelos de regressão: erro quadrático médio (MSE) e coeficiente de determinação (R²).
>
> `import pandas as pd`: Importa a biblioteca Pandas e a atribui ao alias `pd`.
>
> `import matplotlib.pyplot as plt`: Importa o módulo `pyplot` da biblioteca Matplotlib e o atribui ao alias `plt`.
>
> `import seaborn as sns`: Importa a biblioteca Seaborn e a atribui ao alias `sns`.
>
> `from sklearn.linear_model import LinearRegression`: Importa a classe `LinearRegression` para criar o modelo de regressão linear.
>
> `from sklearn.model_selection import train_test_split`: Importa a função `train_test_split` para dividir os dados em conjuntos de treino e teste.

### Função `load_data(filepath: str) -> pd.DataFrame`

```python
def load_data(filepath: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um arquivo CSV."""
    return pd.read_csv(filepath)
```

> Esta função recebe o caminho do arquivo CSV (`filepath`) como argumento e utiliza a função `pd.read_csv()` do Pandas para carregar os dados do arquivo em um DataFrame. A função retorna o DataFrame carregado.

### Função `calculate_correlation(data: pd.DataFrame, col1: str, col2: str) -> float`

```python
def calculate_correlation(data: pd.DataFrame, col1: str, col2: str) -> float:
    """Calcula a correlação entre duas colunas."""
    return data[col1].corr(data[col2])
```

> Esta função recebe um DataFrame (`data`) e os nomes de duas colunas (`col1` e `col2`) como argumentos. Ela utiliza o método `.corr()` do Pandas Series para calcular o coeficiente de correlação de Pearson entre as duas colunas especificadas e retorna esse valor.

### Função `plot_scatter(data: pd.DataFrame, x_col: str, y_col: str, correlation: float) -> None`

```python
def plot_scatter(data: pd.DataFrame, x_col: str, y_col: str, correlation: float) -> None:
    """Plota um gráfico de dispersão entre duas variáveis com a correlação."""
    plt.figure(figsize=(8, 6))
    plt.scatter(data[x_col], data[y_col], alpha=0.7)
    plt.title(f'Scatter Plot: {x_col} vs. {y_col} (Correlation: {correlation:.2f})')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()
```

> Esta função recebe um DataFrame (`data`), os nomes das colunas para o eixo x (`x_col`) e o eixo y (`y_col`), e o valor da correlação (`correlation`) como argumentos. Ela cria um gráfico de dispersão utilizando `plt.scatter()` para visualizar a relação entre as duas colunas. O título do gráfico inclui o valor da correlação formatado.

### Função `plot_boxplot(data: pd.Series, title: str) -> None`

```python
def plot_boxplot(data: pd.Series, title: str) -> None:
    """Plota um boxplot para uma série de dados."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data)
    plt.title(title)
    plt.xlabel(data.name)
    plt.grid(True)
    plt.show()
```

> Esta função recebe uma Series do Pandas (`data`) e um título (`title`) como argumentos. Ela utiliza a função `sns.boxplot()` do Seaborn para criar um boxplot da distribuição dos dados na Series.

### Função `detect_outliers_iqr(data: pd.Series) -> pd.DataFrame`

```python
def detect_outliers_iqr(data: pd.Series) -> pd.DataFrame:
    """Identifica outliers em uma série utilizando o método IQR."""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data < lower_bound) | (data > upper_bound)]
```

> Esta função recebe uma Series do Pandas (`data`) e utiliza o método do Intervalo Interquartil (IQR) para identificar outliers. Ela calcula o primeiro quartil (Q1), o terceiro quartil (Q3), o IQR e, em seguida, define os limites inferior e superior para identificar valores que estão fora de 1.5 vezes o IQR abaixo de Q1 ou acima de Q3. A função retorna uma nova Series contendo apenas os valores considerados outliers.

### Função `train_linear_regression(X: pd.DataFrame, y: pd.Series)`

```python
def train_linear_regression(X: pd.DataFrame, y: pd.Series):
    """Treina um modelo de regressão linear e retorna o modelo e as previsões."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred
```

> Esta função recebe um DataFrame de características (`X`) e uma Series da variável alvo (`y`). Ela divide os dados em conjuntos de treinamento e teste utilizando `train_test_split()` com uma proporção de 80% para treino e 20% para teste, e um `random_state` para garantir a reprodutibilidade. Em seguida, cria uma instância do modelo `LinearRegression`, o treina com os dados de treinamento (`model.fit()`), e faz previsões sobre o conjunto de teste (`model.predict()`). A função retorna o modelo treinado, o conjunto de teste das características, o conjunto de teste da variável alvo e as previsões feitas pelo modelo.

### Função `evaluate_model(y_test: pd.Series, y_pred: pd.Series) -> tuple`

```python
def evaluate_model(y_test: pd.Series, y_pred: pd.Series) -> tuple:
    """Avalia o modelo calculando MSE e R²."""
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
```

> Esta função recebe as verdadeiras variáveis alvo do conjunto de teste (`y_test`) e as previsões do modelo (`y_pred`). Ela calcula o Erro Quadrático Médio (MSE) e o Coeficiente de Determinação (R²) utilizando as funções `mean_squared_error()` e `r2_score()` da biblioteca Scikit-learn. A função retorna uma tupla contendo o MSE e o R².

### Função `predict_salary(model: LinearRegression, years_experience: float) -> float`

```python
def predict_salary(model: LinearRegression, years_experience: float) -> float:
    """Prediz o salário baseado em anos de experiência."""
    prediction = model.predict([[years_experience]])[0]
    return round(prediction, 2)
```

> Esta função recebe um modelo de regressão linear treinado (`model`) e um valor de anos de experiência (`years_experience`). Ela utiliza o método `predict()` do modelo para prever o salário correspondente aos anos de experiência fornecidos. A entrada para `predict()` deve ser uma matriz 2D, por isso `[[years_experience]]`. A função retorna a previsão do salário arredondada para duas casas decimais.

### Função `main()`

```python
def main():
    # Carregamento de dados
    data = load_data('Salary_dataset.csv')

    # Cálculo da correlação
    correlation = calculate_correlation(data, 'YearsExperience', 'Salary')

    # Gráficos
    plot_scatter(data, 'YearsExperience', 'Salary', correlation)
    plot_boxplot(data['Salary'], 'Boxplot of Salary (Outlier Detection)')
    plot_boxplot(data['YearsExperience'], 'Boxplot of YearsExperience (Outlier Detection)')

    # Detecção de outliers
    outliers_salary = detect_outliers_iqr(data['Salary'])
    outliers_experience = detect_outliers_iqr(data['YearsExperience'])

    print(f"Outliers em Salary:\n{outliers_salary}")
    print(f"Outliers em YearsExperience:\n{outliers_experience}")

    # Regressão Linear
    X = data[['YearsExperience']]
    y = data['Salary']

    model, X_test, y_test, y_pred = train_linear_regression(X, y)

    # Avaliação do modelo
    mse, r2 = evaluate_model(y_test, y_pred)
    print(f"MSE: {mse:.2f}, R²: {r2:.2f}")
    print(f"Coeficientes: {model.coef_}, Intercepto: {model.intercept_}")

    # Previsão para 22 anos de experiência
    years_experience = 22
    predicted_salary = predict_salary(model, years_experience)
    print(f"Predicted salary for {years_experience} years of experience: ${predicted_salary}")

if __name__ == "__main__":
    main()
```

> A função `main()` coordena a execução das diferentes etapas do projeto:
>
> 1.  **Carregamento de dados:** Chama `load_data()` para carregar o dataset de salários.
> 2.  **Cálculo da correlação:** Chama `calculate_correlation()` para obter a correlação entre anos de experiência e salário.
> 3.  **Gráficos:** Chama `plot_scatter()` e `plot_boxplot()` para visualizar os dados.
> 4.  **Detecção de outliers:** Chama `detect_outliers_iqr()` para identificar outliers nas colunas de salário e anos de experiência e os imprime.
> 5.  **Regressão Linear:** Define as variáveis independentes (`X`) e dependente (`y`), e chama `train_linear_regression()` para treinar o modelo.
> 6.  **Avaliação do modelo:** Chama `evaluate_model()` para obter as métricas de avaliação do modelo e as imprime. Também imprime os coeficientes e o intercepto do modelo.
> 7.  **Previsão:** Define um valor de anos de experiência (22) e chama `predict_salary()` para prever o salário correspondente, imprimindo o resultado.
>
> A condição `if __name__ == "__main__":` garante que a função `main()` seja executada apenas quando o script é rodado diretamente (e não quando é importado como um módulo).

## 6\. Glossário

  * **Correlação:** Uma medida estatística que expressa a extensão em que duas variáveis estão linearmente relacionadas (coeficiente de correlação de Pearson). Varia de -1 a 1, onde 1 indica uma correlação positiva perfeita, -1 indica uma correlação negativa perfeita e 0 indica nenhuma correlação linear.
  * **Gráfico de Dispersão (Scatter Plot):** Um tipo de gráfico que utiliza pontos para representar os valores de duas variáveis, permitindo visualizar a relação entre elas.
  * **Boxplot (Diagrama de Caixa):** Um gráfico que resume a distribuição de um conjunto de dados com base em cinco números: mínimo, primeiro quartil (Q1), mediana (Q2), terceiro quartil (Q3) e máximo. Também pode identificar outliers.
  * **Outlier:** Um valor que se distancia significativamente de outros valores em um conjunto de dados.
  * **IQR (Intervalo Interquartil):** A diferença entre o terceiro quartil (Q3) e o primeiro quartil (Q1) em um conjunto de dados. É uma medida de dispersão estatística robusta a outliers.
  * **Regressão Linear:** Um modelo estatístico que tenta modelar a relação entre uma variável dependente e uma ou mais variáveis independentes ajustando uma equação linear aos dados observados.
  * **Variável Independente (Feature):** Uma variável que é usada para prever o valor de outra variável. Neste caso, 'YearsExperience'.
  * **Variável Dependente (Target):** Uma variável cujo valor está sendo previsto pelo modelo. Neste caso, 'Salary'.
  * **Conjunto de Treinamento:** A parte do dataset usada para treinar o modelo de aprendizado de máquina.
  * **Conjunto de Teste:** A parte do dataset usada para avaliar o desempenho do modelo treinado em dados não vistos.
  * **MSE (Erro Quadrático Médio):** Uma métrica comum para avaliar modelos de regressão. Calcula a média dos quadrados das diferenças entre os valores previstos e os valores reais. Quanto menor o MSE, melhor o modelo se ajusta aos dados.
  * **R² (Coeficiente de Determinação):** Uma métrica que representa a proporção da variância na variável dependente que é previsível a partir da(s) variável(is) independente(s). Varia de 0 a 1, onde 1 indica que o modelo explica toda a variabilidade dos dados.
  * **Coeficientes:** Os valores que multiplicam as variáveis independentes no modelo de regressão linear. Eles indicam a mudança na variável dependente para uma unidade de mudança na variável independente.
  * **Intercepto:** O valor da variável dependente quando todas as variáveis independentes são zero. No contexto deste projeto, seria o salário previsto para 0 anos de experiência.
