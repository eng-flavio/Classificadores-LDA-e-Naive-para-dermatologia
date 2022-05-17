# Classificadores-LDA-e-Naive-para-dermatologia
 Aplicação dos classificadores  supervisionados LDA e Naive em dermatlogia feito em Scilab

## Pré-requisitos 
 - [Scilab](https://www.scilab.org/)
 - [Dataset dermatology](https://archive.ics.uci.edu/ml/datasets/dermatology)
## Considerações
 - Os atributos da coluna 34 referentes à base “dermatology.data” estão ausentes das amostras 34-36 e 263-265 e por isso foram removidos da base, não participando assim do treinamento nem dos testes realizados
 - Foi definido como função discriminante (função de custo) a distância da amostra para com as médias após a projeção
 - Para a validação hold out foi definido uma amostragem de 20 (5,5%) amostras para teste
 - Na versão simples do LDA é considerado que a matriz de covariâncias é igual para todas as classes
 - A versão Naive Bayes do LDA considera que a matriz de covariância é apenas uma matriz diagonal das variâncias, a hipótese de independência entre os atributos e adota uma normalização por Z-score no processamento dos dados

## Instruções de uso
 - Foram utilizadas validações cruzadas *hould-out* e *leave-one-out* para cada um dos classificadore,s portanto utilize o arquivo correspondente a cada uma
