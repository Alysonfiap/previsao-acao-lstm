📈 Previsão de Preço de Ações com LSTM + API Flask

Este projeto implementa um modelo de rede neural LSTM para prever preços de ações utilizando a biblioteca TensorFlow/Keras.
Após o treinamento, o modelo é exposto através de uma API REST em Flask, permitindo o consumo das previsões via requisições HTTP.

O ativo utilizado como exemplo é a Disney (DIS), com dados históricos coletados do Stooq.

🧠 Tecnologias Utilizadas
🔹 Modelagem e Treinamento

Python

Pandas

Numpy

Pandas DataReader

TensorFlow / Keras

Scikit-learn

Joblib

🔹 API

Flask

Logging

📁 Estrutura do Projeto
.
├── treinar_modelo.py
├── api.py
├── modelo_lstm_acao.h5   ← gerado após o treino
├── scaler.pkl            ← gerado após o treino
├── monitoramento.log     ← gerado pela API
└── README.md

🚀 Funcionalidade do Projeto
✔ Treinamento do Modelo

O script:

✅ baixa dados históricos da ação DIS
✅ usa somente o preço de fechamento
✅ normaliza os dados
✅ cria janelas de 60 dias
✅ treina um modelo LSTM
✅ avalia desempenho (MAE, RMSE, MAPE, R²)
✅ salva o modelo e o scaler para uso posterior

✔ API Flask

A API:

✅ carrega o modelo e o scaler
✅ recebe uma lista com últimos até 60 preços
✅ ajusta automaticamente o tamanho
✅ retorna a previsão do próximo preço

Tudo com logs de monitoramento.

🔧 Como Executar o Projeto
1️⃣ Criar Ambiente Virtual (Opcional, mas recomendado)
python -m venv venv


Ativar:

Windows:

venv\Scripts\activate


Linux/Mac:

source venv/bin/activate

2️⃣ Instalar Dependências
pip install pandas pandas_datareader numpy scikit-learn tensorflow flask joblib

3️⃣ Treinar o Modelo

Execute:

python treinar_modelo.py


Após o treino serão gerados:

modelo_lstm_acao.h5
scaler.pkl

4️⃣ Rodar a API
python api.py


A API ficará disponível em:

http://127.0.0.1:5000/

📡 Endpoints da API
🔹 Verificar status
GET /


Resposta:

{
  "status": "online",
  "msg": "API LSTM funcionando!"
}

🔹 Fazer previsão (GET)
GET /prever?precos=10,11,12,13


📌 Você pode enviar 1 a 60 valores
📌 Se enviar menos de 60, o sistema completa automaticamente
📌 Se enviar mais de 60, usa apenas os últimos

Resposta:

{
  "preco_previsto": 102.55,
  "moeda": "USD",
  "valores_processados": 60
}

🔹 Fazer previsão (POST)
POST /prever

Exemplo JSON
{
  "precos": [10, 11, 12, 13]
}

📊 Métricas Calculadas no Treino

MAE – Erro médio absoluto

RMSE – Raiz do erro quadrático médio

MAPE – Erro percentual

R² – Coeficiente de determinação

Exemplo de saída:

MAE: 1.25
RMSE: 2.10
MAPE: 3.68%
Acurácia aproximada: 96.32%
R²: 0.87

🧪 Lógica da Preparação dos Dados

📌 Os últimos 60 dias são usados como entrada
📌 O modelo prevê o próximo preço (dia 61)
📌 Os dados são normalizados com MinMaxScaler
📌 Após previsão, os valores são desnormalizados

🔐 Tratamento de Erros da API

A API retorna mensagens claras em caso de:

❌ Lista vazia
❌ JSON inválido
❌ Falha no modelo
❌ Valores não numéricos

Exemplo:

{
  "erro": "Use /prever?precos=10,20,30"
}

📜 Logs de Monitoramento

Os logs são gravados em:

monitoramento.log


Incluindo:

✔ inicialização
✔ erros
✔ requisições
✔ previsões geradas


💡 Possíveis Melhorias Futuras

✨ inclusão de mais features (Open, High, Volume etc.)
✨ tuning de hiperparâmetros
✨ salvar logs em banco de dados
✨ adicionar autenticação na API
✨ criar frontend
✨ deploy em nuvem (Render / AWS / Azure / GCP)

👨‍💻 Autor

Projeto desenvolvido por Alyson Alves
📌 Focado em Data Science • Machine Learning • APIs

