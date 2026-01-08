from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import logging

# ===============================
# CONFIGURAÇÃO DO LOG
# ===============================
logging.basicConfig(
    filename="monitoramento.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("API iniciando...")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_lstm_acao.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ===============================
# Carrega Modelo
# ===============================
try:
    modelo = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Modelo e scaler carregados com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar modelo: {str(e)}")
    raise e


@app.route("/", methods=["GET"])
def home():
    logging.info("Rota / acessada.")
    return jsonify({
        "status": "online",
        "msg": "API LSTM funcionando!"
    })


@app.route("/prever", methods=["GET", "POST"])
def prever():

    logging.info("Requisição recebida em /prever")

    # ===============================
    # Captura dados
    # ===============================
    if request.method == "GET":
        precos_str = request.args.get("precos")

        if not precos_str:
            logging.warning("Chamada inválida sem parâmetro.")
            return jsonify({"erro": "Use /prever?precos=10,20,30"}), 400

        try:
            lista = [float(x) for x in precos_str.split(",")]
        except:
            logging.error("Erro ao converter valores recebidos.")
            return jsonify({"erro": "Valores inválidos"}), 400

    else:
        dados = request.get_json()
        if not dados or "precos" not in dados:
            logging.warning("JSON inválido recebido.")
            return jsonify({"erro": "Envie JSON { 'precos': [...] }"}), 400
        lista = dados["precos"]

    if len(lista) == 0:
        logging.warning("Lista vazia recebida.")
        return jsonify({"erro": "Lista vazia"}), 400

    # ===============================
    # Ajuste para 60 valores
    # ===============================
    original_len = len(lista)

    if len(lista) < 60:
        lista = [lista[0]] * (60 - len(lista)) + lista
    elif len(lista) > 60:
        lista = lista[-60:]

    logging.info(f"Valores recebidos: {original_len} | Valores usados: 60")

    # ===============================
    # Processamento
    # ===============================
    try:
        ultimos = np.array(lista).reshape(-1, 1)
        ultimos_scaled = scaler.transform(ultimos)
        entrada = ultimos_scaled.reshape(1, 60, 1)

        pred = modelo.predict(entrada)
        preco = scaler.inverse_transform(pred)[0][0]

        logging.info(f"Previsão gerada com sucesso: {preco:.2f}")

        return jsonify({
            "preco_previsto": round(float(preco), 2),
            "moeda": "USD",
            "valores_processados": len(lista)
        })

    except Exception as e:
        logging.error(f"Erro no processo de previsão: {str(e)}")
        return jsonify({"erro": "Falha ao processar previsão"}), 500


if __name__ == "__main__":
    logging.info("Servidor Flask iniciado em modo debug.")
    app.run(debug=True)
