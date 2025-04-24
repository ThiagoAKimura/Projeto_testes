from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from deep_translator import GoogleTranslator
import os
from db_models import SessionLocal, Registro

import matplotlib
matplotlib.use('Agg')  # <- isso aqui desativa o uso da interface Tkinter
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- Flask App ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}}, supports_credentials=True)

# --- Modelo HuggingFace ---
modelo_huggingface = "GABRYEL25770/TrainedModel"
tokenizer = T5Tokenizer.from_pretrained(modelo_huggingface)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tradutor ---
#translator = Translator()

# --- Funções ---
def traduzir_para_ingles(texto_pt):
    return GoogleTranslator(source='pt', target='en').translate(texto_pt)

def predict_sentiment(text):
    texto_en = traduzir_para_ingles(text)

    input_text = f"classify sentiment: {texto_en}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True).to(device)

    model = T5ForConditionalGeneration.from_pretrained(modelo_huggingface).to(device)

    if torch.cuda.is_available():
        model.half()

    model.eval()
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=20, num_beams=3)

    sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)

    del model
    torch.cuda.empty_cache()

    return sentiment

# --- Rota principal ---
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "Texto vazio"}), 400

    sentimento = predict_sentiment(user_text)

    return jsonify({"sentiment": sentimento})

@app.route("/save", methods=["POST", "OPTIONS"])
def save():
    if request.method == "OPTIONS":
        return '', 200  # responde ao preflight
    
    data = request.json
    texto = data.get("text", "")
    sentimento = data.get("sentiment", "")

    if not texto or not sentimento:
        return jsonify({"error": "Texto e sentimento são obrigatórios."}), 400

    db = SessionLocal()
    novo_registro = Registro(texto=texto, sentimento=sentimento)
    db.add(novo_registro)
    db.commit()
    db.close()

    return jsonify({"message": "Registro salvo com sucesso!"})

@app.route("/grafico", methods=["GET"])
def grafico():
    db = SessionLocal()
    registros = db.query(Registro).all()
    db.close()

    # Conta os sentimentos
    contagem = {}
    for reg in registros:
        sentimento = reg.sentimento
        contagem[sentimento] = contagem.get(sentimento, 0) + 1

    if not contagem:
        return jsonify({"image_base64": None, "error": "Nenhum dado para exibir o gráfico."})


    # Dados para o gráfico
    labels = list(contagem.keys())
    sizes = list(contagem.values())
    colors = plt.cm.Paired.colors

    # Criando o gráfico
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.axis("equal")
    plt.title("Distribuição dos Sentimentos")

    # Convertendo imagem para base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close(fig)

    return jsonify({"image_base64": image_base64})

# --- Rota para exportar dados para Excel ---
@app.route("/exportar_excel", methods=["GET"])
def exportar_excel():
    db = SessionLocal()
    registros = db.query(Registro).all()
    db.close()

    if not registros:
        return jsonify({"error": "Nenhum dado para exportar."}), 404

    # Criando um DataFrame com os dados do banco
    data = [{"texto": reg.texto, "sentimento": reg.sentimento} for reg in registros]
    df = pd.DataFrame(data)

    # Salvando o DataFrame em um arquivo Excel em memória
    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    # Enviando o arquivo Excel como resposta
    return send_file(output, as_attachment=True, download_name="dados_sentimentos.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# --- Início ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
