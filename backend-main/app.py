import pandas as pd
import matplotlib.pyplot as plt
import random
from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator # type: ignore
import os
from db_models import SessionLocal, Registro

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
translator = Translator()

# --- Funções ---
def traduzir_para_ingles(texto_pt):
    traducao = translator.translate(texto_pt, src='pt', dest='en')
    return traducao.text

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


@app.route("/gerar_relatorio", methods=["GET"])
def gerar_relatorio():

    disciplinas = ['Cálculo I', 'Física I', 'Algoritmos', 'Engenharia Econômica']
    semestres = ['2023/1', '2023/2', '2024/1', '2024/2']

    dados = []
    for disciplina in disciplinas:
        for semestre in semestres:
            positivo = random.randint(40, 80)
            negativo = random.randint(5, 30)
            neutro = max(0, 100 - positivo - negativo)
            dados.append([disciplina, semestre, positivo, negativo, neutro])

    df = pd.DataFrame(dados, columns=['Disciplina', 'Semestre', 'Positivo', 'Negativo', 'Neutro'])
    df.to_excel("static/relatorio_sentimentos.xlsx", index=False)

    for disciplina in disciplinas:
        df_disc = df[df['Disciplina'] == disciplina]
        media_positivo = df_disc['Positivo'].mean()
        media_negativo = df_disc['Negativo'].mean()
        media_neutro = df_disc['Neutro'].mean()

        plt.figure(figsize=(6, 6))
        plt.pie([media_positivo, media_negativo, media_neutro],
                labels=['Positivo', 'Negativo', 'Neutro'],
                autopct='%1.1f%%',
                startangle=90,
                colors=['#66bb6a', '#ef5350', '#ffee58'])
        plt.title(f'Distribuição de Sentimentos - {disciplina}')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"static/{disciplina.replace(' ', '_')}.png")
        plt.close()

    return jsonify({"mensagem": "Relatório e gráficos gerados com sucesso!"})



# --- Início ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


