import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Predição de Doença Cardíaca", layout="centered")
st.title("🔍 Predição de Doença Cardíaca")
st.write("Preencha os dados abaixo para prever se o paciente possui risco de doença cardíaca.")

# idade,sexo,pressao_sistolica,colesterol_total,batimentos_em_repouso,fumante,diabetico,historico_familiar,atividade_fisica,estresse,doenca_cardiaca

with st.form(key="form_predicao"):
    idade = st.number_input("Idade", min_value=1, max_value=120, step=1)
    pressao_sistolica = st.number_input("Pressão Sistólica", min_value=50, max_value=300, step=10)
    colesterol = st.number_input("Colesterol", min_value=50, max_value=500, step=10)
    batimentos_em_repouso = st.number_input("Batimentos em Repouso", min_value=20, max_value=200, step=10)
    sexo = st.radio("Sexo", ("Masculino", "Feminino"))
    fumante = st.radio("Fumante?", ("Sim", "Não"))
    diabetico = st.radio("Diabético?", ("Sim", "Não"))
    historico_familiar = st.radio("O paciente tem histórico familiar de pessoas com doença cardíaca?", ("Sim", "Não"))
    atividade_fisica = st.radio("Com que frequência o paciente pratica atividades físicas?", ("Baixo", "Moderado", "Alto"))
    estresse = st.radio("Qual o nível de estresse do paciente?", ("Baixo", "Moderado", "Alto"))

    prever = st.form_submit_button("Prever")

if prever:
    fumante_bin = 1 if fumante == "Sim" else 0
    diabetico_bin = 1 if diabetico == "Sim" else 0
    historico_familiar_bin = 1 if historico_familiar == "Sim" else 0

    sexo_Feminino = 1 if sexo == "Feminino" else 0
    sexo_Masculino = 1 if sexo == "Masculino" else 0

    atividade_fisica_Alta = 1 if atividade_fisica == "Alto" else 0
    atividade_fisica_Moderada = 1 if atividade_fisica == "Moderado" else 0
    atividade_fisica_Baixa = 1 if atividade_fisica == "Baixo" else 0

    estresse_Alto = 1 if estresse == "Alto" else 0
    estresse_Moderado = 1 if estresse == "Moderado" else 0
    estresse_Baixo = 1 if estresse == "Baixo" else 0

    dados = np.array([[idade, pressao_sistolica, colesterol, batimentos_em_repouso, fumante_bin,
                    diabetico_bin, historico_familiar_bin, sexo_Feminino, sexo_Masculino, atividade_fisica_Alta,
                    atividade_fisica_Baixa, atividade_fisica_Moderada, estresse_Alto, estresse_Baixo, estresse_Moderado]])

    colunas = ['idade', 'pressao_sistolica', 'colesterol_total', 'batimentos_em_repouso', 'fumante',
            'diabetico', 'historico_familiar', 'sexo_Feminino', 'sexo_Masculino',
            'atividade_fisica_Alta', 'atividade_fisica_Baixa', 'atividade_fisica_Moderada',
            'estresse_Alto', 'estresse_Baixo', 'estresse_Moderado']

    df = pd.DataFrame(dados, columns=colunas)
    
    colunas_numericas = ['idade', 'pressao_sistolica', 'colesterol_total', 'batimentos_em_repouso']
    
    scaler = joblib.load('scaler.joblib')
    
    df[colunas_numericas] = scaler.transform(df[colunas_numericas])
    
    modelo = joblib.load('modelo_random_forest.joblib')
    previsao = modelo.predict(df)
    previsao_prob = modelo.predict_proba(df)

    st.subheader("Resultado:")
    st.write(f"Previsão: {previsao[0]}")
    st.write(f"Probabilidade de ter: {previsao_prob[0, 1]:.0%}")
    st.write(f"Probabilidade de não ter: {previsao_prob[0, 0]:.0%}")
    
    if previsao[0] == 1:
        st.error("⚠️ Atenção: Há risco de doença cardíaca!")
    else:
        st.success("✅ Sem indícios de doença cardíaca.")