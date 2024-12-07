#!/bin/bash
# Atualizar pacotes
sudo yum update -y

# Instalar pip3 e dependências
sudo yum install python3-pip -y
pip3 install --upgrade pip

# Instalar as bibliotecas necessárias
pip3 install boto3 \
             pandas==1.3.4 \
             numpy==1.24.3 \
             scikit-learn==1.5.2 \
             lightgbm==4.5.0 \
             FLAML==2.3.2 \
             xgboost==1.6.2

# Criar o diretório temporário
mkdir -p /home/ec2-user/script
cd /home/ec2-user/script

# Criar o arquivo Python
cat << EOF > churn_script.py
import boto3
import joblib
import pandas as pd
import numpy as np
from io import StringIO

# Configurações do S3
bucket_name = 'model-script-churn'
file_key = 'churn.pipeline.pkl'  # Caminho do arquivo no bucket
local_file_path = '/tmp/churn.pipeline.pkl'  # Caminho local temporário

# Baixar o arquivo do S3
s3 = boto3.client('s3')
s3.download_file(bucket_name, file_key, local_file_path)

# Carregar o modelo com joblib
model = joblib.load(local_file_path)

print("Modelo carregado com sucesso!")

# Configurações do S3
bucket_name = 'input-test-w'
file_key = 'instances.csv'  # Caminho completo para o arquivo no bucket

# Criar um cliente S3
s3 = boto3.client('s3')

# Baixar o arquivo como objeto
csv_obj = s3.get_object(Bucket=bucket_name, Key=file_key)

# Ler o conteúdo do arquivo em uma string
body = csv_obj['Body'].read().decode('utf-8')

# Carregar o CSV em um DataFrame do pandas
df = pd.read_csv(StringIO(body))

df = df.astype(
    {
        "customerID": "str",
        "gender": "str",
        "SeniorCitizen": "str",
        "Partner": "str",
        "Dependents": "str",
        "tenure": "str",
        "PhoneService": "str",
        "MultipleLines": "str",
        "InternetService": "str",
        "OnlineSecurity": "str",
        "OnlineBackup": "str",
        "DeviceProtection": "str",
        "TechSupport": "str",
        "StreamingTV": "str",
        "StreamingMovies": "str",
        "Contract": "str",
        "PaperlessBilling": "str",
        "PaymentMethod": "str",
        "MonthlyCharges": "str",
        "TotalCharges": "str",
        "Churn": "str",
    }
)


X = df.drop(columns=['Churn'], axis=1)

# Aplicar o modelo para prever a classe
df['Prediction'] = model.predict(X)

# Aplicar o modelo para prever a probabilidade
df['Probability'] = model.predict_proba(X)[:, 1]

# Converter o DataFrame para CSV
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)

# Fazer upload para o S3
bucket_name = 'output-test-w'
output_key = 'output.csv'

s3.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())

print(f"Resultados salvos no bucket '{bucket_name}' com a chave '{output_key}'")
EOF


# Criar o arquivo transform.py
cat << 'EOF' > transformers.py
from sklearn.base import BaseEstimator, TransformerMixin

class CustomerIdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_columns):
        self.id_columns = id_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.id_columns, axis=1)
EOF

# Executar o script Python
python3 churn_script.py
