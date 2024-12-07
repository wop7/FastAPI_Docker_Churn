#!/bin/bash

# Habilitar a execução do User Data em todos os boots
cat << EOF | sudo tee /etc/cloud/cloud.cfg.d/99_force_userdata.cfg
#cloud-config
cloud_final_modules:
 - [scripts-user, always]
EOF

# Atualizar pacotes do sistema
sudo yum update -y

# Instalar pip3 apenas se não estiver instalado
if ! command -v pip3 &> /dev/null; then
  sudo yum install python3-pip -y
  pip3 install --upgrade pip
fi

# Instalar as bibliotecas necessárias apenas se não estiverem instaladas
REQUIRED_PACKAGES=(boto3 pandas numpy scikit-learn==1.5.2 lightgbm==4.5.0 FLAML==2.3.2 xgboost==1.6.2)
for PACKAGE in "${REQUIRED_PACKAGES[@]}"; do
  if ! pip3 show $(echo "$PACKAGE" | cut -d= -f1) &> /dev/null; then
    pip3 install "$PACKAGE"
  fi
done

# Verificar se o diretório do script existe
SCRIPT_DIR="/home/ec2-user/script"
if [ ! -d "$SCRIPT_DIR" ]; then
  mkdir -p "$SCRIPT_DIR"
fi

# Verificar se o churn_script.py já existe
CHURN_SCRIPT="$SCRIPT_DIR/churn_script.py"
if [ ! -f "$CHURN_SCRIPT" ]; then
  cat << 'EOF' > "$CHURN_SCRIPT"
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
fi

# Verificar se o transformers.py já existe
TRANSFORMERS_SCRIPT="$SCRIPT_DIR/transformers.py"
if [ ! -f "$TRANSFORMERS_SCRIPT" ]; then
  cat << 'EOF' > "$TRANSFORMERS_SCRIPT"
from sklearn.base import BaseEstimator, TransformerMixin

class CustomerIdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_columns):
        self.id_columns = id_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.id_columns, axis=1)
EOF
fi

# Configurar permissões para os arquivos
chmod +x "$CHURN_SCRIPT"

# Executar o script Python
python3 "$CHURN_SCRIPT"
