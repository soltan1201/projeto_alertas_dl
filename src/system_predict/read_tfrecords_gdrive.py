import os
import io
import sys
import argparse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# 1. Configurações de Caminho
parser = argparse.ArgumentParser()
parser.add_argument('key_conta', type=str,  default= True, help= "Especifica qual key Cloud conta será usada:" )
args = parser.parse_args()
unicaConta= str(args.key_conta)
# 1. Configurações de Caminho

SERVICE_ACCOUNT_FILE = '/home/superusuario/.config/gcloud/keys/mapbiomas-caatinga-cloud04-78950c04489a.json'
if unicaConta == '4':
    SERVICE_ACCOUNT_FILE = '/home/superuser/.config/gcloud/keys/mapbiomas-caatinga-cloud04-78950c04489a.json'
else: 
    SERVICE_ACCOUNT_FILE = '/home/superusuario/.config/gcloud/keys/ee-solkancengine17-ef2f5f6fe840.json'
LOCAL_BASE_DIR = '/home/superusuario/db_images'
SCOPES = ['https://www.googleapis.com/auth/drive']

# Pastas que queremos copiar
DRIVE_FOLDERS = [
    'PATCHS_S2_Dezembro_CAAT',
    'PATCHS_S2_Novembro_CAAT',
    'PATCHS_S2_Outubro_CAAT',
    'PATCHS_S2_Setembro_CAAT'   
]

# 2. Autenticação
try:
    creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    # TESTE DE CONEXÃO: Obtém informações sobre o usuário autenticado
    # Para Contas de Serviço, o 'user' é a própria conta
    about = service.about().get(fields="user").execute()
    email_conectado = about['user']['emailAddress']
    
    print(f"✅ Conectado com sucesso!")
    print(f"📧 Conta ativa: {email_conectado}")
    
except Exception as e:
    print(f"❌ Falha na conexão: {e}")
    sys.exit(1) # Para a execução se não conectar

def download_files_from_folder(folder_name):
    print(f"\n--- Iniciando busca na pasta: {folder_name} ---")
    
    # Busca o ID da pasta pelo nome
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get('files', [])

    if not folders:
        print(f"Aviso: Pasta {folder_name} não encontrada no Drive.")
        return

    folder_id = folders[0]['id']
    
    # Cria a pasta local correspondente
    dest_path = os.path.join(LOCAL_BASE_DIR, folder_name)
    os.makedirs(dest_path, exist_ok=True)

    # Lista todos os arquivos dentro da pasta do Drive
    file_results = service.files().list(
        q=f"'{folder_id}' in parents",
        fields="files(id, name)").execute()
    files = file_results.get('files', [])

    for file in files:
        file_id = file['id']
        file_name = file['name']
        file_path = os.path.join(dest_path, file_name)

        print(f"Baixando {file_name}...")

        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            
    print(f"Pasta {folder_name} concluída!")

# Execução principal
if __name__ == '__main__':
    if not os.path.exists(LOCAL_BASE_DIR):
        os.makedirs(LOCAL_BASE_DIR)
        
    for folder in DRIVE_FOLDERS:
        download_files_from_folder(folder)
        
    print("\n✅ Todos os patches de TFRecords foram baixados com sucesso.")