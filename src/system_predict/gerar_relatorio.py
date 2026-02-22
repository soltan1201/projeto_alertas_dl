import os
import json
from datetime import datetime

# Configurações
# Como criamos o link simbólico, podemos usar o caminho da home
BASE_DIR = "/home/superusuario/db_images"
FOLDERS = [
    "PATCHS_S2_Dezembro_Caat", 
    "PATCHS_S2_Dezembro_CAAT", 
    "PATCHS_S2_Novembro_CAAT", 
    "PATCHS_S2_Setembro_CAAT", 
    "PATCHS_S2_Outubro_CAAT",
    "rasters_alerts"
]

def gerar_relatorio():
    data_dia = datetime.now().strftime("%Y-%m-%d")
    nome_arquivo = f"relatorio_db_tfrecords_{data_dia}.json"
    
    dict_reports = {}

    print(f"--- Iniciando Relatório de Arquivos Locais ({data_dia}) ---")
    
    for folder in FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        
        if os.path.exists(folder_path):
            # Lista apenas arquivos que terminam com .tfrecord.gz ou .tfrecord
            files = [f for f in os.listdir(folder_path) if ".tfrecord" in f]
            dict_reports[folder] = files
            print(f"✅ {folder}: {len(files)} arquivos encontrados.")
        else:
            dict_reports[folder] = []
            print(f"⚠️ {folder}: Pasta não encontrada no servidor.")

    # Salva o JSON
    with open(os.path.join(BASE_DIR, nome_arquivo), 'w', encoding='utf-8') as f:
        json.dump(dict_reports, f, indent=4, ensure_ascii=False)
    
    print(f"\n🚀 Relatório salvo com sucesso: {nome_arquivo}")

if __name__ == "__main__":
    gerar_relatorio()