import os
# import sys
import glob
import subprocess
# import shutil
import argparse
from datetime import datetime

# --- CONFIGURAÇÕES DE CAMINHOS ---
# --- CONFIGURAÇÕES DE CAMINHOS ---
BASE_DIR = "/home/superusuario/db_images/predAlerts"

# Pastas de saída (Destino)
OUTPUT_RASTER_BASE = os.path.join(BASE_DIR, "rasters_alerts")
OUTPUT_VETOR_BASE = os.path.join(BASE_DIR, "vetor_alerts")

# Sua lista de GRIDs Landsat/Sentinel
LST_ID_GRID = [
    '214/67', '215/67', '216/67', '217/67', '218/67', '219/67', '220/67', '221/67', '215/69', 
    '216/69', '217/69', '218/69', '219/69', '220/69', '215/70', '216/70', '217/70', '218/70',
    '219/70', '220/70', '217/62', '218/62', '219/62', '215/63', '216/63', '217/63', '218/63', 
    '219/63', '220/63', '214/64', '215/64', '216/64', '217/64', '218/64', '219/64', '214/65', 
    '215/65', '216/65', '217/65', '218/65', '219/65', '220/65', '221/65', '214/66', '215/66', 
    '216/66', '217/66', '218/66', '219/66', '220/66', '221/66', '215/68', '216/68', '217/68', 
    '218/68', '219/68', '220/68', '221/68', '215/71', '216/71', '217/71', '218/71', '219/71', 
    '220/71', '221/71', '216/72', '217/72', '218/72', '219/72', '220/72', '217/73', '218/73'
]

def run_command(command, description):
    """Executa comando shell e trata erros"""
    try:
        print(f"  ⚙️ {description}...")
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Erro em {description}: {e.stderr}")
        return False

def processar_grid_para_shp(pasta_mes, grid_id):
    # Formata o nome (ex: 214/67 vira 214_67)
    grid_clean = grid_id.replace("/", "_")
    
    # Caminhos específicos
    # Define o caminho de entrada específico (BASE_DIR/nome_da_pasta_do_mes)
    input_path = os.path.join(BASE_DIR, pasta_mes)
    # 1. Filtro: Busca arquivos que tenham o ID da GRID no nome dentro da pasta do mês
    search_pattern = os.path.join(input_path, f"*{grid_clean}*.tif")
    tif_files = glob.glob(search_pattern)

    if len(tif_files) == 0:
        return # Pula se não encontrar nada para essa grid neste mês

    # 2. Define e cria pastas de saída espelhadas (Ex: rasters_alerts/PATCHS_S2_Novembro_CAAT/)
    # out_raster_dir = os.path.join(OUTPUT_RASTER_BASE, pasta_mes)
    out_vetor_dir = os.path.join(OUTPUT_VETOR_BASE, pasta_mes)
    os.makedirs(OUTPUT_RASTER_BASE, exist_ok=True)
    os.makedirs(out_vetor_dir, exist_ok=True)
    
    # Nomes dos arquivos de saída
    # Ex: tfrecord_219_71_PATCHS_S2_Novembro_CAAT.tif
    nome_base = tif_files[0].split("/")[-1][:23]
    temp_tif = os.path.join(OUTPUT_RASTER_BASE, f"{nome_base}.tif")
    mask_tif = temp_tif.replace(".tif", "_mask.tif")
    # vetor_zip = os.path.join(out_vetor_dir, f"vetor_{nome_base}.zip")
    shp_path = os.path.join(out_vetor_dir, f"vetor_{nome_base}.shp")
    print(f"📦 [{pasta_mes}] Grid {grid_id} -> Usando prefixo: {nome_base} com ({len(tif_files)} arquivos)")

    try:
        # --- SOLUÇÃO PARA "Argument list too long" ---
        # Criamos um arquivo de texto temporário com a lista de arquivos
        list_file_path = os.path.join(OUTPUT_RASTER_BASE, f"list_{nome_base}.txt")
        with open(list_file_path, 'w') as f:
            for tif in tif_files:
                f.write(f"{tif}\n")

        # 1. Mosaico usando a flag --optfile (ou passando a lista via arquivo)
        # O gdal_merge aceita ler os inputs de um arquivo usando a flag --optfile
        subprocess.run([
            "gdal_merge.py", "-ot", "Float32", "-n", "0", "-a_nodata", "0",
            "-o", temp_tif, "--config", "GDAL_CACHEMEM", "2000",
            "--optfile", list_file_path  # <--- MÁGICA AQUI
        ], check=True)

        # Após o merge, podemos apagar a lista
        os.remove(list_file_path)

        # 2. Máscara Binária (gdal_calc) (Threshold 0.5)
        subprocess.run([
            "gdal_calc.py", "-A", temp_tif, "--outfile", mask_tif,
            "--calc", "A>0.5", "--NoDataValue", "0", "--type", "Byte", "--quiet"
        ], check=True)

        # 3. Vetorização (gdal_polygonize)
        subprocess.run([
            "gdal_polygonize.py", mask_tif, "-f", "ESRI Shapefile",
            shp_path, nome_base, "-8", "-q"
        ], check=True)

        # # 4. Compactação ZIP para o GEE
        # # Coleta todos os arquivos gerados pelo shapefile (.shp, .shx, .dbf, .prj)
        # files_to_zip = glob.glob(os.path.join(out_vetor_dir, f"vetor_{grid_clean}.*"))
        # # Remove o zip da lista se ele já existir para não zipar o próprio zip
        # files_to_zip = [f for f in files_to_zip if not f.endswith('.zip')]
        
        # subprocess.run(["zip", "-j", vetor_zip] + files_to_zip, check=True, capture_output=True)
        
        # 4. LIMPEZA (Manter o SSD de 2TB limpo)
        # Remove TIFs temporários (Libera espaço imediatamente)
        for f in [temp_tif, mask_tif]:
            if os.path.exists(f):
                os.remove(f)
        
        print(f"   ✅ Sucesso: {shp_path}")
        return shp_path

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erro na Grid {grid_id}: {e}")
        return None

def merge_e_zip_mensal(pasta_mes):
    out_vetor_dir = os.path.join(OUTPUT_VETOR_BASE, pasta_mes)
    # BUSCA CORRIGIDA: Pega todos os .shp, mas ignora o que começa com VETOR_FINAL
    shps_individuais = [
        f for f in glob.glob(os.path.join(out_vetor_dir, "*.shp")) 
        if "VETOR_FINAL" not in os.path.basename(f)
    ]
    
    if not shps_individuais:
        return

    nome_final_mes = f"VETOR_FINAL_{pasta_mes}"
    output_merged_shp = os.path.join(out_vetor_dir, f"{nome_final_mes}.shp")
    zip_final = os.path.join(out_vetor_dir, f"{nome_final_mes}.zip")

    print(f"\n🚜 Realizando MERGE de {len(shps_individuais)} vetores em um único arquivo...")

    try:
        # Usamos ogrmerge.py para juntar todos os Shapefiles
        # -single: junta tudo em uma única camada
        # -o: arquivo de saída
        merge_cmd = ["ogrmerge.py", "-single", "-o", output_merged_shp] + shps_individuais
        subprocess.run(merge_cmd, check=True)

        # Compactar o resultado final
        print(f"📦 Criando ZIP final do mês: {nome_final_mes}.zip")
        # Pega todos os arquivos do merge (shp, shx, dbf, prj)
        files_to_zip = glob.glob(os.path.join(out_vetor_dir, f"{nome_final_mes}.*"))
        subprocess.run(["zip", "-j", zip_final] + files_to_zip, check=True)

        # LIMPEZA FINAL: Remove todos os SHPs individuais e o SHP do merge (deixa só o ZIP)
        print("🗑️ Limpando arquivos residuais...")
        todos_arquivos = glob.glob(os.path.join(out_vetor_dir, "*.*"))
        for f in todos_arquivos:
            if not f.endswith(".zip"):
                os.remove(f)
                
    except Exception as e:
        print(f"❌ Erro no Merge/Zip: {e}")

if __name__ == "__main__":
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('month_folder', type=str,  default= True, help= "Especifica o nome da pasta dentro do db_images/predAlerts que vai ser processada" )
    args = parser.parse_args()
    month_folder= args.month_folder

    print(f"Iniciando processamento em massa: {start_time}")
    print(f"\n--- 📁 Iniciando Mês: {month_folder} ---")

    # 1. Primeiro processa TODAS as grids para gerar os .shp individuais
    for grid in LST_ID_GRID:
        processar_grid_para_shp(month_folder, grid)
        
    # 2. SÓ DEPOIS que todas as grids acabarem, faz o Merge e o Zip uma única vez
    merge_e_zip_mensal(month_folder)
    
    end_time = datetime.now()
    print(f"\n✨ Tudo pronto! Tempo total: {end_time - start_time}")