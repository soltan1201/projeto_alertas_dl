import os
import glob
import subprocess
import argparse
from datetime import datetime

# --- CONFIGURAÇÕES DE CAMINHOS ---
BASE_DIR = "/home/superusuario/db_images/predAlerts"
OUTPUT_RASTER_BASE = os.path.join(BASE_DIR, "rasters_alerts")
OUTPUT_VETOR_BASE = os.path.join(BASE_DIR, "vetor_alerts")

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

def processar_grid_para_shp(pasta_mes, grid_id):
    grid_clean = grid_id.replace("/", "_")
    input_path = os.path.join(BASE_DIR, pasta_mes)
    search_pattern = os.path.join(input_path, f"*{grid_clean}*.tif")
    tif_files = glob.glob(search_pattern)

    if len(tif_files) == 0:
        return 

    out_vetor_dir = os.path.join(OUTPUT_VETOR_BASE, pasta_mes)
    os.makedirs(OUTPUT_RASTER_BASE, exist_ok=True)
    os.makedirs(out_vetor_dir, exist_ok=True)
    
    nome_base = tif_files[0].split("/")[-1][:23]
    temp_tif = os.path.join(OUTPUT_RASTER_BASE, f"{nome_base}.tif")
    mask_tif = temp_tif.replace(".tif", "_mask.tif")
    shp_path = os.path.join(out_vetor_dir, f"vetor_{nome_base}.shp")
    
    print(f"📦 [{pasta_mes}] Grid {grid_id} -> Prefixo: {nome_base} ({len(tif_files)} arquivos)")

    try:
        # Lista de arquivos para evitar "Argument list too long"
        list_file_path = os.path.join(OUTPUT_RASTER_BASE, f"list_{nome_base}.txt")
        with open(list_file_path, 'w') as f:
            for tif in tif_files:
                f.write(f"{tif}\n")

        # 1. Mosaico
        subprocess.run(["gdal_merge.py", "-ot", "Float32", "-n", "0", "-a_nodata", "0",
                        "-o", temp_tif, "--config", "GDAL_CACHEMEM", "2000",
                        "--optfile", list_file_path], check=True)
        os.remove(list_file_path)

        # 2. Máscara (Threshold 0.5)
        subprocess.run(["gdal_calc.py", "-A", temp_tif, "--outfile", mask_tif,
                        "--calc", "A>0.5", "--NoDataValue", "0", "--type", "Byte", "--quiet", "--overwrite"], check=True)

        # 3. Vetorização
        subprocess.run(["gdal_polygonize.py", mask_tif, "-f", "ESRI Shapefile",
                        shp_path, nome_base, "-8", "-q"], check=True)

        # 4. Limpeza RASTER
        for f in [temp_tif, mask_tif]:
            if os.path.exists(f): os.remove(f)
        
        print(f"   ✅ Sucesso: {shp_path}")
        return shp_path
    except Exception as e:
        print(f"   ❌ Erro na Grid {grid_id}: {e}")
        return None

def merge_e_zip_mensal(pasta_mes):
    out_vetor_dir = os.path.join(OUTPUT_VETOR_BASE, pasta_mes)
    shps_individuais = [f for f in glob.glob(os.path.join(out_vetor_dir, "*.shp")) 
                        if "VETOR_FINAL" not in os.path.basename(f)]
    
    if not shps_individuais:
        print(f"⚠️ Ninguém para mesclar em {pasta_mes}")
        return

    nome_final_mes = f"VETOR_FINAL_{pasta_mes}"
    output_merged_shp = os.path.join(out_vetor_dir, f"{nome_final_mes}.shp")
    zip_final = os.path.join(out_vetor_dir, f"{nome_final_mes}.zip")

    print(f"\n🚜 Mesclando {len(shps_individuais)} vetores...")

    try:
        # CORREÇÃO: Ordem dos argumentos do ogrmerge
        merge_cmd = ["ogrmerge.py", "-single", "-overwrite_ds", "-o", output_merged_shp] + shps_individuais
        subprocess.run(merge_cmd, check=True)

        print(f"📦 Criando ZIP final: {nome_final_mes}.zip")
        files_to_zip = glob.glob(os.path.join(out_vetor_dir, f"{nome_final_mes}.*"))
        subprocess.run(["zip", "-j", zip_final] + files_to_zip, check=True)

        print("🗑️ Limpando arquivos residuais...")
        for f in glob.glob(os.path.join(out_vetor_dir, "*.*")):
            if not f.endswith(".zip"): os.remove(f)
                
    except Exception as e:
        print(f"❌ Erro no Merge/Zip: {e}")

# --- BLOCO PRINCIPAL CORRIGIDO ---
if __name__ == "__main__":
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description="Pipeline Geodatin: Mosaico e Vetorização")
    # Argumento posicional para a pasta
    parser.add_argument('month_folder', type=str, 
                        help="Nome da pasta (ex: PATCHS_S2_Setembro_CAAT)")
    # Argumento opcional usando action='store_true' (O jeito certo para booleanos)
    parser.add_argument('--vetorizar', action='store_true', 
                        help="Se presente, executa a vetorização das grids. Se omitido, apenas tenta o Merge.")
    
    args = parser.parse_args()
    
    print(f"🚀 Iniciando processamento: {start_time}")
    print(f"📂 Pasta alvo: {args.month_folder}")
    print(f"🛠️ Modo Vetorização: {'ATIVADO' if args.vetorizar else 'APENAS MERGE'}")

    # 1. Processamento das Grids (Só roda se você passar --vetorizar)
    if args.vetorizar:
        for grid in LST_ID_GRID:
            processar_grid_para_shp(args.month_folder, grid)
        
    # 2. Consolidação Mensal
    merge_e_zip_mensal(args.month_folder)
    
    print(f"\n✨ Concluído! Tempo total: {datetime.now() - start_time}")