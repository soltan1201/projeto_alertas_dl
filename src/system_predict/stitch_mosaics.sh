#!/bin/bash


# Captura o primeiro argumento passado na chamada do script
FOLDER_NAME=$1

# Verifica se o usuário passou o argumento, se não, encerra com erro
if [ -z "$FOLDER_NAME" ]; then
    echo "Erro: Você precisa passar o nome da pasta. Ex: ./stitch_moisaics.sh NOME_DA_PASTA"
    exit 1
fi
# 1. Ajuste os caminhos (Sem a barra final para evitar confusão)
INPUT_FOLDER="/home/superusuario/db_images/predAlerts/$FOLDER_NAME"
FINAL_OUTPUT_DIR="/home/superusuario/db_images/rasters_alerts/$FOLDER_NAME"
OUTPUT_FILE="$FINAL_OUTPUT_DIR/$FOLDER_NAME.tif"

# 2. Criar a pasta de saída
mkdir -p "$FINAL_OUTPUT_DIR"

echo "🔍 Verificando arquivos em: $INPUT_FOLDER"

# 3. Contar quantos arquivos existem para evitar erro de 'lista vazia'
count=$(ls -1 "$INPUT_FOLDER"/*.tif 2>/dev/null | wc -l)

if [ "$count" -gt 0 ]; then
    echo "🚀 Iniciando Mosaico de $count arquivos..."
    
    # -ot Float32: Mantém a precisão do modelo
    # -n 0: Ignora zeros (áreas pretas dos patches)
    # -a_nodata 0: Define 0 como transparente no final
    # --config GDAL_CACHEMEM 2000: Usa 2GB de RAM para acelerar o processo no Arch
    
    gdal_merge.py -ot Float32 -n 0 -a_nodata 0 \
        -o "$OUTPUT_FILE" \
        "$INPUT_FOLDER"/*.tif
        
    echo "✅ Sucesso! Mosaico gerado em: $OUTPUT_FILE"
else
    echo "❌ ERRO: Nenhum arquivo .tif encontrado em $INPUT_FOLDER"
fi