#!/bin/bash

# ==============================================================================
# Script: vetorizar_para_gee.sh
# Função: Gera Shapefiles e um ZIP pronto para upload no Google Earth Engine
# Uso: ./vetorizar_para_gee.sh NOME_DO_RASTER
# ==============================================================================

# 1. Configurações
RASTER_NAME=$1

# Verifica argumento
if [ -z "$RASTER_NAME" ]; then
    echo "❌ Erro: Informe o nome do raster. Ex: ./vetorizar_para_gee.sh Mosaico_Outubro"
    exit 1
fi

# Caminhos
BASE_DIR="/home/superuser/db_images"
INPUT_TIF="$BASE_DIR/rasters_alerts/$RASTER_NAME.tif"
OUTPUT_DIR="$BASE_DIR/vetor_alerts"
OUTPUT_SHP="$OUTPUT_DIR/$RASTER_NAME.shp"
OUTPUT_ZIP="$OUTPUT_DIR/$RASTER_NAME.zip"
TEMP_MASK="mask_${RASTER_NAME}_temp.tif"

# Cria o diretório de saída se não existir
mkdir -p "$OUTPUT_DIR"

# Verifica se a imagem existe
if [ ! -f "$INPUT_TIF" ]; then
    echo "❌ Erro: Arquivo de entrada não encontrado: $INPUT_TIF"
    exit 1
fi

echo "🚀 Processando: $RASTER_NAME"

# 2. Criar Máscara Binária (Threshold > 0.5)
echo "🔹 1. Gerando máscara binária..."
gdal_calc.py -A "$INPUT_TIF" \
             --outfile="$TEMP_MASK" \
             --calc="A>0.5" \
             --NoDataValue=0 \
             --type='Byte' \
             --quiet

# 3. Vetorizar para Shapefile (Formato aceito pelo GEE)
echo "🔹 2. Vetorizando para Shapefile..."
# -f "ESRI Shapefile" garante compatibilidade
gdal_polygonize.py "$TEMP_MASK" \
                   -f "ESRI Shapefile" \
                   "$OUTPUT_SHP" \
                   "$RASTER_NAME" \
                   -8 \
                   -q

# 4. Compactar para Upload no GEE
# O GEE aceita melhor se você enviar um .zip contendo .shp, .shx, .dbf, .prj
if [ -f "$OUTPUT_SHP" ]; then
    echo "📦 3. Criando ZIP para o GEE..."
    
    # Entra na pasta para zipar sem caminhos absolutos (o GEE prefere assim)
    cd "$OUTPUT_DIR"
    
    # Zipa os 4 arquivos essenciais do shapefile
    zip -j "$RASTER_NAME.zip" "$RASTER_NAME.shp" "$RASTER_NAME.shx" "$RASTER_NAME.dbf" "$RASTER_NAME.prj"
    
    # Volta para o diretório original (opcional, boa prática)
    cd - > /dev/null
    
    echo "✅ ZIP gerado: $OUTPUT_ZIP"
else
    echo "❌ Falha na vetorização."
    exit 1
fi

# 5. Limpeza
rm "$TEMP_MASK" # Remove a máscara tif temporária
# Opcional: Se quiser economizar espaço e manter só o ZIP, descomente a linha abaixo:
# rm "$OUTPUT_DIR/$RASTER_NAME".{shp,shx,dbf,prj,cpg} 

echo "✅ Processo concluído! Agora é só subir o .zip no GEE."