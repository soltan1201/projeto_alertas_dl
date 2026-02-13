#!/bin/bash
# Captura o primeiro argumento passado na chamada do script
RASTER_NAME=$1

# Verifica se o usuário passou o argumento, se não, encerra com erro
if [ -z "$RASTER_NAME" ]; then
    echo "Erro: Você precisa passar o nome da pasta. Ex: ./vetorreizar_gdal.sh NOME_DO_RASTER"
    exit 1
fi

# Caminho do mosaico de entrada
INPUT_TIF="/home/superuser/db_images/predAlerts/$RASTER_NAME.tif"
# Nome do Shapefile de saída (sem a extensão .shp)
OUTPUT_SHP="/home/superuser/db_images/vetor_alerts/$RASTER_NAME"

echo "1. Criando máscara binária (Threshold > 0.5)..."
gdal_calc.py -A "$INPUT_TIF" \
             --outfile=mask_temp.tif \
             --calc="A>0.5" \
             --NoDataValue=0 \
             --type='Byte' \
             --quiet

echo "2. Vetorizando para Shapefile..."
# -8: Conecta pixels na diagonal (8-connectivity)
# -f "ESRI Shapefile": Define o formato de saída
gdal_polygonize.py mask_temp.tif -f "ESRI Shapefile" "$OUTPUT_SHP.shp"

# Limpeza opcional
rm mask_temp.tif

echo "✅ Sucesso! Arquivo $OUTPUT_SHP.shp gerado."