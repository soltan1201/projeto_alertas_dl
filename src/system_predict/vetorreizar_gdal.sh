#!/bin/bash

# Caminho do mosaico de entrada
INPUT_TIF="/run/media/superuser/Almacen/imgDB/ALERTS_S2_Outubro_CAAT.tif"
# Nome do Shapefile de saída (sem a extensão .shp)
OUTPUT_SHP="/run/media/superuser/Almacen/imgDB/Alertas_Outubro_Caatinga"

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