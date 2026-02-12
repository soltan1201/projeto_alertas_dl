# Exemplo para converter onde a predição é maior que 0.5 (threshold)
# Primeiro criamos uma máscara binária, depois vetorizamos
gdal_calc.py -A Mosaico_Outubro.tif --outfile=mask.tif --calc="A>0.5"
gdal_polygonize.py mask.tif -f "GPKG" Alertas_Outubro_Vetor.gpkg