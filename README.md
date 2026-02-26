# projeto_alertas_dl
este é umn projeto para automatizar a revisão de alertas


paths 
PATH_TFRECORDS = "/home/superuser/db_images"
PATH_OUTPUT = "/home/superuser/db_images/predicts"
/run/media/superuser/Almacen/imgDB/predAlerts/
LISTA_FOLDERS = [
    'PATCHS_S2_Setembro_CAAT', 'PATCHS_S2_Outubro_CAAT',  
    'PATCHS_S2_Novembro_CAAT', 'PATCHS_S2_Dezembro_CAAT'
]

para predict usar em local 
python predict_all_in_One.py /run/media/superuser/Almacen/imgDB  /run/media/superuser/Almacen/imgDB/predAlerts False
no servidor 
python predict_all_in_One.py /home/superusuario/db_images  /home/superusuario/db_images/predAlerts True


para juntar todos os .tif e salvar em um único arquivo 
chmod +x stitch_mosaics.sh
./stitch_mosaics.sh PATCHS_S2_Dezembro_CAAT


Para vetorreizar o raster TIF precisa rodar esses dois comandos 
chmod +x vetorreizar_gdal.sh
./vetorreizar_gdal.sh PATCHS_S2_Dezembro_CAAT


LISTA_FOLDERS = [
    'PATCHS_S2_Setembro_CAAT', 
    'PATCHS_S2_Outubro_CAAT',  
    'PATCHS_S2_Novembro_CAAT', 
    'PATCHS_S2_Dezembro_CAAT'
]
Para rodar a vetorização na pasta dos patchs que passaram por predict 
python stitch_vetorixar_gdal.py PATCHS_S2_Setembro_CAAT