#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Produzido por Geodatin - Dados e Geoinformacao
DISTRIBUIDO COM GPLv2
@author: geodatin
"""
import ee
import os
import sys
import collections
collections.Callable = collections.abc.Callable

# from pathlib import Path
# pathparent = str(Path(os.getcwd()).parents[0])
pathparent = str('/home/superuser/Dados/projAlertas/proj_alertas_ML/src')
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account, get_project_from_account
projAccount = get_current_account()
print(f"projetos selecionado >>> {projAccount} <<<")

try:
    ee.Initialize( project= projAccount )
    print(' 🕸️ 🌵 The Earth Engine package initialized successfully!')
except ee.EEException as e:
    print('The Earth Engine package failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise
date_inic = '2025-09-01'
name_month = 'setembro'
name_folder = 'Setembro2025'

# ==============================================================================
# 1. DEFINIÇÃO DE CONSTANTES E PARÂMETROS
# ==============================================================================
PATCH_SIZE = 256
SCALE = 10  # Resolução do Sentinel-2 (10m)
PATCH_SIDE_METERS = PATCH_SIZE * SCALE # 2560 metros
STRIDE = 128 # 50% de sobreposição (overlap) para garantir que pegamos bordas
PATCH_METERS = PATCH_SIZE * SCALE  # 2560 metros
STRIDE_METERS = STRIDE * SCALE     # Stride em metros (1280m)
EXPORT_FOLDER = 'DATASET_CHANGE_DETECTION_S2'

asset_fcAlert = f'projects/mapbiomas-caatinga-cloud04/assets/Alertas/revisados_2025/alerts_pol_{name_month}_{date_inic}'
asset_folderAlert = f'projects/mapbiomas-caatinga-cloud04/assets/Alertas/gerados_2025/{name_folder}'
asset_output = 'projects/mapbiomas-caatinga-cloud04/assets/Alertas/featCol_samples/' + name_month
asset_grade_landsat = 'users/CartasSol/shapes/grade_landsat_tm_Br'
# 1. Carregar alertas aprovados
featC_alerts = (ee.FeatureCollection(asset_fcAlert) 
                        .map(lambda feat: feat.set('aprovado', True, 'idCod', 1))
            )

print("Feature Collection alerts (limit 1):", featC_alerts.limit(1).getInfo())
size_alerts = featC_alerts.size().getInfo()
print("Feature Collection alerts tamanho:", size_alerts)



def add_bbox_dimensions(feature):
    """
    Calcula largura e altura da Bounding Box em metros.
    Adiciona as propriedades 'width_m' e 'height_m' ao feature.
    """
    # Pega o bounds projetado em metros (EPSG:3857 é seguro para medir distâncias lineares curtas)
    # O erro de projeção é desprezível para patches pequenos.
    bounds = feature.geometry().bounds(1, 'EPSG:3857')
    coords = ee.List(ee.Geometry(bounds).coordinates()).get(0)
    
    # Coordenadas do retângulo: 
    # 0: [minx, miny], 1: [maxx, miny], 2: [maxx, maxy] ...
    p0 = ee.Geometry.Point(ee.List(coords).get(0), 'EPSG:3857')
    p1 = ee.Geometry.Point(ee.List(coords).get(1), 'EPSG:3857') # Diferença no X
    p2 = ee.Geometry.Point(ee.List(coords).get(2), 'EPSG:3857') # Diferença no Y
    
    width = p0.distance(p1)
    height = p1.distance(p2)
    
    return feature.set('width_m', width, 'height_m', height)


# Função de Grid (Mesma lógica anterior, mas agora aplicada corretamente aos finos)
def create_grid_points_optimized(feature):
    """
    CORREÇÃO DE MEMÓRIA: Usa image.sample() ao invés de reduceToVectors.
    """
    geom = feature.geometry()
    proj = ee.Projection('EPSG:3857').atScale(SCALE)
    
    # Cria imagem de coordenadas
    coords = ee.Image.pixelCoordinates(proj)
    mask = coords.select('x').mod(STRIDE).eq(0).And(
            coords.select('y').mod(STRIDE).eq(0))
    
    # Clip pela geometria ORIGINAL (irregular)
    grid_mask = mask.clip(geom)
    
    # sample() retorna FeatureCollection de pontos diretamente dos pixels ativos
    points = grid_mask.updateMask(grid_mask).sample(
        region=geom,
        scale=SCALE,
        projection=proj,
        geometries=True,
        dropNulls=True
    )
    return points

def export_feat(featcol, nome_exp):
    task = ee.batch.Export.table.toAsset(
        collection=featcol,
        description=nome_exp,
        assetId= os.path.join(asset_output , nome_exp)
    )
    task.start()
    print(f"Exportação iniciada: {nome_exp}")

# Geometria para filtro (Centroides)
featPoints_alerts = ee.FeatureCollection(asset_fcAlert).map(lambda feat: feat.centroid())
geom_alerts = featPoints_alerts.geometry()

gradesCaatL8 = (ee.FeatureCollection(asset_grade_landsat)
                    .map(lambda feat: feat.set('id_cod', 1)))

# 1. Listar todos os assets no folder
asset_list = ee.data.listAssets(asset_folderAlert)  # , maxResults= 1000
asset_list = ee.Dictionary(asset_list).get('assets').getInfo()
print(" ============================================================ ")
number_grids = len(asset_list)
print("Número de featCollection em folders:", number_grids)

# Cálculo do número de coleta (usando getInfo() para cálculos locais se necessário)
# (size_alerts_true * 10) / number_grids
num_coleta = ee.Number(size_alerts).multiply(5).divide(number_grids)
print("Numero de alertas a serem coletados por grids:", num_coleta.getInfo())

# sys.exit()
# 2. Carregar todos os FeatureCollections do folder e combiná-los
counting = 0
numb_inic = 36
# Iterar sobre os assets encontrados
for cc, assetId in enumerate(asset_list[numb_inic:]):
    
    asset_id = assetId['id']
    name_export = asset_id.split('/')[-1]
    print(f">>>>>  processing #{cc + numb_inic} / {len(asset_list)} >>> {name_export}")
    # Extração dos índices X e Y do nome
    parts = name_export.split('_')
    # shp(0) alerta(1) 215(2) 64(3)
    try:
        wrs_path = parts[2] # "215" 
        wrs_row = parts[3] # "64"
    
    except IndexError:
        wrs_path = 'N/A'
        wrs_row = 'N/A'

    grade_tmp = gradesCaatL8.filter(
                    ee.Filter.And(
                        ee.Filter.eq('ORBITA', int(wrs_path)),
                        ee.Filter.eq('PONTO', int(wrs_row))
                    )
                ).geometry()
    # print(grade_tmp.size().getInfo())
    # sys.exit()
    alertGridTrue = featC_alerts.filter(ee.Filter.bounds(grade_tmp))
    numberAlertTrue = alertGridTrue.size().getInfo()
    # alertGridTrue = (alertGridTrue.map(lambda feat: feat.select([])
    #                                 .set('WRS_PATH',int(wrs_path), 'WRS_ROW', int(wrs_row))))

    alertGridTrue =alertGridTrue.select([]).toList(alertGridTrue.size())
    if numberAlertTrue> 0:
        # print('Coluna:', wrs_path)
        # print('Linha:', wrs_row)
        fc = ee.FeatureCollection(asset_id)
        # Processamento de área e amostragem
        fc_size = fc.size().getInfo()

        # Evitar divisão por zero se o FC estiver vazio
        sampling_ratio = num_coleta.divide(fc_size)
        
        fc = fc.map(lambda feat: feat.set('area',  feat.area()))
        
        fc_remain = (fc.randomColumn(rowKeys = ['area']) 
                            .filter(ee.Filter.lt('random', sampling_ratio)))
        
        # Log de progresso
        restantes = fc_remain.size().getInfo()
        print(f"Grid {wrs_path}/{wrs_row} - Restante: {restantes}")
        
        fc_remain = fc_remain.select([]).toList(fc_remain.size())
        fc_list = ee.List(fc_remain).cat(ee.List(alertGridTrue))
        fc_list = ee.FeatureCollection(fc_list)
        # fc_list = fc_remain.merge(alertGridTrue)
        numFeatend = fc_list.size().getInfo()
        print('size lista final  ', numFeatend)

        
        fc_box =  fc_list.map(add_bbox_dimensions)
        # print('size lista final modificada ', fc_box.size().getInfo())

        # grade_mosaic = full_stack.clip(grade_tmp)
        # grade_mosaic = ee.Image(grade_mosaic).toInt16()
        nome_export = f"sample_{wrs_path}_{wrs_row}_{name_month}"
        # get_training_patches(grade_tmp, ee.FeatureCollection(fc_box), nome_export, numFeatend)

        # ==========================================================================
        # LÓGICA DE SEPARAÇÃO: DIMENSÕES, NÃO ÁREA
        # ==========================================================================
        
        # Critério: Se QUALQUER lado for maior que o patch, ele é "Grande/Longo"
        # Usamos subtract(10) apenas para lidar com precisão de float
        condition_large = ee.Filter.Or(
            ee.Filter.gte('width_m', PATCH_METERS - 10),
            ee.Filter.gte('height_m', PATCH_METERS - 10)
        )
        
        # GRUPO 1: Polígonos que cabem inteiros num único patch
        # (Pequenos em ambas as dimensões)
        small_polys = ee.FeatureCollection(fc_box).filter(condition_large.Not())
        points_small = small_polys.map(lambda f: f.centroid(1))

        # GRUPO 2: Polígonos que estouram o patch em largura OU altura
        large_polys = ee.FeatureCollection(fc_box).filter(condition_large)

        # Flatten pode ser perigoso se large_polys for enorme, mas sample() ajuda muito
        points_large = large_polys.map(create_grid_points_optimized).flatten()

        all_sample_points = points_small.merge(points_large)
        all_sample_points = all_sample_points.map(lambda feat: feat.set('WRS_PATH',int(wrs_path), 'WRS_ROW', int(wrs_row)))
        # print(f"   Polígonos Pequenos: {small_polys.size().getInfo()}")
        # print(f"   Pontos gerados de Grandes: {points_large.size().getInfo()}") # Evite chamar info aqui se for lento        
        
        export_feat(all_sample_points, nome_export)
        print("exportar >> " + nome_export)

        # sys.exit()
print("Script finalizado. Monitore as tarefas no EE Tasks.")