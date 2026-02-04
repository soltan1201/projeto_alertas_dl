var date_inic = '2025-09-01';
var name_month = 'setembro';
var name_folder = 'Setembro2025';

var asset_fcAlert = 'projects/mapbiomas-caatinga-cloud04/assets/Alertas/revisados_2025/alerts_pol_' + name_month + '_' + date_inic;
var asset_folderAlert = 'projects/mapbiomas-caatinga-cloud04/assets/Alertas/gerados_2025/' + name_folder;
var asset_output = 'projects/mapbiomas-caatinga-cloud04/assets/Alertas/featCol_samples/';
// 1. Carregar alertas aprovados
var featC_alerts = ee.FeatureCollection(asset_fcAlert)
                          .map(function(feat){return feat.set('aprovado', true, 'idCod', 1)});

print("Feature Collection alerts ", featC_alerts.limit(4));
var size_alerts = featC_alerts.size();
print("Feature Collection alerts tamanho ", size_alerts);

function export_feat (featcol, nome_exp){
    Export.table.toAsset({
                    collection: featcol,
                    description: nome_exp,
                    assetId: asset_output + nome_exp
                });
    print(" exporting ...  " + nome_exp);
}
var mask_alerts = featC_alerts.reduceToImage(['idCod'], ee.Reducer.first())
Map.addLayer(mask_alerts.selfMask(), {min:0 , max: 1, palette: ['#FF0000']}, 'alerts');

var featPoints_alerts = ee.FeatureCollection(asset_fcAlert)
                            .map(function(feat){return feat.centroid()});
                            
featPoints_alerts = featPoints_alerts.geometry();
// 1. Listar todos os assets no folder
var assetList = ee.data.listAssets(asset_folderAlert, {'maxResults': 1000});
var number_grids = ee.List(assetList.assets).size();
print("numero de featCollection em folders  ", number_grids);

var num_coleta = ee.Number(size_alerts).multiply(100).divide(number_grids);
print("Numero de alertas a sere coletado por grids", num_coleta);

var assetIds = assetList.assets.map(function(asset) {return asset.id; });
print('Assets encontrados no folder:', assetIds);

// 2. Carregar todos os FeatureCollections do folder e combiná-los

var counting = 0;
// Combinar todos os FeatureCollections em um único
assetIds.forEach(function(assetId) {
    var fc = ee.FeatureCollection(assetId);
    // 1. Extrair o nome da feature a partir do assetId
    var assetIdString = assetId; // Garante que é string
    var name_export = assetIdString.split('/').pop();
    print(name_export);
    // 2. Acessa os índices (lembrando que a contagem começa em 0)
    // shp(0) alerta(1) 215(2) 64(3)
    var grid_x = parts[2]; // "215"
    var grid_y = parts[3]; // "64"

    print('Coluna:', grid_x);
    print('Linha:', grid_y);
    if (counting === 0){
        print("show metadata ", fc.limit(10));
        counting += 1;
    }
    // Filtra a pasta: mantém apenas o que NÃO intersecta a geometria dos aprovados
    var fc_remain = fc.filter(ee.Filter.bounds(featPoints_alerts).not());
    // print(fc_remain.size());
    fc_remain = fc_remain.map(function(feat){return feat.set('area', feat.area())});
    fc_remain = fc_remain.randomColumn({rowKeys: ['area']}); //distribution: 'normal', 
    fc_remain = fc_remain.filter(ee.Filter.lt('random', ee.Number(num_coleta).divide(ee.Number(fc.size()))));
    print(" rest feature Collection ", fc_remain.size());
    // Opcional: Exportar o resultado limpo
    var name_export = '';
    export_feat(fc_remain, name_export);
    
});

print("Total de alerts no folder (combinados):", allFolderAlerts.size());









