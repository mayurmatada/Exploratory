// -----------------------------
// GEE Script: Sentinel-1, MODIS LAI, SMAP, with Urban + Yellow River Masking (Using MODIS Land Cover)
// -----------------------------

// 1. Define 11x11 km region in North China Plain
var region = ee.Geometry.Polygon([
  [
    [114.1000, 34.7950],
    [114.1000, 35.1150],
    [114.3900, 35.1150],
    [114.3900, 34.7950]
  ]
]);

Map.centerObject(region, 10);
Map.addLayer(region, {color: 'red'}, 'Study Region');

// 2. Date Range
var startDate = '2015-01-01';
var endDate = '2023-12-31';

// 3. Sentinel-1 GRD VV, VH
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(region)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH', 'angle']);

// 4. MODIS LAI
var laiCol = ee.ImageCollection('MODIS/061/MCD15A3H')
  .filterBounds(region)
  .filterDate(startDate, endDate)
  .select('Lai');

// 5. SMAP Rootzone Soil Moisture
var smCol = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
  .filterBounds(region)
  .filterDate(startDate, endDate)
  .select('sm_rootzone');

// 6. Mask Urban Areas using MODIS Land Cover (MODIS Collection 5)
var modisLandCover = ee.ImageCollection('MODIS/006/MCD12Q1')
  .filterBounds(region)
  .filterDate(startDate, endDate)
  .first()
  .select('LC_Type1');

var urbanMask = modisLandCover.eq(12);  // Urban area is typically class 12 in MODIS land cover
Map.addLayer(urbanMask.updateMask(urbanMask), {palette: ['gray']}, 'Urban Mask');

// 7. Mask Yellow River using MNDWI
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(region)
  .filterDate('2020-06-01', '2020-09-01')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .median()
  .clip(region);

var green = s2.select('B3');
var swir = s2.select('B11');
var mndwi = green.subtract(swir).divide(green.add(swir));
var waterMask = mndwi.gt(0.2);  // Threshold may be adjusted
Map.addLayer(waterMask.updateMask(waterMask), {palette: ['blue']}, 'Yellow River Mask');

// 8. Combine Masks (urban + river)
var fullMask = urbanMask.or(waterMask);
Map.addLayer(fullMask.updateMask(fullMask), {palette: ['black']}, 'Final Mask');

// 9. Map and extract data
var combined = s1.map(function(image) {
  var date = ee.Date(image.get('system:time_start'));

  // Apply full mask
  var maskedImage = image.updateMask(fullMask.not());

  // VV, VH
  var vv = maskedImage.select('VV').reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: 10,
    bestEffort: true
  }).get('VV');

  var vh = maskedImage.select('VH').reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: 10,
    bestEffort: true
  }).get('VH');

  var angle = maskedImage.select('angle').reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: 30,
    bestEffort: true
  }).get('angle');

  // Get LAI ±3 days
  var laiImg = laiCol.filterDate(date.advance(-3, 'day'), date.advance(3, 'day')).first();
  var lai = ee.Algorithms.If(
    laiImg,
    ee.Number(
      ee.Image(laiImg).reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: region,
        scale: 500,
        bestEffort: true
      }).get('Lai')
    ),
    null
  );

  // Only multiply if LAI is not null
  var laiValue = ee.Algorithms.If(
    ee.Algorithms.IsEqual(lai, null),
    null, // If lai is null, return null
    ee.Number(lai).multiply(0.1) // Multiply by 0.1 if valid number
  );

  // Get SM ±3 days
  var smImg = smCol.filterDate(date.advance(-3, 'day'), date.advance(3, 'day')).first();
  var sm = ee.Algorithms.If(
    smImg,
    ee.Image(smImg).reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: region,
      scale: 9000,
      bestEffort: true
    }).get('sm_rootzone'),
    null
  );

  // Null check for each value before return
  return ee.Feature(null, {
    'date': date.format('YYYY-MM-dd'),
    'VV': ee.Algorithms.If(ee.Algorithms.IsEqual(vv, null), null, vv),
    'VH': ee.Algorithms.If(ee.Algorithms.IsEqual(vh, null), null, vh),
    'LAI': laiValue,
    'SoilMoisture': ee.Algorithms.If(ee.Algorithms.IsEqual(sm, null), null, sm),
    'IncidenceAngle': ee.Algorithms.If(ee.Algorithms.IsEqual(angle, null), null, angle),
    'Frequency_GHz': 5.405,
    'SoilRoughness_placeholder': null
  });
});

// 10. Export as CSV
Export.table.toDrive({
  collection: ee.FeatureCollection(combined),
  description: 'Sentinel1_MODIS_SM_Masked_Urban_YellowRiver_11km',
  fileFormat: 'CSV'
});
