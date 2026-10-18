/**
 * Demo Mode Data
 * 
 * This file strictly isolates all frontend placeholder data.
 * When real APIs are connected, VITE_USE_DEMO_DATA=false will completely bypass this file.
 */

export const DEMO_SINGLE_ANALYSIS = {
  annotations: [
    { id: 'a1', x: 28, y: 35, label: 'Urban Area (DEMO)',       color: '#06b6d4' },
    { id: 'a2', x: 62, y: 22, label: 'Vegetation (DEMO)',       color: '#34d399' },
    { id: 'a3', x: 75, y: 65, label: 'Water Body (DEMO)',       color: '#60a5fa' },
    { id: 'a4', x: 40, y: 78, label: 'Agricultural Land (DEMO)', color: '#fbbf24' },
  ],
  sceneSummary: '[DEMO DATA] High-resolution satellite capture showing a mixed-use region with urban development, agricultural zones, natural vegetation, and a visible water body.',
  detectedFeatures: [
    { name: 'Urban Infrastructure', confidence: 0.94 },
    { name: 'Vegetation Cover',    confidence: 0.91 },
    { name: 'Water Body',          confidence: 0.88 },
    { name: 'Agricultural Land',   confidence: 0.85 },
  ],
  observations: [
    '[DEMO] Dense urban settlement visible in the north-west quadrant.',
    '[DEMO] Healthy vegetation canopy detected in the north-east region.',
    '[DEMO] Surface water body identified — likely a reservoir or lake.',
    '[DEMO] Regular agricultural field patterns in the southern portion.',
  ],
  overallConfidence: 0.92,
  aiInsight: '[DEMO DATA] The scene suggests an actively developing peri-urban zone where agricultural land borders expanding urban infrastructure. The water body may serve as a shared resource for both urban and agricultural use.',
};

export const DEMO_CHANGE_ANALYSIS = {
  metadata: {
    beforeDate: '2023-03-15 (DEMO)',
    afterDate: '2024-09-22 (DEMO)',
  },
  changeClassification: [
    { className: 'Urban Expansion (DEMO)',     area: '12.4 km²', severity: 'high' },
    { className: 'Deforestation (DEMO)',        area: '3.8 km²',  severity: 'high' },
    { className: 'Water Level Change (DEMO)',   area: '1.2 km²',  severity: 'medium' },
    { className: 'Agricultural Shift (DEMO)',   area: '5.6 km²',  severity: 'low' },
  ],
  affectedArea: {
    totalChanged: '23.0 km²',
    percentChanged: '18.4%',
    totalAnalyzed: '125.0 km²',
  },
  changeSummary: '[DEMO DATA] Significant land-use changes detected between the two temporal captures. Urban areas have expanded notably in the south-western quadrant, with corresponding vegetation loss. Water body boundaries have shifted, suggesting seasonal or man-made changes.',
  confidence: 0.89,
  aiInsight: '[DEMO DATA] The pattern of urban expansion combined with deforestation suggests active development. The water level change may correlate with upstream infrastructure projects or seasonal variation. Recommend monitoring the deforestation front in subsequent captures.',
};

export const DEMO_OPTICAL_SAR = {
  metadata: {
    optical: { sensor: 'Sentinel-2 (DEMO)', date: '2023-11-14', bands: 'RGB (True Color)' },
    sar: { sensor: 'Sentinel-1 (DEMO)', date: '2023-11-15', polarization: 'VV + VH' }
  },
  detectedFeatures: [
    'Urban Infrastructure (DEMO)',
    'Vegetation Canopy (DEMO)',
    'Water Bodies (DEMO)',
    'Metallic Structures / Ships (DEMO)'
  ],
  agreement: '[DEMO DATA] High agreement on primary water body boundaries and permanent urban structures. Both modalities clearly identify the coastline and major highway infrastructure.',
  differences: '[DEMO DATA] SAR identifies several metallic point targets (likely vessels) in the water body that are obscured by cloud cover in the optical capture. Optical identifies vegetation health which is absent in SAR.',
  analysisResult: '[DEMO DATA] Cross-modal fusion successfully penetrated partial cloud cover. Critical infrastructure is fully mapped by combining optical texture with SAR structural returns.',
  confidence: 0.94,
  aiInsight: '[DEMO DATA] The fusion model leveraged SAR to compensate for the 15% cloud occlusion in the optical image, yielding a complete structural map of the coastal zone.',
};
