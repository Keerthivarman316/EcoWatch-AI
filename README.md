📌 Overview
EcoWatch AI is a deep learning-powered environmental compliance monitoring system designed for Pollution Control Boards and environmental regulators. It automatically detects green belt violations and unauthorized construction within industrial premises by analysing multi-temporal satellite imagery fused with KGIS (Karnataka GIS) boundary data.
Traditional compliance monitoring relies on manual inspections — slow, expensive, and easily manipulated. EcoWatch AI makes it continuous, scalable, and evidence-backed.
What it does

Ingests Sentinel-2 satellite imagery and KGIS industrial boundary shapefiles
Detects vegetation (green cover) loss within factory premises over time
Identifies land-use changes indicating unauthorized construction
Generates geo-stamped compliance reports with annotated before/after imagery
Highlights violation regions using GradCAM for explainable, legally defensible evidence

Sentinel-2 Imagery (Multi-temporal)
           +
KGIS Industrial Boundary Shapefiles
           │
           ▼
   ┌───────────────────┐
   │  Preprocessing    │
   │  • Radiometric    │
   │    correction     │
   │  • Cloud masking  │
   │  • NDVI compute   │
   │  • GIS clip & AOI │
   └────────┬──────────┘
            │
            ▼
   ┌────────────────────────────────┐
   │   Shared Encoder               │
   │   ResNet-50 / ViT-B            │
   │   (pretrained on BigEarthNet)  │
   └──────────┬─────────────────────┘
              │
       ┌──────┴──────┐
       ▼             ▼
  ┌─────────┐   ┌──────────────┐
  │ Change  │   │ Vegetation   │
  │ Det.    │   │ Segmentation │
  │ Head    │   │ Head         │
  │ (BIT /  │   │ (SegFormer)  │
  │ Siamese │   │              │
  │ U-Net)  │   │              │
  └────┬────┘   └──────┬───────┘
       │               │
       └───────┬───────┘
               ▼
   ┌───────────────────────┐
   │  GIS Boundary Mask    │
   │  (KGIS factory poly)  │
   └──────────┬────────────┘
              │
              ▼
   ┌──────────────────────┐
   │  Rule Engine         │
   │  Green cover Δ > 15% │
   │  → Violation flagged │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │  GradCAM++ / SHAP    │
   │  Explainability      │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │  Compliance Report   │
   │  • Violation alerts  │
   │  • Geo-coordinates   │
   │  • Annotated imagery │
   │  • Temporal trends   │
   └──────────────────────┘

   
