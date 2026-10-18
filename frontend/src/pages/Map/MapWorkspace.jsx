import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysis } from '../../context/AnalysisContext';
import { ArrowLeft, Plus, Minus, Maximize, Layers as LayersIcon, MapPin } from 'lucide-react';
import Navbar from '../../components/Navbar/Navbar';
import './MapWorkspace.css';

/**
 * ─────────────────────────────────────────────────────────────────
 * MapWorkspace — Shared "View on Map" experience for all models.
 * Uses a lightweight custom 2D grid viewer to avoid unnecessary dependencies,
 * fulfilling the UI preparation requirement.
 * ─────────────────────────────────────────────────────────────────
 */
const MapWorkspace = () => {
  const { analysisResult, files } = useAnalysis();
  const navigate = useNavigate();

  // Map state
  const containerRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  

  // Routing fallback
  const previousRoute = analysisResult?.route || '/';
  
  // Extract data to show on map based on current model
  const footprintImage = files.length > 0 ? files[0].preview : null;
  const overlayTitle = analysisResult?.selectedModel || 'Analysis Region';

  const handleBack = () => {
    navigate(previousRoute);
  };

  // ── Pan & Zoom Logic ──
  const ZOOM_STEP = 0.5;
  const MIN_SCALE = 0.5;
  const MAX_SCALE = 5;

  const zoomIn = () => setScale(s => Math.min(MAX_SCALE, s + ZOOM_STEP));
  const zoomOut = () => setScale(s => Math.max(MIN_SCALE, s - ZOOM_STEP));
  const resetView = () => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  };

  const handlePointerDown = (e) => {
    setIsPanning(true);
    setPanStart({ x: e.clientX - translate.x, y: e.clientY - translate.y });
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e) => {
    if (isPanning) {
      setTranslate({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
    }

  };

  const handlePointerUp = () => {
    setIsPanning(false);
  };

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.2 : 0.2;
    setScale(s => Math.min(MAX_SCALE, Math.max(MIN_SCALE, s + delta)));
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.addEventListener('wheel', handleWheel, { passive: false });
      return () => el.removeEventListener('wheel', handleWheel);
    }
  }, [handleWheel]);

  return (
    <div className="map-page-container">
      <Navbar />
      
      <main className="map-workspace">
        
        {/* ── Top Controls Bar ── */}
        <div className="map-top-bar">
          <button className="btn-map-nav" onClick={handleBack}>
            <ArrowLeft size={16} /> Back to Analysis
          </button>
          
          <div className="map-title">
            <MapPin size={16} className="text-cyan" />
            <span>Geospatial View — {overlayTitle}</span>
          </div>

          <div className="map-tools">
            <button className="btn-map-tool" title="Layers" aria-label="Layers">
              <LayersIcon size={16} />
            </button>
            <div className="map-tool-divider" />
            <button className="btn-map-tool" onClick={zoomOut} title="Zoom Out" aria-label="Zoom Out">
              <Minus size={16} />
            </button>
            <button className="btn-map-tool" onClick={resetView} title="Reset View" aria-label="Reset View">
              <Maximize size={14} />
            </button>
            <button className="btn-map-tool" onClick={zoomIn} title="Zoom In" aria-label="Zoom In">
              <Plus size={16} />
            </button>
          </div>
        </div>

        {/* ── Interactive Map Canvas ── */}
        <div 
          className="map-canvas-container"
          ref={containerRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
          style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
        >
          {/* Simulated Background Grid */}
          <div className="map-grid-bg" />

          {/* Transform Layer */}
          <div 
            className="map-transform-layer"
            style={{ transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})` }}
          >
            {/* Image Footprint Overlay */}
            {footprintImage && (
              <div className="map-footprint">
                <img src={footprintImage} alt="Footprint" className="footprint-img" draggable={false} />
                
                
                <div className="footprint-label glass-panel">
                  {analysisResult?.inputType || 'Image Footprint'}
                </div>
              </div>
            )}
            
          </div>
        </div>

        {/* ── Bottom HUD ── */}
        <div className="map-bottom-hud">
          <div className="map-coords glass-panel">
            <span className="coord-label">Image-space preview — no georeferencing supplied</span>
          </div>
          
          <div className="map-scale-bar">
            <span>Scale unavailable</span>
            <div className="scale-line" />
          </div>
        </div>

      </main>
    </div>
  );
};

export default MapWorkspace;
