import React, { useState, useRef, useCallback, useEffect } from 'react';
import { ZoomIn, ZoomOut, Maximize, Move } from 'lucide-react';
import './ImageViewer.css';

/**
 * ImageViewer — Zoomable, pannable satellite image viewer with annotation overlay.
 *
 * Props:
 *   src          – image URL (object URL from upload or remote)
 *   annotations  – array of { id, x, y, label, color? } for overlay markers
 *                   x/y are percentages (0–100) relative to the image
 */
const ImageViewer = ({ src, annotations = [] }) => {
  const containerRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [tool, setTool] = useState('pan'); // 'pan'

  const MIN_SCALE = 0.5;
  const MAX_SCALE = 8;
  const ZOOM_STEP = 0.3;

  const handleZoomIn = () => {
    setScale(prev => Math.min(MAX_SCALE, prev + ZOOM_STEP));
  };

  const handleZoomOut = () => {
    setScale(prev => Math.max(MIN_SCALE, prev - ZOOM_STEP));
  };

  const handleReset = () => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  };

  // Mouse wheel zoom
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
    setScale(prev => Math.min(MAX_SCALE, Math.max(MIN_SCALE, prev + delta)));
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  // Pan handlers
  const handlePointerDown = (e) => {
    if (tool !== 'pan') return;
    setIsPanning(true);
    setPanStart({ x: e.clientX - translate.x, y: e.clientY - translate.y });
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e) => {
    if (!isPanning) return;
    setTranslate({
      x: e.clientX - panStart.x,
      y: e.clientY - panStart.y,
    });
  };

  const handlePointerUp = () => {
    setIsPanning(false);
  };

  return (
    <div className="img-viewer">
      {/* Toolbar */}
      <div className="img-viewer__toolbar">
        <button
          className={`toolbar-btn ${tool === 'pan' ? 'active' : ''}`}
          onClick={() => setTool('pan')}
          title="Pan"
          aria-label="Pan"
        >
          <Move size={16} />
        </button>
        <div className="toolbar-divider" />
        <button className="toolbar-btn" onClick={handleZoomIn} title="Zoom In" aria-label="Zoom In">
          <ZoomIn size={16} />
        </button>
        <button className="toolbar-btn" onClick={handleZoomOut} title="Zoom Out" aria-label="Zoom Out">
          <ZoomOut size={16} />
        </button>
        <button className="toolbar-btn" onClick={handleReset} title="Reset View" aria-label="Reset View">
          <Maximize size={16} />
        </button>
        <div className="toolbar-divider" />
        <span className="toolbar-zoom-label">{Math.round(scale * 100)}%</span>
      </div>

      {/* Canvas area */}
      <div
        ref={containerRef}
        className="img-viewer__canvas"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
      >
        <div
          className="img-viewer__transform"
          style={{
            transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
          }}
        >
          <img
            src={src}
            alt="Satellite imagery"
            className="img-viewer__image"
            draggable={false}
          />

          {/* Annotation overlay */}
          <div className="img-viewer__annotations">
            {annotations.map((a) => (
              <div
                key={a.id}
                className="annotation-marker"
                style={{
                  left: `${a.x}%`,
                  top: `${a.y}%`,
                  '--marker-color': a.color || 'var(--accent-cyan)',
                }}
              >
                <span className="annotation-dot" />
                <span className="annotation-label">{a.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageViewer;
