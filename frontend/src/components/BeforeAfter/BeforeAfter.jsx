import React from 'react';
import './BeforeAfter.css';

const BeforeAfter = () => {
  return (
    <section className="before-after-section section-container" id="compare">
      <h2 className="section-title text-cyan">VISUAL COMPARISON</h2>
      
      <div className="labels">
        <span className="label">BEFORE</span>
        <span className="label">AFTER</span>
      </div>

      <div className="slider-container glass-panel">
        <div className="slider-placeholder">
          BEFORE IMAGE ←──── draggable slider ────→ AFTER IMAGE
        </div>
      </div>

      <div className="metrics glass-panel">
        <div className="metric-item">
          <span className="metric-label">Change %</span>
          <span className="metric-value text-cyan">XX%</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Affected Area</span>
          <span className="metric-value text-cyan">XX km²</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Confidence</span>
          <span className="metric-value text-cyan">XX%</span>
        </div>
      </div>
    </section>
  );
};

export default BeforeAfter;
