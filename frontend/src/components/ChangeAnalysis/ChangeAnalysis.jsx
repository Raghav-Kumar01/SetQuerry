import React from 'react';
import './ChangeAnalysis.css';

const ChangeAnalysis = () => {
  return (
    <section className="change-analysis-section section-container" id="analyze">
      <h2 className="section-title text-cyan">CHANGE ANALYSIS</h2>
      
      <div className="selection-bar glass-panel">
        <div className="selection-item">
          <span className="label">Selected Region:</span>
          <span className="value">Himalayan Glacier</span>
        </div>
        <div className="selection-item">
          <span className="label">Date:</span>
          <span className="value">2023 - 2024</span>
        </div>
      </div>

      <div className="image-comparison-grid">
        <div className="image-box glass-panel">
          <div className="image-label">BEFORE</div>
          <div className="image-placeholder">Satellite Image</div>
        </div>
        <div className="image-box glass-panel">
          <div className="image-label">AFTER</div>
          <div className="image-placeholder">Satellite Image</div>
        </div>
      </div>

      <button className="analyze-btn shadow-cyan">
        ANALYZE CHANGE
      </button>
    </section>
  );
};

export default ChangeAnalysis;
