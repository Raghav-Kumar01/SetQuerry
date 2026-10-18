import React from 'react';
import './RegionPanel.css';

const RegionPanel = () => {
  return (
    <div className="region-panel glass-panel">
      <h3 className="panel-title text-cyan">REGION INFO</h3>
      <div className="panel-content">
        <div className="info-item">
          <span className="label">Location</span>
          <span className="value">Himalayas, India</span>
        </div>
        <div className="info-item">
          <span className="label">Date</span>
          <span className="value">2024-08-15</span>
        </div>
        <div className="info-item">
          <span className="label">Source</span>
          <span className="value">Sentinel-2</span>
        </div>
      </div>
    </div>
  );
};

export default RegionPanel;
