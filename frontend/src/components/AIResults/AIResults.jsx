import React from 'react';
import './AIResults.css';

const AIResults = () => {
  return (
    <section className="ai-results-section section-container">
      <h2 className="section-title text-cyan">AI ANALYSIS</h2>
      
      <div className="model-cards">
        <div className="model-card glass-panel">
          <div className="card-title">IMAGE UNDERSTANDING</div>
          <div className="indicator text-cyan">◉</div>
        </div>
        <div className="model-card glass-panel">
          <div className="card-title">TEMPORAL CHANGE</div>
          <div className="indicator text-cyan">◉</div>
        </div>
        <div className="model-card glass-panel">
          <div className="card-title">OPTICAL–SAR ANALYSIS</div>
          <div className="indicator text-cyan">◉</div>
        </div>
      </div>

      <div className="analysis-result">
        <h3 className="result-title">Analysis Result</h3>
        <div className="result-box glass-panel">
          <div className="result-content">
            <h4>Detected Changes</h4>
            <ul>
              <li>Affected region</li>
              <li>Change visualization</li>
              <li>AI-generated insights</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AIResults;
