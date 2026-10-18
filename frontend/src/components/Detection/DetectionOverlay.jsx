import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertTriangle, CheckCircle, ArrowRight, BrainCircuit } from 'lucide-react';
import './DetectionOverlay.css';

const DetectionOverlay = ({ result, error, onRetry }) => {
  const navigate = useNavigate();
  const [phase, setPhase] = useState('detecting');

  useEffect(() => {
    if (error) {
      // eslint-disable-next-line react-hooks/exhaustive-deps
      setPhase('error');
      return;
    }

    if (result) {
      setPhase('complete');
    }
  }, [result, error]);

  const handleProceed = () => {
    if (result && result.route) {
      navigate(result.route);
    }
  };

  return (
    <div className="detection-overlay glass-panel">
      
      {phase === 'error' && (
        <div className="detection-state error-state">
          <AlertTriangle size={48} className="text-red mb-4" style={{ color: '#ef4444' }} />
          <h3>Upload could not be routed</h3>
          <p className="text-muted">{error}</p>
          <button className="btn-retry" onClick={onRetry}>Try Again</button>
        </div>
      )}

      {(phase === 'understanding' || phase === 'detecting') && (
        <div className="detection-state processing-state">
          <BrainCircuit size={48} className="text-cyan pulse-anim mb-4" />
          <h3 className="pulse-text">
            Selecting route from file count and names...
          </h3>
          <div className="progress-bar-container mt-4">
            <div className={`progress-bar-fill ${phase}`} />
          </div>
        </div>
      )}

      {phase === 'complete' && result && (
        <div className="detection-state complete-state">
          <CheckCircle size={48} className="text-cyan mb-4" />
          <h3 className="mb-4">Route Selected</h3>
          <p className="text-muted mb-6">Selection uses file count and filename hints; imagery is evaluated on the next screen.</p>
          
          <div className="detection-result-box mb-6">
            <div className="result-row">
              <span className="result-label">AUTO-DETECTED INPUT</span>
              <span className="result-value text-cyan">{result.inputType}</span>
            </div>
            <div className="result-divider" />
            <div className="result-row">
              <span className="result-label">AUTO-SELECTED MODEL</span>
              <span className="result-value text-cyan">{result.selectedModel}</span>
            </div>
          </div>

          <button className="btn-proceed" onClick={handleProceed}>
            Proceed to Analysis <ArrowRight size={18} />
          </button>
        </div>
      )}
    </div>
  );
};

export default DetectionOverlay;
