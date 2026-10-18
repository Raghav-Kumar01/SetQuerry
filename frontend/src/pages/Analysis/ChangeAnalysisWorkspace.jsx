import React, { useState, useEffect } from 'react';
import { useAnalysis } from '../../context/AnalysisContext';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, MapPin, Layers, CheckCircle2, AlertTriangle } from 'lucide-react';
import Navbar from '../../components/Navbar/Navbar';
import ComparisonSlider from '../../components/ComparisonSlider/ComparisonSlider';
import ProcessingState from '../../components/ProcessingState/ProcessingState';
import { analysisService } from '../../services/analysisService';
import './ChangeAnalysisWorkspace.css';
import InferenceNotice from '../../components/InferenceNotice';

const SEVERITY_COLORS = {
  high: '#ef4444',
  medium: '#f59e0b',
  low: '#34d399',
};

const ChangeAnalysisWorkspace = () => {
  const { files, analysisResult, resetAnalysis } = useAnalysis();
  const navigate = useNavigate();
  const [status, setStatus] = useState('processing');
  const [resultData, setResultData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const beforeSrc = files.length > 0 ? files[0].preview : null;
  const afterSrc  = files.length > 1 ? files[1].preview : null;

  useEffect(() => {
    if (!files || files.length < 2) return;
    
    let isMounted = true;
    
    const fetchAnalysis = async () => {
      try {
        const data = await analysisService.runChangeAnalysis(files[0].file, files[1].file);
        if (isMounted) {
          setResultData(data);
          setStatus('complete');
        }
      } catch (err) {
        if (isMounted) {
          console.error(err);
          setErrorMessage(err.message || 'Analysis failed');
          setStatus('error');
        }
      }
    };
    
    fetchAnalysis();
    
    return () => { isMounted = false; };
  }, [files]);

  const handleBack = () => {
    resetAnalysis();
    navigate('/');
  };

  if (!beforeSrc || !afterSrc) {
    return (
      <div className="home-container">
        <Navbar />
        <main className="ca-empty">
          <p>Two images required for change analysis. Please go back and upload a before/after pair.</p>
          <button className="btn-back" onClick={handleBack}>
            <ArrowLeft size={16} /> Back to Upload
          </button>
        </main>
      </div>
    );
  }

  return (
    <div className="home-container">
      <Navbar />

      <main className="ca-workspace">
        {/* ── Top Status Bar ── */}
        <div className="ca-status-bar">
          <div className="ca-status-item">
            <span className="ca-status-label">INPUT</span>
            <span className="ca-status-value">
              {analysisResult?.inputType || 'Before + After Image Pair'}
            </span>
          </div>
          <div className="ca-status-divider" />
          <div className="ca-status-item">
            <span className="ca-status-label">SELECTED MODEL</span>
            <span className="ca-status-value text-cyan">
              {analysisResult?.selectedModel || 'Bi-Temporal Change Analysis'}
            </span>
          </div>
          <div className="ca-status-actions">
            <button className="btn-action-sm" onClick={handleBack}>
              <ArrowLeft size={14} /> Analyze Another Pair
            </button>
            <button className="btn-action-sm" onClick={() => navigate('/map')}>
              <MapPin size={14} /> View Change on Map
            </button>
          </div>
        </div>

        {/* ── Main Split Layout ── */}
        <div className="ca-split">

          {/* Processing Overlay */}
          {status === 'processing' && (
            <ProcessingState 
              modelName={analysisResult?.selectedModel || 'Bi-Temporal Change Analysis'} 
            />
          )}

          {/* LEFT — Comparison Viewer ~70% */}
          <div className="ca-comparison-panel">
            <ComparisonSlider
              beforeSrc={beforeSrc}
              afterSrc={afterSrc}
              beforeDate={resultData?.metadata?.beforeDate || 'Before'}
              afterDate={resultData?.metadata?.afterDate || 'After'}
            />

            {/* Change Mask Preview */}
            <div className="ca-mask-preview glass-panel">
              <div className="ca-mask-header">
                <Layers size={14} className="text-cyan" />
                <span>CHANGE MASK</span>
              </div>
              <div className="ca-mask-placeholder">
                {status === 'complete' && resultData?.maskBase64
                  ? <img src={`data:image/png;base64,${resultData.maskBase64}`} alt="Model-predicted binary change mask" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                  : <span>{status === 'error' ? 'Mask unavailable' : 'Processing...'}</span>}
              </div>
            </div>
          </div>

          {/* RIGHT — Result Panel ~30% */}
          <aside className="ca-result-panel glass-panel">
            <h2 className="ca-panel-title text-cyan">Temporal Change Detection</h2>
            <div className="ca-analysis-status">
              {status === 'complete' ? (
                <>
                  <CheckCircle2 size={14} className="text-cyan" />
                  Analysis Complete
                </>
              ) : status === 'error' ? (
                <>
                  <AlertTriangle size={14} style={{ color: '#ef4444' }} />
                  <span style={{ color: '#ef4444' }}>{errorMessage || 'Analysis Failed'}</span>
                </>
              ) : (
                <>
                  <span className="ca-status-dot pulse" style={{ background: '#f59e0b', boxShadow: '0 0 8px #f59e0b' }} />
                  Processing...
                </>
              )}
            </div>

            <div style={{ opacity: status === 'complete' ? 1 : 0.3, transition: 'opacity 0.5s ease', pointerEvents: status === 'complete' ? 'auto' : 'none' }}>
            
            {resultData && (
              <>
                {/* Change Classification */}
                <section className="ca-section">
                  <h3 className="ca-section-title">Change Classification</h3>
                    <InferenceNotice result={resultData} />
                  <ul className="ca-class-list">
                    {resultData.changeClassification.map((c, i) => (
                      <li key={i} className="ca-class-item">
                        <span
                          className="ca-severity-dot"
                          style={{ background: SEVERITY_COLORS[c.severity] }}
                        />
                        <span className="ca-class-name">{c.className}</span>
                        <span className="ca-class-area">{c.area}</span>
                      </li>
                    ))}
                  </ul>
                </section>

                {/* Affected Area */}
                <section className="ca-section">
                  <h3 className="ca-section-title">Affected Area</h3>
                  <div className="ca-area-grid">
                    <div className="ca-area-stat">
                      <span className="ca-area-number text-cyan">
                        {resultData.affectedArea.totalChanged}
                      </span>
                      <span className="ca-area-label">Total Changed</span>
                    </div>
                    <div className="ca-area-stat">
                      <span className="ca-area-number text-cyan">
                        {resultData.affectedArea.percentChanged}
                      </span>
                      <span className="ca-area-label">% of Scene</span>
                    </div>
                    <div className="ca-area-stat">
                      <span className="ca-area-number">
                        {resultData.affectedArea.totalAnalyzed}
                      </span>
                      <span className="ca-area-label">Total Analyzed</span>
                    </div>
                  </div>
                </section>

                {/* Change Summary */}
                <section className="ca-section">
                  <h3 className="ca-section-title">Change Summary</h3>
                  <p className="ca-section-body">{resultData.changeSummary}</p>
                </section>

                {/* Confidence */}
                <section className="ca-section">
                  <h3 className="ca-section-title">{resultData.label === 'BASELINE' ? 'Changed pixels' : 'Uncalibrated model score'}</h3>
                  <div className="ca-confidence">
                    <span className="ca-conf-number text-cyan">
                      {resultData.label === 'BASELINE' ? resultData.change_percentage.toFixed(2) : Math.round(resultData.confidence * 100)}%
                    </span>
                    <span className="ca-conf-label">{resultData.label === 'BASELINE' ? 'Image-difference area; not confidence' : 'Mean selected-pixel probability'}</span>
                  </div>
                </section>

                {/* AI Insight */}
                <section className="ca-section">
                  <h3 className="ca-section-title">AI Insight</h3>
                  <p className="ca-section-body ca-insight">{resultData.aiInsight}</p>
                </section>
              </>
            )}

            {/* Demo notice */}
            <div className="ca-demo-notice">
              Pixel differences can include lighting, motion, and registration errors.
            </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
};

export default ChangeAnalysisWorkspace;
