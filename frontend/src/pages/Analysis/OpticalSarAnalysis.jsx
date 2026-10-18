import React, { useState, useEffect } from 'react';
import { useAnalysis } from '../../context/AnalysisContext';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, MapPin, Activity, CheckCircle2, AlertTriangle } from 'lucide-react';
import Navbar from '../../components/Navbar/Navbar';
import ProcessingState from '../../components/ProcessingState/ProcessingState';
import { analysisService } from '../../services/analysisService';
import './OpticalSarAnalysis.css';
import InferenceNotice from '../../components/InferenceNotice';

const OpticalSarAnalysis = () => {
  const { files, analysisResult, resetAnalysis } = useAnalysis();
  const navigate = useNavigate();
  const [status, setStatus] = useState('processing');
  const [resultData, setResultData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // The detection logic doesn't guarantee which is which, 
  // but for the UI we'll assign one to optical and one to SAR based on filename or order.
  const isSarFile = (f) => {
    const name = (f.name || '').toLowerCase().replace(/\.[^.]+$/, '');
    return name === 'sar' || name === 's1' || name.startsWith('sar_') || name.startsWith('s1_') || name.includes('_sar') || name.includes('radar');
  };

  const reverseAutoPair = analysisResult?.selection !== 'manual' && files[0] && isSarFile(files[0]) && files[1] && !isSarFile(files[1]);
  const opticalFile = files[reverseAutoPair ? 1 : 0];
  const sarFile = files[reverseAutoPair ? 0 : 1];

  useEffect(() => {
    if (!opticalFile || !sarFile) return;
    
    let isMounted = true;
    
    const fetchAnalysis = async () => {
      try {
        const data = await analysisService.runOpticalSarAnalysis(opticalFile.file, sarFile.file);
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
  }, [opticalFile, sarFile]);

  const handleBack = () => {
    resetAnalysis();
    navigate('/');
  };

  if (!opticalFile || !sarFile) {
    return (
      <div className="home-container">
        <Navbar />
        <main className="osa-empty">
          <p>An Optical and SAR image pair is required. Please go back and upload the correct pair.</p>
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

      <main className="osa-workspace">
        {/* ── Top Status Bar ── */}
        <div className="osa-status-bar">
          <div className="osa-status-item">
            <span className="osa-status-label">INPUT</span>
            <span className="osa-status-value">
              {analysisResult?.inputType || 'Optical + SAR Pair'}
            </span>
          </div>
          <div className="osa-status-divider" />
          <div className="osa-status-item">
            <span className="osa-status-label">SELECTED MODEL</span>
            <span className="osa-status-value text-cyan">
              {analysisResult?.selectedModel || 'Optical–SAR Analysis'}
            </span>
          </div>
          <div className="osa-status-actions">
            <button className="btn-action-sm" onClick={handleBack}>
              <ArrowLeft size={14} /> Analyze Another Pair
            </button>
            <button className="btn-action-sm" onClick={() => navigate('/map')}>
              <MapPin size={14} /> View on Map
            </button>
          </div>
        </div>

        {/* Processing Overlay */}
        {status === 'processing' && (
          <ProcessingState 
            modelName={analysisResult?.selectedModel || 'Optical–SAR Analysis'} 
          />
        )}

        <div style={{ opacity: status === 'complete' ? 1 : 0.3, transition: 'opacity 0.5s ease', pointerEvents: status === 'complete' ? 'auto' : 'none' }}>
        
        {/* ── Main Symmetric Visual Layout ── */}
        <div className="osa-visual-container">
          
          {/* OPTICAL PANEL */}
          <div className="osa-image-panel">
            <div className="osa-panel-header glass-panel">
              <span className="osa-panel-title-text">OPTICAL</span>
              <div className="osa-metadata">
                <span>Sensor: {resultData?.metadata?.optical?.sensor || 'Optical'}</span>
                <span>Date: {resultData?.metadata?.optical?.date || 'Unknown'}</span>
                <span>Bands: {resultData?.metadata?.optical?.bands || 'RGB'}</span>
              </div>
            </div>
            <div className="osa-image-wrapper">
              <img src={opticalFile.preview} alt="Optical Satellite" className="osa-image" />
            </div>
          </div>

          {/* CENTER RELATIONSHIP INDICATOR */}
          <div className="osa-center-indicator">
            <div className="osa-indicator-line" />
            <div className="osa-indicator-icon glass-panel shadow-cyan">
              {status === 'complete' ? <Activity size={18} className="text-cyan" /> : status === 'error' ? <AlertTriangle size={16} className="text-red-500" /> : <div className="pulse" style={{ width: 8, height: 8, background: '#f59e0b', borderRadius: '50%' }} />}
            </div>
            <div className="osa-indicator-line" />
          </div>

          {/* SAR PANEL */}
          <div className="osa-image-panel">
            <div className="osa-panel-header glass-panel">
              <span className="osa-panel-title-text text-cyan">SAR</span>
              <div className="osa-metadata">
                <span>Sensor: {resultData?.metadata?.sar?.sensor || 'SAR'}</span>
                <span>Date: {resultData?.metadata?.sar?.date || 'Unknown'}</span>
                <span>Polarization: {resultData?.metadata?.sar?.polarization || 'Unknown'}</span>
              </div>
            </div>
            <div className="osa-image-wrapper">
              <img src={sarFile.preview} alt="SAR Radar" className="osa-image osa-image--sar" />
            </div>
          </div>

        </div>

        {/* ── Wide Result Panel ── */}
        <div className="osa-result-container">
          <section className="osa-result-panel glass-panel">
            <div className="osa-result-header">
              <h2 className="text-cyan">
                {status === 'complete' ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><CheckCircle2 size={16} /> Cross-Modal AI Analysis</span>
                ) : status === 'error' ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#ef4444' }}><AlertTriangle size={16} /> {errorMessage || 'Analysis Failed'}</span>
                ) : 'Processing...'}
              </h2>
              <div className="osa-confidence-badge">
                <span className="osa-conf-val text-cyan">{resultData?.label === 'BASELINE' ? (resultData.similarity_score == null ? 'N/A' : resultData.similarity_score.toFixed(4)) : (resultData ? `${Math.round(resultData.confidence * 100)}%` : '—')}</span>
                <span className="osa-conf-label">{resultData?.label === 'BASELINE' ? 'Structural similarity (0–1), not probability' : 'Uncalibrated class score'}</span>
              </div>
            </div>

            <div className="osa-result-grid">
              
              {resultData && (
                <>
                  {/* Left Column */}
                  <div className="osa-result-col">
                    <div className="osa-result-section">
                      <h3 className="osa-section-title">Method and limitations</h3>
                    <InferenceNotice result={resultData} />
                      <ul className="osa-features-list">
                        {resultData.evidence.map((f, i) => (
                          <li key={i}>{f}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="osa-result-section mt-4">
                      <h3 className="osa-section-title">AI Insight</h3>
                      <p className="osa-insight-text">{resultData.aiInsight}</p>
                    </div>
                  </div>

                  {/* Middle Column */}
                  <div className="osa-result-col">
                    <div className="osa-result-section">
                      <h3 className="osa-section-title">Cross-Modal Agreement</h3>
                      <p className="osa-body-text">{resultData.agreement}</p>
                    </div>
                    
                    <div className="osa-result-section mt-4">
                      <h3 className="osa-section-title">Cross-Modal Differences</h3>
                      <p className="osa-body-text">{resultData.differences}</p>
                    </div>
                  </div>

                  {/* Right Column */}
                  <div className="osa-result-col">
                    <div className="osa-result-section">
                      <h3 className="osa-section-title">Analysis Result</h3>
                      <p className="osa-body-text">{resultData.analysisResult}</p>
                    </div>
                  </div>
                </>
              )}

            </div>
            
            <div className="osa-demo-notice">
              Alignment scores require validation on real optical and SAR pairs.
            </div>
          </section>
        </div>
        </div>
      </main>
    </div>
  );
};

export default OpticalSarAnalysis;
