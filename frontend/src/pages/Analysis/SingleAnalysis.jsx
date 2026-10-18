import React, { useState, useEffect } from 'react';
import { useAnalysis } from '../../context/AnalysisContext';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, MapPin, Send, CheckCircle2, AlertTriangle } from 'lucide-react';
import Navbar from '../../components/Navbar/Navbar';
import ImageViewer from '../../components/ImageViewer/ImageViewer';
import ProcessingState from '../../components/ProcessingState/ProcessingState';
import { analysisService } from '../../services/analysisService';
import './SingleAnalysis.css';
import InferenceNotice from '../../components/InferenceNotice';

const SingleAnalysis = () => {
  const { files, analysisResult, resetAnalysis } = useAnalysis();
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [submittedQuery, setSubmittedQuery] = useState(analysisResult?.question || 'What is in this satellite image?');
  const [status, setStatus] = useState('processing'); // 'processing', 'complete', 'error'
  const [resultData, setResultData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const imageSrc = files.length > 0 ? files[0].preview : null;

  useEffect(() => {
    if (!files || files.length === 0) return;
    
    let isMounted = true;
    
    const fetchAnalysis = async () => {
      try {
        setStatus('processing');
        setErrorMessage(null);
        setResultData(null);
        const data = await analysisService.runSingleAnalysis(files[0].file, submittedQuery);
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
  }, [files, submittedQuery]);

  const handleBack = () => {
    resetAnalysis();
    navigate('/');
  };

  const handleQuerySubmit = (e) => {
    e.preventDefault();
    if (query.trim()) setSubmittedQuery(query.trim());
    setQuery('');
  };

  if (!imageSrc) {
    return (
      <div className="home-container">
        <Navbar />
        <main className="sa-empty">
          <p>No image uploaded. Please go back and upload an image first.</p>
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

      <main className="sa-workspace">
        {/* ── Top Status Bar ── */}
        <div className="sa-status-bar">
          <div className="sa-status-item">
            <span className="sa-status-label">INPUT</span>
            <span className="sa-status-value">
              {analysisResult?.inputType || 'Single Satellite Image'}
            </span>
          </div>
          <div className="sa-status-divider" />
          <div className="sa-status-item">
            <span className="sa-status-label">SELECTED MODEL</span>
            <span className="sa-status-value text-cyan">
              {analysisResult?.selectedModel || 'Single-Image Understanding'}
            </span>
          </div>
          <div className="sa-status-actions">
            <button className="btn-back-sm" onClick={handleBack}>
              <ArrowLeft size={14} /> Analyze Another Image
            </button>
            <button className="btn-map-sm" onClick={() => navigate('/map')}>
              <MapPin size={14} /> View on Map
            </button>
          </div>
        </div>

        {/* ── Main Split Layout ── */}
        <div className="sa-split">
          
          {/* Processing Overlay */}
          {status === 'processing' && (
            <ProcessingState 
              modelName={analysisResult?.selectedModel || 'Single-Image Understanding'} 
            />
          )}

          {/* LEFT — Image Viewer ~70% */}
          <div className="sa-image-panel">
            <ImageViewer src={imageSrc} annotations={status === 'complete' && resultData ? resultData.annotations : []} />
          </div>

          {/* RIGHT — Analysis Panel ~30% */}
          <aside className="sa-result-panel glass-panel">
            <div className="sa-panel-header-section">
              <h2 className="sa-panel-title text-cyan">SetQuery AI Analysis</h2>
              {status === 'complete' && (
                <div className="sa-analysis-status">
                  <CheckCircle2 size={14} className="text-cyan" />
                  <span>Analysis Complete</span>
                </div>
              )}
              {status === 'error' && (
                <div className="sa-analysis-status error">
                  <AlertTriangle size={14} style={{ color: '#ef4444' }} />
                  <span style={{ color: '#ef4444' }}>{errorMessage || 'Analysis Failed'}</span>
                </div>
              )}
            </div>

            {/* If still processing, we blur or hide the results */}
            <div style={{ opacity: status === 'processing' ? 0.3 : 1, transition: 'opacity 0.5s ease', pointerEvents: status === 'processing' ? 'none' : 'auto' }}>
              
              {resultData && (
                <>
                  {/* Scene Summary */}
                  <section className="sa-section">
                    <h3 className="sa-section-title">Scene Summary</h3>
                    <InferenceNotice result={resultData} />
                    <p className="sa-section-body">{resultData.sceneSummary}</p>
                  </section>

                  {/* Detected Features */}
                  <section className="sa-section">
                    <h3 className="sa-section-title">Decoded answer</h3>
                    <ul className="sa-features-list">
                      {resultData.detectedFeatures.map((f, i) => (
                        <li key={i} className="sa-feature-item">
                          <span className="sa-feature-name">{f.name}</span>
                          <div className="sa-confidence-bar">
                            <div
                              className="sa-confidence-fill"
                              style={{ width: `${f.confidence * 100}%` }}
                            />
                          </div>
                          <span className="sa-confidence-val">
                            {(f.confidence * 100).toFixed(4)}%
                          </span>
                        </li>
                      ))}
                    </ul>
                  </section>

                  {/* Observations */}
                  <section className="sa-section">
                    <h3 className="sa-section-title">Observations</h3>
                    <ul className="sa-obs-list">
                      {resultData.observations.map((o, i) => (
                        <li key={i}>{o}</li>
                      ))}
                    </ul>
                  </section>

                  {/* Confidence */}
                  <section className="sa-section">
                    <h3 className="sa-section-title">Uncalibrated model score</h3>
                    <div className="sa-overall-confidence">
                      <span className="sa-conf-number text-cyan">
                        {(resultData.overallConfidence * 100).toFixed(4)}%
                      </span>
                      <span className="sa-conf-label">Answer softmax score</span>
                    </div>
                  </section>

                  {/* AI Insight */}
                  <section className="sa-section">
                    <h3 className="sa-section-title">AI Insight</h3>
                    <p className="sa-section-body sa-insight">{resultData.aiInsight}</p>
                  </section>
                </>
              )}

            {/* ── Query Area ── */}
            <div className="sa-query-area">
              <form onSubmit={handleQuerySubmit} className="sa-query-form">
                <input
                  type="text"
                  className="sa-query-input"
                  placeholder="Ask about this satellite scene..."
                  value={query}
                  maxLength={500}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <button type="submit" className="sa-query-btn" disabled={query.trim().length < 3} aria-label="Submit Query">
                  <Send size={16} />
                </button>
              </form>
            </div>

            </div>
          </aside>
        </div>
      </main>
    </div>
  );
};

export default SingleAnalysis;
