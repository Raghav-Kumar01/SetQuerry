import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, X, Image as ImageIcon, Layers, Map, Play } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { analysisService } from '../../services/analysisService';
import DetectionOverlay from '../Detection/DetectionOverlay';
import './UploadWorkspace.css';

const UploadWorkspace = () => {
  const { files, setFiles, analysisResult, setAnalysisResult } = useAnalysis();
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  const [selectedModel, setSelectedModel] = useState('single_image');
  const [question, setQuestion] = useState('What is in this satellite image?');

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(Array.from(e.target.files));
    }
  };

  const processFiles = (newFiles) => {
    const imageExtensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp'];
    const validFiles = newFiles.filter(file => {
      const lower = (file.name || '').toLowerCase();
      return imageExtensions.some(ext => lower.endsWith(ext));
    });
    if (validFiles.length !== newFiles.length || files.length + validFiles.length > 2) {
      setAnalysisError('Choose at most two PNG, JPEG, TIFF, WebP, or BMP images.');
      return;
    }
    setAnalysisError(null);
    
    const mappedFiles = validFiles.map(file => ({
      file,
      id: Math.random().toString(36).substring(2, 9),
      preview: URL.createObjectURL(file),
      name: file.name,
      size: (file.size / (1024 * 1024)).toFixed(2) + ' MB'
    }));
    
    setFiles(prev => [...prev, ...mappedFiles]);
    
    // Reset input so the same file can be selected again if needed
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (idToRemove) => {
    setFiles(prev => {
      const fileToRemove = prev.find(f => f.id === idToRemove);
      if (fileToRemove && fileToRemove.preview) {
        URL.revokeObjectURL(fileToRemove.preview);
      }
      return prev.filter(f => f.id !== idToRemove);
    });
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleAnalyze = async () => {
    if (files.length === 0) return;
    setIsAnalyzing(true);
    setAnalysisError(null);
    setAnalysisResult(null);

    try {
      const result = await analysisService.detectInputAndModel(files, selectedModel);
      setAnalysisResult({ ...result, question });
      navigate(result.route);
    } catch (err) {
      setAnalysisError(err.message);
    }
  };

  const handleRetry = () => {
    setIsAnalyzing(false);
    setAnalysisError(null);
    setAnalysisResult(null);
  };

  return (
    <section className="upload-workspace section-container" id="upload">
      <div className="upload-header">
        <h2 className="upload-title text-cyan">Analyze Earth Changes with AI</h2>
        <p className="upload-subtitle">
          Choose an analysis and upload its images in the order shown below.
        </p>
      </div>

      <div className="model-selection">
        <label htmlFor="analysis-model">Analysis model</label>
        <select id="analysis-model" value={selectedModel} onChange={e => setSelectedModel(e.target.value)} disabled={isAnalyzing}>
          <option value="single_image">Model 1 — Single Image VQA</option>
          <option value="change_analysis">Model 2 — Change Detection</option>
          <option value="optical_sar">Model 3 — Optical-SAR Diagnostic</option>
          <option value="auto">Automatic selection</option>
        </select>
        <p>{selectedModel === 'single_image' ? 'Upload one image and enter a question.' : selectedModel === 'optical_sar' ? 'Image 1: optical. Image 2: SAR. Manual selection preserves this order.' : 'Image 1: before. Image 2: after. Images should cover the same area.'}</p>
        {(selectedModel === 'single_image' || selectedModel === 'auto') && <>
          <label htmlFor="analysis-question">Question</label>
          <input id="analysis-question" value={question} onChange={e => setQuestion(e.target.value)} minLength={3} maxLength={500} />
        </>}
        {files.length === 2 && <button className="btn-action-sm" onClick={() => setFiles([...files].reverse())}>Swap image order</button>}
        {!isAnalyzing && analysisError && <p role="alert">{analysisError}</p>}
      </div>

      <div className="upload-area-wrapper">
        <div 
          className={`upload-dropzone glass-panel ${isDragging ? 'dragging' : ''} ${files.length > 0 ? 'has-files' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {isAnalyzing && (
            <DetectionOverlay 
              result={analysisResult} 
              error={analysisError} 
              onRetry={handleRetry} 
            />
          )}

          {files.length === 0 ? (
            <div className="dropzone-content empty-state" onClick={triggerFileInput}>
              <div className="upload-icon-container shadow-cyan">
                <UploadCloud size={48} className="text-cyan" />
              </div>
              <h3 className="upload-prompt">Upload Satellite Imagery</h3>
              <p className="upload-instruction">
                <span>Drag & Drop files here</span>
                <span className="upload-or">or</span>
                <span className="browse-link text-cyan">Browse Files</span>
              </p>
            </div>
          ) : (
            <div className="dropzone-content populated-state">
              <div className="file-gallery">
                {files.map((f, index) => (
                  <div key={f.id} className="file-preview-card glass-panel">
                    <button 
                      className="remove-file-btn" 
                      onClick={(e) => { e.stopPropagation(); removeFile(f.id); }}
                      title="Remove file"
                      aria-label="Remove file"
                    >
                      <X size={16} />
                    </button>
                    <div className="preview-image-wrapper">
                      <img src={f.preview} alt={f.name} className="preview-image" />
                    </div>
                    <div className="file-info">
                      <span className="file-name" title={f.name}>Image {index + 1}: {f.name}</span>
                      <span className="file-size text-muted">{f.size}</span>
                    </div>
                  </div>
                ))}
                
                {/* Add more button */}
                <div className="add-more-card glass-panel" onClick={triggerFileInput}>
                  <UploadCloud size={24} className="text-cyan mb-2" />
                  <span>Add More</span>
                </div>
              </div>
              
              <div className="analyze-action-bar">
                <button className="btn-analyze" onClick={handleAnalyze} disabled={isAnalyzing || ((selectedModel === 'single_image' || selectedModel === 'auto') && question.trim().length < 3)}>
                  <Play size={18} /> Analyze Imagery
                </button>
              </div>
            </div>
          )}
          
          <input 
            type="file" 
            aria-label="Upload images"
            ref={fileInputRef} 
            onChange={handleFileSelect} 
            multiple 
            accept="image/*" 
            style={{ display: 'none' }} 
          />
        </div>
      </div>

      <div className="analysis-info-items">
        <div className="info-item glass-panel">
          <ImageIcon size={20} className="info-icon text-cyan" />
          <div className="info-text">
            <h4>Single Image</h4>
            <p>Single-Image Understanding</p>
          </div>
        </div>
        
        <div className="info-item glass-panel">
          <Layers size={20} className="info-icon text-cyan" />
          <div className="info-text">
            <h4>Temporal Pair</h4>
            <p>Bi-Temporal Change Analysis</p>
          </div>
        </div>
        
        <div className="info-item glass-panel">
          <Map size={20} className="info-icon text-cyan" />
          <div className="info-text">
            <h4>Optical + SAR Pair</h4>
            <p>Optical–SAR Analysis</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default UploadWorkspace;
