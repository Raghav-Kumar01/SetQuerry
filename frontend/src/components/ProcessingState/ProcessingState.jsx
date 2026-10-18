import React from 'react';
import { Loader2 } from 'lucide-react';
import './ProcessingState.css';

const ProcessingState = ({ modelName = 'AI Model' }) => (
  <div className="processing-state-overlay glass-panel" role="status" aria-live="polite">
    <div className="processing-content">
      <div className="processing-spinner"><Loader2 size={48} className="text-cyan animate-spin" /></div>
      <h2 className="processing-title">Running Analysis</h2>
      <p className="processing-status-text">Waiting for {modelName} inference...</p>
    </div>
  </div>
);

export default ProcessingState;
