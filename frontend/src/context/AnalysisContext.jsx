import React, { createContext, useState, useContext } from 'react';

const AnalysisContext = createContext();

// eslint-disable-next-line react-refresh/only-export-components
export const useAnalysis = () => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};

export const AnalysisProvider = ({ children }) => {
  // Store uploaded files and analysis metadata globally
  const [files, setFiles] = useState([]);
  const [analysisResult, setAnalysisResult] = useState(null);

  const resetAnalysis = () => {
    // Revoke object URLs to avoid memory leaks
    files.forEach(f => {
      if (f.preview) URL.revokeObjectURL(f.preview);
    });
    setFiles([]);
    setAnalysisResult(null);
  };

  return (
    <AnalysisContext.Provider value={{
      files,
      setFiles,
      analysisResult,
      setAnalysisResult,
      resetAnalysis
    }}>
      {children}
    </AnalysisContext.Provider>
  );
};
