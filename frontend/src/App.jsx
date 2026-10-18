import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home/Home';
import SingleAnalysis from './pages/Analysis/SingleAnalysis';
import ChangeAnalysisWorkspace from './pages/Analysis/ChangeAnalysisWorkspace';
import OpticalSarAnalysis from './pages/Analysis/OpticalSarAnalysis';
import MapWorkspace from './pages/Map/MapWorkspace';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/analyze/single" element={<SingleAnalysis />} />
      <Route path="/analyze/change" element={<ChangeAnalysisWorkspace />} />
      <Route path="/analyze/optical-sar" element={<OpticalSarAnalysis />} />
      <Route path="/map" element={<MapWorkspace />} />
    </Routes>
  );
}

export default App;
