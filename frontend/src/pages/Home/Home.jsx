import React from 'react';
import Navbar from '../../components/Navbar/Navbar';
import ScrollIntro from '../../components/ScrollIntro/ScrollIntro';
import UploadWorkspace from '../../components/UploadWorkspace/UploadWorkspace';
import EarthMap from '../../components/EarthMap/EarthMap';
import RegionPanel from '../../components/RegionPanel/RegionPanel';
import ChangeAnalysis from '../../components/ChangeAnalysis/ChangeAnalysis';
import AIResults from '../../components/AIResults/AIResults';
import BeforeAfter from '../../components/BeforeAfter/BeforeAfter';
import Footer from '../../components/Footer/Footer';
import './Home.css';

/**
 * Page order:
 *  1. Navbar            (fixed, overlaid — z-index 100)
 *  2. ScrollIntro       (pinned 100vh hero, scroll-driven video)
 *  3. UploadWorkspace   (main file dropzone)
 *  4. Explore / EarthMap
 *  5. ChangeAnalysis
 *  6. BeforeAfter
 *  7. AIResults
 *  8. Footer
 */
const Home = () => {
  return (
    <div className="home-container">

      {/* Fixed transparent navbar — sits above the cinematic video */}
      <Navbar />

      <main>
        {/* ── 1. Cinematic scroll-driven intro ── */}
        <ScrollIntro />

        {/* ── 2. Main Upload Workspace ── */}
        <UploadWorkspace />

        {/* ── 3. Interactive Earth / Map ── */}
        <section className="explore-section section-container" id="explore">
          <div className="explore-header">
            <h2 className="section-title text-cyan">EXPLORE THE EARTH</h2>
          </div>
          <div className="explore-content">
            <EarthMap />
            <RegionPanel />
          </div>
        </section>

        {/* ── 3. Change Analysis ── */}
        <ChangeAnalysis />

        {/* ── 4. Before / After visual comparison ── */}
        <BeforeAfter />

        {/* ── 5. AI Analysis results ── */}
        <AIResults />
      </main>

      <Footer />
    </div>
  );
};

export default Home;
