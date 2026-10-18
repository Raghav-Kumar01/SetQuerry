import React from 'react';
import './Hero.css';

const Hero = () => {
  return (
    <section className="hero-section section-container">
      <div className="hero-content">
        <h1 className="hero-title text-cyan shadow-cyan-text">Earth</h1>
        <div className="hero-indicator">
          <span>↓</span>
        </div>
        <h2 className="hero-subtitle">India</h2>
        <div className="hero-indicator">
          <span>↓</span>
        </div>
        <h2 className="hero-subtitle">Glacier / Flood</h2>
        <div className="hero-indicator">
          <span>↓</span>
        </div>
        <h3 className="hero-description">India + comparison panels</h3>
      </div>
      
      <div className="scroll-indicator">
        <span>Scroll to Explore ↓</span>
      </div>
    </section>
  );
};

export default Hero;
