import React, { useState, useRef, useCallback, useEffect } from 'react';
import './ComparisonSlider.css';

/**
 * ComparisonSlider — Draggable before/after image comparison.
 *
 * Props:
 *   beforeSrc   – URL for the "before" image
 *   afterSrc    – URL for the "after" image
 *   beforeLabel – label text (default "BEFORE")
 *   afterLabel  – label text (default "AFTER")
 *   beforeDate  – date string for before image
 *   afterDate   – date string for after image
 */
const ComparisonSlider = ({
  beforeSrc,
  afterSrc,
  beforeLabel = 'BEFORE',
  afterLabel = 'AFTER',
  beforeDate = '',
  afterDate = '',
}) => {
  const containerRef = useRef(null);
  const [position, setPosition] = useState(50); // percentage 0–100
  const [isDragging, setIsDragging] = useState(false);

  const updatePosition = useCallback((clientX) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setPosition(pct);
  }, []);

  // Mouse handlers
  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
    updatePosition(e.clientX);
  };

  const handleMouseMove = useCallback((e) => {
    if (!isDragging) return;
    updatePosition(e.clientX);
  }, [isDragging, updatePosition]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Touch handlers
  const handleTouchStart = (e) => {
    setIsDragging(true);
    updatePosition(e.touches[0].clientX);
  };

  const handleTouchMove = useCallback((e) => {
    if (!isDragging) return;
    e.preventDefault();
    updatePosition(e.touches[0].clientX);
  }, [isDragging, updatePosition]);

  const handleTouchEnd = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Keyboard accessibility
  const handleKeyDown = (e) => {
    if (e.key === 'ArrowLeft') {
      setPosition(prev => Math.max(0, prev - 2));
    } else if (e.key === 'ArrowRight') {
      setPosition(prev => Math.min(100, prev + 2));
    }
  };

  // Global mouse/touch listeners while dragging
  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove, { passive: false });
      window.addEventListener('touchend', handleTouchEnd);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleTouchEnd);
    };
  }, [isDragging, handleMouseMove, handleMouseUp, handleTouchMove, handleTouchEnd]);

  return (
    <div
      ref={containerRef}
      className={`cs-container ${isDragging ? 'dragging' : ''}`}
      onMouseDown={handleMouseDown}
      onTouchStart={handleTouchStart}
      role="slider"
      aria-label="Before and after image comparison slider"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(position)}
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      {/* AFTER image — full width, sits behind */}
      <div className="cs-layer cs-after">
        <img src={afterSrc} alt="After" draggable={false} />
        <span className="cs-label cs-label--after">
          {afterLabel}
          {afterDate && <span className="cs-date">{afterDate}</span>}
        </span>
      </div>

      {/* BEFORE image — clipped by slider position */}
      <div
        className="cs-layer cs-before"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      >
        <img src={beforeSrc} alt="Before" draggable={false} />
        <span className="cs-label cs-label--before">
          {beforeLabel}
          {beforeDate && <span className="cs-date">{beforeDate}</span>}
        </span>
      </div>

      {/* Slider handle */}
      <div className="cs-handle" style={{ left: `${position}%` }}>
        <div className="cs-handle-line" />
        <div className="cs-handle-grip">
          <svg width="12" height="24" viewBox="0 0 12 24" fill="none">
            <path d="M4 8L1 12L4 16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M8 8L11 12L8 16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>
    </div>
  );
};

export default ComparisonSlider;
