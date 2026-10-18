import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import './ScrollIntro.css';

gsap.registerPlugin(ScrollTrigger);

export default function ScrollIntro() {
  const heroRef = useRef(null);
  const videoRef = useRef(null);
  const hintRef = useRef(null);
  const barRef = useRef(null);

  const [isLoaded, setIsLoaded] = useState(false);
  const stRef = useRef(null);
  const isInitialized = useRef(false);
  const lastSeek = useRef(-1);
  
  // Clean up any existing ScrollTrigger when the component unmounts
  useEffect(() => {
    return () => {
      if (stRef.current) {
        stRef.current.kill();
        stRef.current = null;
      }
      isInitialized.current = false;
    };
  }, []);

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video || !heroRef.current) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setIsLoaded(true);
      video.pause();
      return;
    }
    
    if (isInitialized.current) return;
    isInitialized.current = true;
    
    const duration = video.duration;
    if (!duration || duration <= 0) return;


    // Initial State: video.currentTime = 0
    video.currentTime = 0;
    video.pause();

    setIsLoaded(true);

    // Prevent double init in dev mode if StrictMode is on
    if (stRef.current) {
      stRef.current.kill();
    }

    // SCROLL CONTROL: The exact architecture requested
    stRef.current = ScrollTrigger.create({
      trigger: heroRef.current,
      start: "top top",
      end: "+=500%",
      pin: true,
      scrub: 1,
      onUpdate: (self) => {
        // SCROLL DOWN/UP: video.currentTime increases/decreases
        const nextTime = self.progress * duration;
        if (Math.abs(nextTime - lastSeek.current) >= 1 / 24) {
          video.currentTime = nextTime;
          lastSeek.current = nextTime;
        }

        // Fade out hint
        if (hintRef.current) {
          hintRef.current.style.opacity = Math.max(0, 1 - self.progress * 15);
        }

        // Update progress bar
        if (barRef.current) {
          barRef.current.style.transform = `scaleY(${self.progress})`;
        }
      }
    });

    // Refresh ScrollTrigger to ensure correct pin spacing
    ScrollTrigger.refresh();
  };

  return (
    <section ref={heroRef} className="sq-intro">
      <video
        ref={videoRef}
        className="sq-intro__video"
        src="/videos/setquery-intro.mp4"
        muted
        playsInline
        preload="metadata"
        controls={false}
        onLoadedMetadata={handleLoadedMetadata}
      />

      {!isLoaded && (
        <div className="sq-intro__loading">
          <span>Loading</span>
        </div>
      )}

      {isLoaded && (
        <div ref={hintRef} className="sq-intro__hint" aria-hidden="true">
          <span className="sq-intro__hint-text">Scroll to explore</span>
          <div className="sq-intro__hint-line" />
        </div>
      )}

      <div className="sq-intro__bar-track" aria-hidden="true">
        <div ref={barRef} className="sq-intro__bar-fill" />
      </div>
    </section>
  );
}
