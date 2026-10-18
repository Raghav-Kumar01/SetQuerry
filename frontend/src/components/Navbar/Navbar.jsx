import React from 'react';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar glass-panel">
      <div className="navbar-logo">
        <span className="text-cyan font-bold tracking-widest">SETQUERY</span>
      </div>
      <div className="navbar-links">
        <a href="/#explore">Explore</a>
        <a href="/#upload">Analyze</a>
        <a href="/#about">About</a>
      </div>
    </nav>
  );
};

export default Navbar;
