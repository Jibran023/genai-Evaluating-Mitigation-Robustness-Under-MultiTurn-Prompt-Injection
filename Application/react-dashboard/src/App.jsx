import React, { useState } from 'react';
import Overview from './components/Overview';
import Findings from './components/Findings';
import Dataset from './components/Dataset';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="app-container">
      {/* Hero Header */}
      <div className="hero">
        <div className="hero-chip">Generative AI • 2026</div>
        <h1 className="hero-title">
          Evaluating Mitigation Robustness<br/>
          Under Multi-Turn Prompt Injection
        </h1>
        <p className="hero-sub">
          A systematic empirical study of LLM safety defences against adversarial conversational attacks
        </p>
      </div>

      <div className="hr"></div>

      {/* Tab Navigation */}
      <div className="tabs-nav">
        <button 
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          🎯 Goal · Metrics · Harness
        </button>
        <button 
          className={`tab-btn ${activeTab === 'findings' ? 'active' : ''}`}
          onClick={() => setActiveTab('findings')}
        >
          📊 Findings & Results
        </button>
        <button 
          className={`tab-btn ${activeTab === 'dataset' ? 'active' : ''}`}
          onClick={() => setActiveTab('dataset')}
        >
          🗂️ Our Dataset
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        <ErrorBoundary>
          {activeTab === 'overview' && <Overview />}
          {activeTab === 'findings' && <Findings />}
          {activeTab === 'dataset' && <Dataset />}
        </ErrorBoundary>
      </div>
      
      {/* Footer */}
      <div style={{ textAlign: 'center', padding: '3rem 0 1rem', color: '#cbd5e1', fontSize: '0.8rem' }}>
        Evaluating Mitigation Robustness Under Multi-Turn Prompt Injection · Jibran Shaikh & Syeda Wania Hussain · GenAI Research — 2026
      </div>
    </div>
  );
}

export default App;
