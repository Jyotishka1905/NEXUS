import { useNavigate } from "react-router-dom";
import "./dashboard.css";

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="dashboard">

      {/* ================================
          SIDEBAR
      ================================= */}

      <aside className="dashboard-sidebar">

        <div className="dashboard-logo">
          <span className="logo-mark">N</span>

          <div>
            <h1>NEXUS</h1>
            <span>DISASTER INTELLIGENCE</span>
          </div>
        </div>


        <nav className="dashboard-nav">

          <button className="nav-item active">
            <span>⌂</span>
            Overview
          </button>

          <button
            className="nav-item"
            onClick={() => navigate("/detection")}
          >
            <span>◉</span>
            Flood Detection
          </button>

          <button
            className="nav-item"
            onClick={() => navigate("/satellite")}
          >
            <span>◈</span>
            Satellite Data
          </button>

          <button
            className="nav-item"
            onClick={() => navigate("/environment")}
          >
            <span>◇</span>
            Environmental Data
          </button>

          <div className="nav-divider" />

          <button
            className="nav-item"
            onClick={() => navigate("/research")}
          >
            <span>⌬</span>
            Research
          </button>

          <button
            className="nav-item"
            onClick={() => navigate("/experiments")}
          >
            <span>△</span>
            Experiments
          </button>

          <button
            className="nav-item"
            onClick={() => navigate("/results")}
          >
            <span>▣</span>
            Results
          </button>

        </nav>


        <div className="sidebar-bottom">

          <div className="system-status">
            <span className="status-dot" />

            <div>
              <strong>SYSTEM ONLINE</strong>
              <small>NEXUS CORE</small>
            </div>
          </div>

          <button
            className="logout-button"
            onClick={() => navigate("/")}
          >
            Sign Out
          </button>

        </div>

      </aside>


      {/* ================================
          MAIN CONTENT
      ================================= */}

      <main className="dashboard-main">

        {/* HEADER */}

        <header className="dashboard-header">

          <div>
            <div className="eyebrow">
              NEXUS / INTELLIGENCE CENTER
            </div>

            <h2>Research Dashboard</h2>

            <p>
              Multimodal disaster intelligence and flood detection research.
            </p>
          </div>


          <div className="header-status">
            <span className="status-dot" />
            <span>RESEARCH MODE</span>
          </div>

        </header>


        {/* ================================
            OVERVIEW CARDS
        ================================= */}

        <section className="overview-grid">

          <div className="overview-card">

            <div className="card-label">
              ACTIVE STUDY
            </div>

            <div className="card-value">
              FLOOD
            </div>

            <p>
              Satellite-based flood detection
            </p>

            <span className="card-accent">
              NEXUS-01
            </span>

          </div>


          <div className="overview-card">

            <div className="card-label">
              DATA SOURCES
            </div>

            <div className="card-value">
              02
            </div>

            <p>
              Satellite + environmental
            </p>

            <span className="card-accent">
              MULTIMODAL
            </span>

          </div>


          <div className="overview-card">

            <div className="card-label">
              MODEL STATUS
            </div>

            <div className="card-value">
              READY
            </div>

            <p>
              Research pipeline initialized
            </p>

            <span className="card-accent">
              PIPELINE
            </span>

          </div>


          <div className="overview-card">

            <div className="card-label">
              RESEARCH STAGE
            </div>

            <div className="card-value">
              PHASE 01
            </div>

            <p>
              Dataset & baseline preparation
            </p>

            <span className="card-accent">
              ACTIVE
            </span>

          </div>

        </section>


        {/* ================================
            MAIN RESEARCH AREA
        ================================= */}

        <section className="dashboard-content-grid">


          {/* FLOOD DETECTION */}

          <div className="dashboard-panel detection-panel">

            <div className="panel-header">

              <div>
                <span className="panel-kicker">
                  PRIMARY MODULE
                </span>

                <h3>Flood Detection</h3>
              </div>

              <span className="panel-number">
                01
              </span>

            </div>


            <div className="detection-placeholder">

              <div className="radar-circle">
                <div className="radar-line" />
                <span>SCAN</span>
              </div>

              <div className="detection-text">

                <strong>
                  SATELLITE ANALYSIS
                </strong>

                <p>
                  Upload satellite imagery to begin flood
                  detection analysis.
                </p>

              </div>

            </div>


            <button
              className="primary-action"
              onClick={() => navigate("/detection")}
            >
              OPEN DETECTION MODULE
              <span>→</span>
            </button>

          </div>


          {/* RESEARCH QUESTION */}

          <div className="dashboard-panel research-panel">

            <div className="panel-header">

              <div>
                <span className="panel-kicker">
                  RESEARCH QUESTION
                </span>

                <h3>Multimodal Intelligence</h3>
              </div>

              <span className="panel-number">
                02
              </span>

            </div>


            <div className="research-question">

              <span className="quote-mark">
                “
              </span>

              <p>
                Does combining satellite imagery with
                environmental information improve flood
                detection compared with satellite imagery alone?
              </p>

            </div>


            <div className="research-meta">

              <div>
                <span>IMAGE INPUT</span>
                <strong>SATELLITE</strong>
              </div>

              <div>
                <span>CONTEXT</span>
                <strong>ENVIRONMENTAL</strong>
              </div>

              <div>
                <span>OBJECTIVE</span>
                <strong>COMPARISON</strong>
              </div>

            </div>

          </div>

        </section>


        {/* ================================
            PIPELINE
        ================================= */}

        <section className="pipeline-section">

          <div className="section-heading">

            <div>
              <span className="panel-kicker">
                RESEARCH PIPELINE
              </span>

              <h3>Experimental Workflow</h3>
            </div>

            <span className="pipeline-status">
              INITIALIZED
            </span>

          </div>


          <div className="pipeline">

            <div className="pipeline-step active">

              <span className="step-number">
                01
              </span>

              <div>
                <strong>Satellite Data</strong>
                <small>Imagery acquisition</small>
              </div>

            </div>


            <div className="pipeline-connector" />


            <div className="pipeline-step">

              <span className="step-number">
                02
              </span>

              <div>
                <strong>Preprocessing</strong>
                <small>Data preparation</small>
              </div>

            </div>


            <div className="pipeline-connector" />


            <div className="pipeline-step">

              <span className="step-number">
                03
              </span>

              <div>
                <strong>Baseline</strong>
                <small>Image-only model</small>
              </div>

            </div>


            <div className="pipeline-connector" />


            <div className="pipeline-step">

              <span className="step-number">
                04
              </span>

              <div>
                <strong>Multimodal</strong>
                <small>Fusion model</small>
              </div>

            </div>


            <div className="pipeline-connector" />


            <div className="pipeline-step">

              <span className="step-number">
                05
              </span>

              <div>
                <strong>Evaluation</strong>
                <small>Compare results</small>
              </div>

            </div>

          </div>

        </section>


        {/* ================================
            FOOTER
        ================================= */}

        <footer className="dashboard-footer">

          <span>
            NEXUS — Multimodal Disaster Intelligence Platform
          </span>

          <span>
            RESEARCH ENVIRONMENT / v0.1
          </span>

        </footer>

      </main>

    </div>
  );
}