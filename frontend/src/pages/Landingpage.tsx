import { useNavigate } from "react-router-dom";
import NexusScene from "../components/NexusScene";
import "../styles/landing.css";

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <main className="nexus-landing">

      {/* 3D WORLD */}
      <div className="nexus-3d">
        <NexusScene />
      </div>

      {/* SPACE OVERLAY */}
      <div className="nexus-space-overlay" />

      {/* NAVIGATION */}
      <header className="nexus-nav">

        <div className="nexus-logo">
          NE<span>X</span>US
        </div>

        <nav>
          <a className="active">Home</a>
          <a>About</a>
          <a>Research</a>
          <a>Contact</a>
        </nav>

        <button
          className="nav-signin"
          onClick={() => navigate("/signin")}
        >
          <span>♙</span>
          Sign In
        </button>

      </header>

      {/* LEFT TOP LABEL */}
      <div className="nexus-label">
        EARTH&nbsp;&nbsp; / &nbsp;&nbsp;SATELLITES&nbsp;&nbsp; / &nbsp;&nbsp;AI&nbsp;&nbsp; / &nbsp;&nbsp;SAFER TOMORROWS
      </div>

      {/* MAIN CONTENT */}
      <section className="nexus-center">

        <h1>NEXUS</h1>

        <h2>
          MULTIMODAL DISASTER INTELLIGENCE
        </h2>

        <p>
          Connecting satellite vision and environmental intelligence
          <br />
          to understand disasters.
        </p>

        <div className="nexus-buttons">

          <button
            className="primary-button"
            onClick={() => navigate("/signin")}
          >
            SIGN IN
            <span>→</span>
          </button>

          <button className="secondary-button">
            EXPLORE RESEARCH
            <span>→</span>
          </button>

        </div>

      </section>

      {/* LEFT HUD */}
      <div className="hud-panel hud-left">

        <div className="hud-icon">
          ◉
        </div>

        <div>
          <strong>REAL-TIME SATELLITE FEED</strong>
          <small>ELEVATION 408 KM</small>
          <small>LAT 28.6139°</small>
          <small>LON 77.2090°</small>

          <div className="progress">
            <span />
          </div>
        </div>

      </div>

      {/* RIGHT HUD */}
      <div className="hud-panel hud-right">

        <strong>EARTH MONITORING</strong>

        <div>WEATHER ✓</div>
        <div>LAND USE ✓</div>
        <div>DISASTERS ✓</div>
        <div>ANALYSIS ✓</div>

      </div>

      {/* LOWER LEFT */}
      <div className="hud-panel hud-analysis">

        <div className="wave">
          ∿∿∿∿
        </div>

        <div>
          <strong>AI ANALYSIS</strong>
          <small>DETECTING ANOMALIES...</small>
          <small>PROCESSING MULTIMODAL DATA...</small>

          <div className="progress">
            <span />
          </div>
        </div>

      </div>

      {/* LOWER RIGHT */}
      <div className="global-alert">

        <span className="pulse" />

        <div>
          <strong>GLOBAL ALERTS</strong>
          <small>ACTIVE MONITORING</small>
        </div>

      </div>

      {/* SCROLL */}
      <div className="scroll-indicator">

        <div className="scroll-circle">
          ↓
        </div>

        <span>SCROLL TO EXPLORE</span>

      </div>

    </main>
  );
}