import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./detection.css";

export default function Detection() {
  const navigate = useNavigate();
  const [imageName, setImageName] = useState("");

  const handleImageUpload = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];

    if (file) {
      setImageName(file.name);
    }
  };

  return (
    <div className="detection-page">
      <button
        className="back-button"
        onClick={() => navigate("/dashboard")}
      >
        ← DASHBOARD
      </button>

      <header className="detection-header">
        <div>
          <div className="eyebrow">NEXUS / AI RESEARCH</div>
          <h1>Flood Detection</h1>
          <p>
            Multimodal flood analysis using satellite imagery and
            environmental data.
          </p>
        </div>

        <div className="model-status">
          <span className="status-dot" />
          MODEL STATUS: NOT TRAINED
        </div>
      </header>

      <section className="model-selection">
        <h2>01 / SELECT MODEL</h2>

        <div className="model-grid">
          <div className="model-card active">
            <span className="model-number">01</span>
            <h3>Satellite Only</h3>
            <p>
              Image-based flood detection baseline.
            </p>
          </div>

          <div className="model-card">
            <span className="model-number">02</span>
            <h3>Environmental Only</h3>
            <p>
              Prediction using environmental variables.
            </p>
          </div>

          <div className="model-card">
            <span className="model-number">03</span>
            <h3>Multimodal Fusion</h3>
            <p>
              Satellite imagery combined with environmental data.
            </p>
          </div>
        </div>
      </section>

      <section className="detection-grid">
        <div className="panel">
          <div className="panel-title">
            <span>02 / SATELLITE INPUT</span>
          </div>

          <label className="upload-area">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
            />

            <div className="upload-icon">+</div>

            <strong>
              {imageName || "UPLOAD SATELLITE IMAGE"}
            </strong>

            <span>
              Sentinel / satellite imagery
            </span>
          </label>
        </div>

        <div className="panel">
          <div className="panel-title">
            <span>03 / ENVIRONMENTAL INPUT</span>
          </div>

          <div className="environment-grid">
            <div className="data-field">
              <label>RAINFALL</label>
              <div>— mm</div>
            </div>

            <div className="data-field">
              <label>TEMPERATURE</label>
              <div>— °C</div>
            </div>

            <div className="data-field">
              <label>ELEVATION</label>
              <div>— m</div>
            </div>

            <div className="data-field">
              <label>LOCATION</label>
              <div>—</div>
            </div>
          </div>
        </div>
      </section>

      <section className="prediction-panel">
        <div className="panel-title">
          <span>04 / AI PREDICTION</span>
          <span className="prediction-status">
            AWAITING MODEL
          </span>
        </div>

        <div className="prediction-content">
          <div className="prediction-main">
            <div className="prediction-label">
              PREDICTED CLASS
            </div>

            <div className="prediction-value">
              NOT AVAILABLE
            </div>
          </div>

          <div className="prediction-metric">
            <span>CONFIDENCE</span>
            <strong>—</strong>
          </div>

          <div className="prediction-metric">
            <span>MODEL</span>
            <strong>NOT TRAINED</strong>
          </div>
        </div>
      </section>

      <section className="research-note">
        <div className="note-marker">RESEARCH QUESTION</div>

        <p>
          Does combining satellite imagery with environmental
          information improve flood detection compared with
          satellite imagery alone?
        </p>
      </section>
    </div>
  );
}