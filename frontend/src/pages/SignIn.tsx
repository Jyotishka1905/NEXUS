import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import "./signin.css";

export default function SignIn() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();

    // Temporary frontend authentication
    // FastAPI authentication will be connected later.
    navigate("/dashboard");
  };

  return (
    <main className="signin-page">
      {/* Background grid */}
      <div className="signin-grid" />

      {/* Ambient glow */}
      <div className="signin-glow signin-glow-one" />
      <div className="signin-glow signin-glow-two" />

      {/* Top status bar */}
      <div className="signin-status">
        <span className="status-indicator" />
        NEXUS RESEARCH NETWORK
        <span className="status-line" />
        SECURE ACCESS
      </div>

      <div className="signin-container">
        {/* Brand */}
        <div className="signin-brand">
          <div className="brand-mark">
            <span />
            <span />
            <span />
          </div>

          <h1>NEXUS</h1>

          <p>Multimodal Disaster Intelligence Platform</p>
        </div>

        {/* Login panel */}
        <section className="signin-card">
          <div className="card-top">
            <div>
              <span className="card-label">AUTHENTICATION</span>
              <h2>Research Access</h2>
            </div>

            <div className="access-code">
              <span>NODE</span>
              <strong>NX-01</strong>
            </div>
          </div>

          <div className="card-divider" />

          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <label htmlFor="email">RESEARCHER EMAIL</label>

              <div className="input-wrapper">
                <span className="input-prefix">@</span>

                <input
                  id="email"
                  type="email"
                  placeholder="researcher@nexus.ai"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="input-group">
              <label htmlFor="password">ACCESS KEY</label>

              <div className="input-wrapper">
                <span className="input-prefix">◆</span>

                <input
                  id="password"
                  type="password"
                  placeholder="Enter access key"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>

            <button className="signin-button" type="submit">
              <span>INITIALIZE SESSION</span>
              <span className="button-arrow">→</span>
            </button>
          </form>

          <div className="card-footer">
            <span>
              <i />
              ENCRYPTED CONNECTION
            </span>

            <span>PROTOCOL / NEXUS-AUTH</span>
          </div>
        </section>

        {/* Research notice */}
        <div className="signin-notice">
          <div className="notice-icon">i</div>

          <div>
            <strong>RESEARCH ENVIRONMENT</strong>
            <p>
              Authorized access to the NEXUS multimodal disaster
              intelligence research environment.
            </p>
          </div>
        </div>

        {/* Bottom metadata */}
        <div className="signin-meta">
          <span>FLOOD INTELLIGENCE / SATELLITE + ENVIRONMENTAL DATA</span>
          <span>v0.1.0 / RESEARCH BUILD</span>
        </div>
      </div>
    </main>
  );
}