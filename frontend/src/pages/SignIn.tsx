import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import "./signin.css";

export default function SignIn() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
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
        NEXUS DISASTER INTELLIGENCE
        <span className="status-line" />
        SECURE LOGIN
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

          <p>Disaster Intelligence & Early Warning Platform</p>
        </div>

        {/* Login panel */}
        <section className="signin-card">
          <div className="card-top">
            <div>
              <span className="card-label">ACCOUNT ACCESS</span>
              <h2>Sign In</h2>
            </div>

            <div className="access-code">
              <span>SYSTEM</span>
              <strong>ONLINE</strong>
            </div>
          </div>

          <div className="card-divider" />

          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <label htmlFor="email">EMAIL ADDRESS</label>

              <div className="input-wrapper">
                <span className="input-prefix">@</span>

                <input
                  id="email"
                  type="email"
                  placeholder="name@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="input-group">
              <label htmlFor="password">PASSWORD</label>

              <div className="input-wrapper">
                <span className="input-prefix">◆</span>

                <input
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="signin-options">
              <label className="remember-checkbox">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                />
                <span>Remember me</span>
              </label>

              <button
                type="button"
                className="forgot-password"
                onClick={() => alert("Password reset: Please contact your system administrator.")}
              >
                Forgot password?
              </button>
            </div>

            <button className="signin-button" type="submit">
              <span>SIGN IN</span>
              <span className="button-arrow">→</span>
            </button>
          </form>

          <div className="card-footer">
            <span>
              <i />
              ENCRYPTED CONNECTION
            </span>

            <button
              type="button"
              className="back-home-link"
              onClick={() => navigate("/")}
            >
              Back to Home
            </button>
          </div>
        </section>

        {/* Bottom metadata */}
        <div className="signin-meta">
          <span>FLOOD PREDICTION & SATELLITE MONITORING</span>
          <span>NEXUS PLATFORM</span>
        </div>
      </div>
    </main>
  );
}