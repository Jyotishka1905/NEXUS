import { BrowserRouter, Routes, Route } from "react-router-dom";

import LandingPage from "./pages/Landingpage";
import SignIn from "./pages/SignIn";
import Dashboard from "./pages/Dashboard";
import Detection from "./pages/Detection";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/detection" element={<Detection />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;