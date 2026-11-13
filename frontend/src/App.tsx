import { useEffect, useRef, useState } from "react";
import { BrowserRouter, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { SessionProvider, useSession } from "./state/session";
import Home from "./pages/Home";
import VoiceCheck from "./pages/VoiceCheck";
import Questionnaire from "./pages/Questionnaire";
import Results from "./pages/Results";
import About from "./pages/About";
import Privacy from "./pages/Privacy";

const NAV = [
  { to: "/", label: "Home", end: true },
  { to: "/voice", label: "Voice check" },
  { to: "/questionnaire", label: "Questionnaire" },
  { to: "/results", label: "Results" },
  { to: "/about", label: "How it works" },
  { to: "/privacy", label: "Privacy" },
];

/** Clears all results the moment the user navigates away from /results. */
function RouteWatcher() {
  const { pathname } = useLocation();
  const { clearAll } = useSession();
  const prev = useRef(pathname);
  useEffect(() => {
    if (prev.current === "/results" && pathname !== "/results") clearAll();
    prev.current = pathname;
    window.scrollTo(0, 0);
  }, [pathname, clearAll]);
  return null;
}

function Header() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => setOpen(false), [pathname]);
  const link = ({ isActive }: { isActive: boolean }) =>
    `block rounded-md px-3 py-2 text-sm font-medium ${isActive ? "bg-teal-50 text-teal-800" : "text-ink-soft hover:bg-sand-100"}`;
  return (
    <header className="border-b border-line bg-paper/95">
      <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
        <NavLink to="/" className="text-lg font-semibold text-ink">
          Voice &amp; Wellbeing Check
        </NavLink>
        <button
          type="button"
          className="btn-ghost px-3 py-1.5 md:hidden"
          aria-expanded={open}
          aria-controls="main-nav"
          onClick={() => setOpen((o) => !o)}
        >
          Menu
        </button>
        <nav aria-label="Main" className="hidden md:block">
          <ul className="flex gap-1">
            {NAV.map((n) => (
              <li key={n.to}>
                <NavLink to={n.to} end={n.end} className={link}>
                  {n.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </div>
      {open && (
        <nav id="main-nav" aria-label="Main" className="border-t border-line px-4 py-2 md:hidden">
          <ul>
            {NAV.map((n) => (
              <li key={n.to}>
                <NavLink to={n.to} end={n.end} className={link}>
                  {n.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </header>
  );
}

export default function App() {
  return (
    <SessionProvider>
      <BrowserRouter>
        <RouteWatcher />
        <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:rounded focus:bg-card focus:px-3 focus:py-2">
          Skip to content
        </a>
        <Header />
        <main id="main" className="mx-auto max-w-4xl px-4 py-8 sm:py-10">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/voice" element={<VoiceCheck />} />
            <Route path="/questionnaire" element={<Questionnaire />} />
            <Route path="/results" element={<Results />} />
            <Route path="/about" element={<About />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </main>
        <footer className="mx-auto max-w-4xl px-4 pb-10 text-sm text-ink-muted">
          A learning prototype. Not a medical device and not a diagnosis. Nothing you record or answer is stored.
        </footer>
      </BrowserRouter>
    </SessionProvider>
  );
}
