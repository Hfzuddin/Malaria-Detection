const { useState, useEffect } = React;

// --- App Root ---
function App() {
  const [currentPath, setCurrentPath] = useState(window.location.pathname);
  // Default to dark mode since it's a premium feel, matching JobTrack's default dark setup where configured
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');

  useEffect(() => {
    const onLocationChange = () => setCurrentPath(window.location.pathname);
    window.addEventListener('popstate', onLocationChange);
    return () => window.removeEventListener('popstate', onLocationChange);
  }, []);

  useEffect(() => {
    if (theme === 'dark') document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');

  const renderContent = () => {
    if (currentPath === '/gallery') return <Gallery />;
    if (currentPath === '/history' || currentPath === '/result') return <ScannedResults />;
    return <Dashboard />;
  };

  return (
    <AuthenticatedLayout currentPath={currentPath} theme={theme} toggleTheme={toggleTheme}>
      {renderContent()}
    </AuthenticatedLayout>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
