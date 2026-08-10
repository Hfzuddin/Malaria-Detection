const { useState } = React;

// --- ApplicationLogo ---
// function ApplicationLogo({ className }) {
//   return (
//     <div className={`flex items-center justify-center font-bold text-white bg-indigo-600 rounded-md shadow-sm ${className || 'h-8 w-8 text-sm'}`}>
//       MS
//     </div>
//   );
// }

// --- AuthenticatedLayout (Ported exactly from JobTrack) ---
function AuthenticatedLayout({ currentPath, theme, toggleTheme, children }) {
  const [isDesktopSidebarOpen, setIsDesktopSidebarOpen] = useState(true);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // --- Custom Router ---
  const navigate = (path) => {
    window.history.pushState({}, '', path);
    const navEvent = new PopStateEvent('popstate');
    window.dispatchEvent(navEvent);
  };

  const navigation = [
    {
      name: 'Upload Scan',
      href: '/',
      current: currentPath === '/',
      icon: (
        <svg className="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      ),
    },
    {
      name: 'Scanned Results',
      href: '/history',
      current: currentPath === '/history' || currentPath === '/result',
      icon: (
        <svg className="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
        </svg>
      ),
    },
    {
      name: 'Model Analytics',
      href: '/gallery',
      current: currentPath === '/gallery',
      icon: (
        <svg className="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      )
    },
  ];

  const renderNavItems = (isDesktop) => (
    <ul role="list" className="-mx-2 space-y-1">
      {navigation.map((item) => (
        <li key={item.name}>
          <a
            href={item.href}
            onClick={(e) => { e.preventDefault(); navigate(item.href); setIsMobileMenuOpen(false); }}
            className={`group flex w-full items-center gap-x-3 rounded-md p-2 text-sm leading-6 font-semibold transition-colors
                ${item.current ? 'bg-gray-50 text-indigo-600 dark:bg-zinc-800 dark:text-indigo-400' : 'text-gray-700 hover:text-indigo-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-zinc-800'}
                ${isDesktop && !isDesktopSidebarOpen ? 'justify-center' : ''}
            `}
            title={isDesktop && !isDesktopSidebarOpen ? item.name : ''}
          >
            {item.icon}
            <span className={`overflow-hidden whitespace-nowrap transition-all duration-300 ${!isDesktop || isDesktopSidebarOpen ? 'w-auto opacity-100' : 'w-0 opacity-0'}`}>
              {item.name}
            </span>
          </a>
        </li>
      ))}
    </ul>
  );

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50 text-gray-900 dark:bg-zinc-950 dark:text-white transition-colors duration-200">
      {/* Mobile Sidebar Overlay */}
      {isMobileMenuOpen && (
        <div className="relative z-50 md:hidden">
          <div className="fixed inset-0 bg-gray-900/80 transition-opacity" onClick={() => setIsMobileMenuOpen(false)} />
          <div className="fixed inset-0 flex">
            <div className="relative mr-16 flex w-full max-w-xs flex-1 transform transition ease-in-out duration-300">
              <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-white dark:bg-zinc-900 px-6 pb-4 shadow-xl">
                <div className="flex h-16 shrink-0 items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="font-bold text-indigo-600 dark:text-indigo-400">Malaria System</span>
                  </div>
                  <button type="button" className="-m-2.5 p-2.5 text-gray-700 dark:text-gray-200" onClick={() => setIsMobileMenuOpen(false)}>
                    <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                  </button>
                </div>
                <nav className="flex flex-1 flex-col">{renderNavItems(false)}</nav>
                <button onClick={toggleTheme} className="flex items-center gap-x-3 rounded-md p-2 text-sm font-semibold leading-6 text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-zinc-800">
                  {theme === 'light' ? (
                    <><svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" /></svg> Dark Mode</>
                  ) : (
                    <><svg className="w-6 h-6 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" /></svg> Light Mode</>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Desktop Static Sidebar */}
      <div className={`hidden md:flex md:flex-col ${isDesktopSidebarOpen ? 'w-64' : 'w-20'} transition-all duration-300 ease-in-out border-r border-gray-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 z-10 shadow-xl`}>
        <div className="flex h-16 shrink-0 items-center justify-between px-4 border-b border-gray-200 dark:border-zinc-800">
          <a href="/" onClick={(e) => { e.preventDefault(); navigate('/'); }} className={`flex items-center gap-3 overflow-hidden whitespace-nowrap ${isDesktopSidebarOpen ? 'opacity-100' : 'w-0 opacity-0'} transition-all duration-300`}>
            <span className="font-bold text-indigo-600 dark:text-indigo-400">Malaria System</span>
          </a>
          <button onClick={() => setIsDesktopSidebarOpen(!isDesktopSidebarOpen)} className={`p-1.5 rounded-md text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-zinc-800 focus:outline-none flex-shrink-0 ${!isDesktopSidebarOpen ? 'mx-auto' : ''}`}>
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>
          </button>
        </div>

        <nav className="flex flex-1 flex-col gap-y-7 px-3 py-4 overflow-y-auto overflow-x-hidden">
          {renderNavItems(true)}
        </nav>

        {/* Desktop Footer (Theme) */}
        <div className="border-t border-gray-200 dark:border-zinc-800 p-3 space-y-4">
          <button onClick={toggleTheme} className={`flex items-center rounded-lg border border-gray-200 dark:border-zinc-700 bg-gray-50 dark:bg-zinc-800 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-zinc-700 focus:outline-none transition-colors w-full ${!isDesktopSidebarOpen ? 'justify-center' : ''}`} title={!isDesktopSidebarOpen ? "Toggle Theme" : ''}>
            {theme === 'light' ? (
              <><svg className={`w-5 h-5 flex-shrink-0 ${isDesktopSidebarOpen ? 'mr-3' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" /></svg><span className={`overflow-hidden whitespace-nowrap transition-all duration-300 ${isDesktopSidebarOpen ? 'w-auto opacity-100' : 'w-0 opacity-0'}`}>Dark Mode</span></>
            ) : (
              <><svg className={`w-5 h-5 flex-shrink-0 text-yellow-400 ${isDesktopSidebarOpen ? 'mr-3' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" /></svg><span className={`overflow-hidden whitespace-nowrap transition-all duration-300 ${isDesktopSidebarOpen ? 'w-auto opacity-100' : 'w-0 opacity-0'}`}>Light Mode</span></>
            )}
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Unified Top Bar */}
        <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center justify-between border-b border-gray-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 px-4 shadow-sm sm:px-6 lg:px-8 transition-colors duration-200">
          <div className="flex items-center md:hidden">
            <button type="button" className="-m-2.5 p-2.5 text-gray-700 dark:text-gray-200" onClick={() => setIsMobileMenuOpen(true)}>
              <span className="sr-only">Open sidebar</span>
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" /></svg>
            </button>
            <span className="ml-4 font-bold text-indigo-600 dark:text-indigo-400">Malaria System</span>
          </div>
          <div className="hidden md:flex flex-1 max-w-lg">
          </div>
          <div className="flex items-center gap-x-4 ml-4">
            <span className="text-sm font-medium text-gray-600 dark:text-gray-400 hidden sm:block">User</span>
          </div>
        </div>

        <main className="flex-1 overflow-y-auto">
          <div className="p-4 sm:p-6 lg:p-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
