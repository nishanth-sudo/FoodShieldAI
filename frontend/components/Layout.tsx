import { ReactNode, useState } from "react";
import Link from "next/link";
import { useAuth } from "context/AuthContext";
import { useTheme } from "context/ThemeContext";
import { useLanguage } from "context/LanguageContext";
import { useInspection } from "context/InspectionContext";

export function Layout({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const { language, setLanguage, t } = useLanguage();
  const { isOnline } = useInspection();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen flex flex-col bg-gray-50 text-gray-900 transition-colors duration-200 dark:bg-gray-900 dark:text-gray-100">
      {/* Offline Banner */}
      {!isOnline && (
        <div
          role="alert"
          aria-live="assertive"
          className="bg-amber-500 text-white text-center py-2 px-4 text-xs font-semibold tracking-wide shadow-sm"
        >
          ⚠️ {t.offlineWarning}
        </div>
      )}

      <nav className="bg-white shadow-sm border-b dark:bg-gray-800 dark:border-gray-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            {/* Left side brand + links */}
            <div className="flex items-center gap-8">
              <Link
                href="/"
                className="text-xl font-bold text-primary-600 dark:text-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded"
                aria-label={t.appName}
              >
                {t.appName}
              </Link>
              {user && (
                <div className="hidden md:flex gap-4 text-sm font-medium">
                  <Link
                    href="/inspect"
                    className="text-gray-600 hover:text-primary-600 dark:text-gray-300 dark:hover:text-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded px-1"
                  >
                    {t.inspect}
                  </Link>
                  <Link
                    href="/history"
                    className="text-gray-600 hover:text-primary-600 dark:text-gray-300 dark:hover:text-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded px-1"
                  >
                    {t.history}
                  </Link>
                  {user.role === "admin" && (
                    <Link
                      href="/admin"
                      className="text-gray-600 hover:text-primary-600 dark:text-gray-300 dark:hover:text-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded px-1"
                    >
                      {t.admin}
                    </Link>
                  )}
                </div>
              )}
            </div>

            {/* Right side options */}
            <div className="hidden md:flex items-center gap-4">
              {/* Language Switcher */}
              <div className="flex items-center border rounded-lg overflow-hidden dark:border-gray-600 bg-gray-50 dark:bg-gray-700">
                <button
                  onClick={() => setLanguage("en")}
                  className={`px-2 py-1 text-xs font-semibold focus:outline-none focus:ring-1 focus:ring-primary-500 ${
                    language === "en"
                      ? "bg-primary-600 text-white"
                      : "text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-600"
                  }`}
                  aria-label="Switch to English"
                >
                  EN
                </button>
                <button
                  onClick={() => setLanguage("es")}
                  className={`px-2 py-1 text-xs font-semibold focus:outline-none focus:ring-1 focus:ring-primary-500 ${
                    language === "es"
                      ? "bg-primary-600 text-white"
                      : "text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-600"
                  }`}
                  aria-label="Cambiar a Español"
                >
                  ES
                </button>
              </div>

              {/* Theme Toggle */}
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500"
                aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
              >
                {theme === "light" ? "🌙" : "☀️"}
              </button>

              {user ? (
                <div className="flex items-center gap-3">
                  <span className="text-sm text-gray-500 dark:text-gray-400">{user.name}</span>
                  <button
                    onClick={logout}
                    className="text-sm font-medium text-gray-600 hover:text-red-600 dark:text-gray-300 dark:hover:text-red-400 focus:outline-none focus:ring-2 focus:ring-red-500 rounded px-1"
                    aria-label={t.logout}
                  >
                    {t.logout}
                  </button>
                </div>
              ) : (
                <Link
                  href="/login"
                  className="text-sm font-semibold text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded px-1"
                >
                  {t.login}
                </Link>
              )}
            </div>

            {/* Mobile Menu & Action Buttons */}
            <div className="md:hidden flex items-center gap-2">
              <button
                onClick={toggleTheme}
                className="p-1.5 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200"
                aria-label="Toggle theme"
              >
                {theme === "light" ? "🌙" : "☀️"}
              </button>

              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500 text-gray-500 dark:text-gray-400"
                aria-expanded={mobileMenuOpen}
                aria-label="Toggle main menu"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  {mobileMenuOpen ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  )}
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Tray */}
        {mobileMenuOpen && (
          <div className="md:hidden px-2 pt-2 pb-4 space-y-1 bg-white border-t border-gray-100 dark:bg-gray-800 dark:border-gray-700 animate-fadeIn">
            {user && (
              <>
                <Link
                  href="/inspect"
                  onClick={() => setMobileMenuOpen(false)}
                  className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-gray-700"
                >
                  {t.inspect}
                </Link>
                <Link
                  href="/history"
                  onClick={() => setMobileMenuOpen(false)}
                  className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-gray-700"
                >
                  {t.history}
                </Link>
                {user.role === "admin" && (
                  <Link
                    href="/admin"
                    onClick={() => setMobileMenuOpen(false)}
                    className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-gray-700"
                  >
                    {t.admin}
                  </Link>
                )}
              </>
            )}

            <div className="pt-4 pb-2 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between px-3">
              <div className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                Language / Idioma
              </div>
              <div className="flex items-center border rounded-lg overflow-hidden dark:border-gray-600">
                <button
                  onClick={() => {
                    setLanguage("en");
                    setMobileMenuOpen(false);
                  }}
                  className={`px-3 py-1.5 text-xs font-semibold ${
                    language === "en" ? "bg-primary-600 text-white" : "text-gray-600 bg-gray-50 dark:bg-gray-700 dark:text-gray-300"
                  }`}
                >
                  EN
                </button>
                <button
                  onClick={() => {
                    setLanguage("es");
                    setMobileMenuOpen(false);
                  }}
                  className={`px-3 py-1.5 text-xs font-semibold ${
                    language === "es" ? "bg-primary-600 text-white" : "text-gray-600 bg-gray-50 dark:bg-gray-700 dark:text-gray-300"
                  }`}
                >
                  ES
                </button>
              </div>
            </div>

            {user ? (
              <div className="pt-2 border-t border-gray-100 dark:border-gray-700 px-3">
                <div className="text-sm text-gray-500 mb-1 dark:text-gray-400">
                  {user.name} ({user.role})
                </div>
                <button
                  onClick={() => {
                    logout();
                    setMobileMenuOpen(false);
                  }}
                  className="block w-full text-left py-2 font-medium text-red-600 dark:text-red-400"
                >
                  {t.logout}
                </button>
              </div>
            ) : (
              <Link
                href="/login"
                onClick={() => setMobileMenuOpen(false)}
                className="block px-3 py-2 rounded-md text-base font-semibold text-primary-600 dark:text-primary-400"
              >
                {t.login}
              </Link>
            )}
          </div>
        )}
      </nav>

      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {children}
      </main>
    </div>
  );
}
