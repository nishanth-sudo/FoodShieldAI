import { useAuth } from "context/AuthContext";
import { useLanguage } from "context/LanguageContext";
import Link from "next/link";

export default function HomePage() {
  const { user } = useAuth();
  const { t } = useLanguage();

  if (!user) {
    return (
      <div className="text-center py-20 animate-fadeIn">
        <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 dark:text-white tracking-tight mb-4">
          {t.appName}
        </h1>
        <p className="text-lg md:text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto mb-10">
          {t.tagline}
        </p>
        <Link
          href="/login"
          className="inline-block bg-primary-600 text-white px-8 py-3.5 rounded-lg font-semibold shadow-lg hover:bg-primary-700 hover:shadow-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
          aria-label={t.getStarted}
        >
          {t.getStarted}
        </Link>
      </div>
    );
  }

  return (
    <div className="animate-fadeIn">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          {t.welcome}, {user.name}
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">{t.uploadToStart}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Link
          href="/inspect"
          className="block group bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 hover:shadow-md hover:border-primary-500 dark:hover:border-primary-500 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500"
          aria-label={`${t.newInspection}: ${t.newInspectionDesc}`}
        >
          <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-200">📷</div>
          <h2 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors duration-200">
            {t.newInspection}
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2 leading-relaxed">
            {t.newInspectionDesc}
          </p>
        </Link>

        <Link
          href="/history"
          className="block group bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 hover:shadow-md hover:border-primary-500 dark:hover:border-primary-500 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500"
          aria-label={`${t.history}: ${t.historyDesc}`}
        >
          <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-200">📋</div>
          <h2 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors duration-200">
            {t.history}
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2 leading-relaxed">
            {t.historyDesc}
          </p>
        </Link>

        {user.role === "admin" && (
          <Link
            href="/admin"
            className="block group bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 hover:shadow-md hover:border-primary-500 dark:hover:border-primary-500 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500"
            aria-label={`${t.admin}: ${t.adminDesc}`}
          >
            <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-200">⚙️</div>
            <h2 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors duration-200">
              {t.admin}
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-2 leading-relaxed">
              {t.adminDesc}
            </p>
          </Link>
        )}
      </div>
    </div>
  );
}
