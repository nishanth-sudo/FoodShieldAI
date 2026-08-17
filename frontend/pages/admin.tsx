import { useEffect, useState } from "react";
import { ProtectedRoute } from "components/ProtectedRoute";
import { adminApi } from "lib/api";
import { LoadingSpinner } from "components/LoadingSpinner";
import { useLanguage } from "context/LanguageContext";
import type { HealthStatus, Inspection, User } from "lib/types";

export default function AdminPage() {
  return (
    <ProtectedRoute>
      <AdminContent />
    </ProtectedRoute>
  );
}

function AdminContent() {
  const [users, setUsers] = useState<User[]>([]);
  const [inspections, setInspections] = useState<Inspection[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const { t } = useLanguage();

  useEffect(() => {
    Promise.all([
      adminApi.users(),
      adminApi.inspections(),
      adminApi.health(),
    ])
      .then(([usersRes, inspectionsRes, healthRes]) => {
        setUsers(usersRes.data);
        setInspections(inspectionsRes.data);
        setHealth(healthRes.data);
      })
      .catch((err) => {
        console.error("Failed to load admin stats", err);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text={t.loading} />;

  return (
    <div className="animate-fadeIn space-y-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
        {t.adminDashboard}
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 flex flex-col justify-between">
          <p className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            {t.totalUsers}
          </p>
          <p className="text-4xl font-extrabold text-gray-900 dark:text-white mt-4">{users.length}</p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 flex flex-col justify-between">
          <p className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            {t.inspections}
          </p>
          <p className="text-4xl font-extrabold text-gray-900 dark:text-white mt-4">{inspections.length}</p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 flex flex-col justify-between">
          <p className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            {t.systemStatus}
          </p>
          <p
            className={`text-2xl font-bold mt-4 px-3 py-1 rounded-lg inline-block text-center ${
              health?.status === "healthy"
                ? "bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                : "bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-400"
            }`}
          >
            {health?.status ? health.status.toUpperCase() : "UNKNOWN"}
          </p>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 overflow-hidden">
        <div className="p-5 border-b border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/50 flex justify-between items-center">
          <h2 className="text-lg font-bold text-gray-900 dark:text-white">
            {t.usersTable}
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-50 dark:bg-gray-700/50 text-gray-500 dark:text-gray-400 uppercase text-xs font-semibold tracking-wider border-b border-gray-100 dark:border-gray-700">
              <tr>
                <th className="px-6 py-4">{t.name}</th>
                <th className="px-6 py-4">{t.email}</th>
                <th className="px-6 py-4">{t.role}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
              {users.map((u) => (
                <tr key={u.id} className="hover:bg-gray-50/80 dark:hover:bg-gray-700/50 transition">
                  <td className="px-6 py-4 font-semibold text-gray-900 dark:text-white">{u.name}</td>
                  <td className="px-6 py-4 text-gray-500 dark:text-gray-400">{u.email}</td>
                  <td className="px-6 py-4">
                    <span className="inline-block px-2.5 py-1 rounded-full text-xs font-semibold bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
                      {u.role}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
