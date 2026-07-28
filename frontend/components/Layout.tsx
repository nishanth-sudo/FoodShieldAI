import { ReactNode } from "react";
import Link from "next/link";
import { useAuth } from "context/AuthContext";

export function Layout({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth();

  return (
    <div className="min-h-screen flex flex-col">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-8">
              <Link href="/" className="text-xl font-bold text-primary-600">
                FoodShield AI
              </Link>
              {user && (
                <div className="flex gap-4 text-sm">
                  <Link href="/inspect" className="text-gray-600 hover:text-primary-600">Inspect</Link>
                  <Link href="/history" className="text-gray-600 hover:text-primary-600">History</Link>
                  {user.role === "admin" && (
                    <Link href="/admin" className="text-gray-600 hover:text-primary-600">Admin</Link>
                  )}
                </div>
              )}
            </div>
            <div className="flex items-center gap-4">
              {user ? (
                <>
                  <span className="text-sm text-gray-500">{user.name}</span>
                  <button onClick={logout} className="text-sm text-gray-600 hover:text-red-600">
                    Logout
                  </button>
                </>
              ) : (
                <Link href="/login" className="text-sm text-primary-600 hover:text-primary-700">
                  Login
                </Link>
              )}
            </div>
          </div>
        </div>
      </nav>
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {children}
      </main>
    </div>
  );
}
