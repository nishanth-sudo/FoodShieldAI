import type { AppProps } from "next/app";
import { useEffect } from "react";
import { SessionProvider } from "next-auth/react";
import { AuthProvider } from "context/AuthContext";
import { ThemeProvider } from "context/ThemeContext";
import { LanguageProvider } from "context/LanguageContext";
import { InspectionProvider } from "context/InspectionContext";
import { Layout } from "components/Layout";
import "styles/globals.css";

export default function App({ Component, pageProps: { session, ...pageProps } }: AppProps) {
  useEffect(() => {
    if ("serviceWorker" in navigator) {
      window.addEventListener("load", () => {
        navigator.serviceWorker
          .register("/sw.js")
          .then((reg) => console.log("Service Worker registered successfully: ", reg.scope))
          .catch((err) => console.error("Service Worker registration failed: ", err));
      });
    }
  }, []);

  return (
    <SessionProvider session={session}>
      <AuthProvider>
        <ThemeProvider>
          <LanguageProvider>
            <InspectionProvider>
              <Layout>
                <Component {...pageProps} />
              </Layout>
            </InspectionProvider>
          </LanguageProvider>
        </ThemeProvider>
      </AuthProvider>
    </SessionProvider>
  );
}
