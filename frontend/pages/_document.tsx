import { Html, Head, Main, NextScript } from "next/document";

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function () {
              try {
                var saved = localStorage.getItem("theme");
                var theme = saved === "light" || saved === "dark"
                  ? saved
                  : (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
                var root = document.documentElement;
                root.classList.add(theme);
                root.style.colorScheme = theme;
              } catch (e) {}
            })();`,
          }}
        />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
