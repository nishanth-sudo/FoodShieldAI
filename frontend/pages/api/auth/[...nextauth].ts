import NextAuth, { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

const API_URL = process.env.API_URL || "http://localhost:8000";

async function refreshAccessToken(token: {
  refreshToken: string;
  accessToken: string;
  accessTokenExpires: number;
  id: string;
  role: string;
}) {
  try {
    const res = await fetch(`${API_URL}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: token.refreshToken }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error("Refresh failed");

    return {
      ...token,
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      accessTokenExpires: Date.now() + 30 * 60 * 1000,
    };
  } catch {
    return { ...token, error: "RefreshAccessTokenError" as const };
  }
}

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) return null;

        const res = await fetch(`${API_URL}/auth/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: credentials.email,
            password: credentials.password,
          }),
        });

        if (!res.ok) return null;

        const data = await res.json();

        const userRes = await fetch(`${API_URL}/auth/me`, {
          headers: { Authorization: `Bearer ${data.access_token}` },
        });
        if (!userRes.ok) return null;

        const user = await userRes.json();

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          role: user.role,
          accessToken: data.access_token,
          refreshToken: data.refresh_token,
        };
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        return {
          accessToken: user.accessToken,
          refreshToken: user.refreshToken,
          accessTokenExpires: Date.now() + 30 * 60 * 1000,
          id: user.id,
          role: user.role,
          name: user.name,
          email: user.email,
        };
      }

      if (Date.now() < token.accessTokenExpires) {
        return token;
      }

      return refreshAccessToken(token);
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken;
      session.refreshToken = token.refreshToken;
      session.user = {
        id: token.id,
        name: token.name ?? "",
        email: token.email ?? "",
        role: token.role,
      };
      return session;
    },
  },
  pages: {
    signIn: "/login",
  },
  session: {
    strategy: "jwt",
    maxAge: 7 * 24 * 60 * 60,
  },
};

export default NextAuth(authOptions);
