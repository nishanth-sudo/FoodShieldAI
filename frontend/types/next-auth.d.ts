import "next-auth";

declare module "next-auth" {
  interface Session {
    accessToken: string;
    refreshToken: string;
    user: {
      id: string;
      name: string;
      email: string;
      role: string;
    };
  }

  interface User {
    accessToken: string;
    refreshToken: string;
    role: string;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken: string;
    refreshToken: string;
    accessTokenExpires: number;
    id: string;
    role: string;
  }
}
