import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true, // Enable React strict mode if desired
  eslint: {
    ignoreDuringBuilds: true, // Disable eslint warnings during builds
  },
  typescript: {
    ignoreBuildErrors: true, // Disable typescript errors during builds
  },
  webpack(config, { dev }) {
    if (dev) {
      // Disabling the error overlay in development
      config.devServer = {
        ...config.devServer,
        overlay: false, // Disable the error overlay
      };
    }
    return config;
  },
};

export default nextConfig;
