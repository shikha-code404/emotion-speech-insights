import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command, mode }) => {
  if (command === "build" && mode === "production") {
    // The API URL is baked in at build time; refuse to ship a build that would call the wrong host.
    const url = loadEnv(mode, process.cwd(), "VITE_")["VITE_API_URL"] ?? "";
    if (!/^https:\/\/[^/]+[^/]$/.test(url)) {
      throw new Error(
        `VITE_API_URL must be an https URL with no trailing slash for production builds (got "${url}"). ` +
          "Set it in the Vercel project's Environment Variables.",
      );
    }
  }
  return {
    plugins: [react()],
    server: { port: 5173, strictPort: true },
  };
});
