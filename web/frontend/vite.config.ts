import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Built assets go into the Python package so `simulsi ui` can serve them.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/",
  build: {
    outDir: "../../src/simulsi/web/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8642" },
  },
});
