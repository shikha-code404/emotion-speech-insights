/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#faf7f2",      // warm off-white page
        card: "#fffdf9",
        line: "#e6dfd3",
        ink: { DEFAULT: "#2b2a28", soft: "#55524c", muted: "#77736b" },
        teal: {
          50: "#eef6f5", 100: "#d9ece9", 200: "#b3d6d1", 300: "#86bcb5",
          400: "#5d9f97", 500: "#3f827b", 600: "#2f6b65", 700: "#245550", 800: "#1b403c",
        },
        sand: { 100: "#f3ede3", 200: "#e8dfd0" },
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
      },
    },
  },
  plugins: [],
};
