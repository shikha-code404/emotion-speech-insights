import { defineConfig, devices } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const repo = path.resolve(__dirname, "..");
// Fake microphone input: an acted speech clip from the SAVEE corpus (looped by Chromium).
const FAKE_MIC_WAV = process.env.E2E_FAKE_MIC_WAV ?? path.join(repo, "ml/data/raw/savee/AudioData/AudioData/DC/a01.wav");
const FFMPEG_DIR =
  process.env.FFMPEG_DIR ??
  path.join(
    process.env.LOCALAPPDATA ?? "",
    "Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin",
  );
const python = path.join(repo, process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python");
const pathSep = process.platform === "win32" ? ";" : ":";

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 15_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          args: [
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            `--use-file-for-fake-audio-capture=${FAKE_MIC_WAV}`,
            "--autoplay-policy=no-user-gesture-required",
          ],
        },
      },
    },
  ],
  webServer: [
    {
      command: `"${python}" -m uvicorn backend.app.main:app --port 8000`,
      cwd: repo,
      url: "http://localhost:8000/api/health",
      reuseExistingServer: true,
      timeout: 120_000,
      env: { PATH: fs.existsSync(FFMPEG_DIR) ? `${FFMPEG_DIR}${pathSep}${process.env.PATH}` : (process.env.PATH ?? "") },
    },
    {
      command: "npm run dev",
      url: "http://localhost:5173",
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
});
