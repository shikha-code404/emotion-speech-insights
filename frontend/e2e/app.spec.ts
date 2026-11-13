import { expect, test, type Page } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const repo = path.resolve(__dirname, "../..");
const UPLOAD_CLIP = path.join(repo, "ml/data/raw/savee/AudioData/AudioData/JK/sa05.wav");

// DASS-21 item membership (used only by the tests to build specific answer patterns).
const DEPRESSION = [3, 5, 10, 13, 16, 17, 21];
const ANXIETY = [2, 4, 7, 9, 15, 19, 20];
const SPEECH_BANNED = /depress|anxi|stress|disorder|risk score/i;

/** Records every microphone stream the page opens, so tests can check it was released. */
async function trackMicStreams(page: Page) {
  await page.addInitScript(() => {
    const w = window as unknown as { __streams: MediaStream[] };
    w.__streams = [];
    const orig = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (c) => {
      const s = await orig(c);
      w.__streams.push(s);
      return s;
    };
  });
}

async function liveMicTracks(page: Page): Promise<number> {
  return page.evaluate(() =>
    (window as unknown as { __streams: MediaStream[] }).__streams
      .flatMap((s) => s.getTracks())
      .filter((t) => t.readyState === "live").length,
  );
}

async function answerQuestionnaire(page: Page, valueFor: (n: number) => number) {
  for (let n = 1; n <= 21; n++) {
    await page.locator(`#q${n}-${valueFor(n)}`).check();
    const isPageEnd = n % 3 === 0;
    if (isPageEnd && n < 21) await page.getByRole("button", { name: "Next" }).click();
  }
  await page.getByRole("button", { name: "See my result" }).click();
}

async function expectNoHorizontalScroll(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
}

test("voice only: record with microphone, analyse, results show only the voice card", async ({ page }) => {
  await trackMicStreams(page);
  await page.goto("/");
  await expect(page.getByText("Your privacy:")).toBeVisible();
  await page.getByRole("link", { name: "Analyse a voice clip" }).click();

  // Consent gates everything.
  await expect(page.getByRole("button", { name: "Record with microphone" })).toHaveCount(0);
  await page.getByLabel(/I agree to send this clip/).check();
  await page.getByRole("button", { name: "Record with microphone" }).click();
  await page.getByRole("button", { name: "Allow microphone and start recording" }).click();

  const stop = page.getByRole("button", { name: /Stop/ });
  await expect(stop).toBeDisabled(); // minimum 5 s
  expect(await liveMicTracks(page)).toBeGreaterThan(0);
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeEnabled({ timeout: 10_000 });
  await page.waitForTimeout(1500);
  await page.getByRole("button", { name: "Stop recording" }).click();

  // Stopping releases the microphone.
  await expect.poll(() => liveMicTracks(page)).toBe(0);

  await expect(page.getByTestId("waveform")).toBeVisible();
  // The requirement is a 5-15 s clip. The exact length varies with machine load, so assert the range.
  await expect(page.getByTestId("clip-duration")).toHaveText(/^\d+\.\d s$/);
  const seconds = parseFloat(await page.getByTestId("clip-duration").innerText());
  expect(seconds).toBeGreaterThanOrEqual(5);
  expect(seconds).toBeLessThanOrEqual(15);
  await expect(page.getByLabel("Play back your clip")).toBeVisible();
  await expect(page.getByRole("button", { name: "Re-record" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Delete clip" })).toBeVisible();

  await page.getByRole("button", { name: "Analyse this clip" }).click();
  await expect(page.getByTestId("progress-steps")).toBeVisible();
  await expect(page.getByText("Voice check complete")).toBeVisible();
  await page.getByRole("link", { name: "View results" }).click();

  const card = page.getByTestId("voice-card");
  await expect(card).toBeVisible();
  await expect(page.getByTestId("survey-card")).toHaveCount(0);
  await expect(card.getByRole("heading", { level: 2 })).toHaveText(/Most likely vocal expression: \w+|couldn't tell clearly/);
  await expect(card.getByTestId("voice-confidence")).toContainText("%");
  await expect(card.getByText("Does this match how you feel right now?")).toBeVisible();
  await expect(card.getByTestId("not-diagnosis-note")).toBeVisible();
  // Every label from /api/model-info appears in the chart.
  for (const label of ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness"]) {
    await expect(card.locator(".recharts-yAxis").getByText(label, { exact: true })).toBeVisible();
  }
  expect(await card.innerText()).not.toMatch(SPEECH_BANNED);

  // Leaving the results page clears everything.
  await page.getByRole("link", { name: "Home" }).first().click();
  await page.getByRole("link", { name: "Results" }).first().click();
  await expect(page.getByText("There are no results to show.")).toBeVisible();
});

test("questionnaire only: keyboard answers, back keeps answers, results show only the questionnaire card", async ({ page }) => {
  await page.goto("/questionnaire");
  await expect(page.getByRole("radio")).toHaveCount(0); // consent first
  await page.getByLabel(/I agree to send my answers/).check();
  await expect(page.getByText("Over the past week…")).toBeVisible();
  await expect(page.getByTestId("item-q1")).toContainText("1.");
  await expect(page.getByText("Did not apply to me at all").first()).toBeVisible();

  // Keyboard: focus q1, arrow keys move the selection within the group.
  await page.locator("#q1-0").focus();
  await page.keyboard.press("Space");
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#q1-1")).toBeChecked();

  // Required: Next with unanswered items shows an error and focuses the first missing one.
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByText("Every statement needs an answer before you continue.")).toBeVisible();
  await expect(page.locator("#q2-0")).toBeFocused();

  // Answer q2 and q3 by keyboard, submit the page with Enter.
  await page.keyboard.press("Space");
  await page.locator("#q3-0").focus();
  await page.keyboard.press("Space");
  await page.keyboard.press("Enter");
  await expect(page.getByText("Page 2 of 7")).toBeVisible();

  // Back keeps earlier answers.
  await page.getByRole("button", { name: "Back" }).click();
  await expect(page.locator("#q1-1")).toBeChecked();
  await expect(page.getByText("3 of 21 answered")).toBeVisible();
  await page.getByRole("button", { name: "Next" }).click();

  for (let n = 4; n <= 21; n++) {
    await page.locator(`#q${n}-0`).check();
    if (n % 3 === 0 && n < 21) await page.getByRole("button", { name: "Next" }).click();
  }
  await page.getByRole("button", { name: "See my result" }).click();
  await expect(page.getByText("Questionnaire complete")).toBeVisible();
  await page.getByRole("link", { name: "View results" }).click();

  const card = page.getByTestId("survey-card");
  await expect(card.getByRole("heading", { name: "DASS-21 self-report screening result" })).toBeVisible();
  await expect(page.getByTestId("voice-card")).toHaveCount(0);
  for (const row of ["depression", "anxiety", "stress"]) {
    await expect(card.getByTestId(`row-${row}`)).toContainText("Score");
    await expect(card.getByTestId(`row-${row}`)).toContainText("Band Normal");
  }
  await expect(card.getByTestId("survey-summary")).toHaveText("All three scores are in the Normal range.");
  await expect(card.getByTestId("supportive-message")).toHaveCount(0);
  await expect(card.getByTestId("not-diagnosis-note")).toBeVisible();
});

test("both: upload a clip, then the questionnaire with a tie; two separate cards", async ({ page }) => {
  await page.goto("/voice");
  await page.getByLabel(/I agree to send this clip/).check();
  await page.getByRole("button", { name: "Upload an audio file" }).click();
  await page.getByLabel(/Choose an audio file/).setInputFiles(UPLOAD_CLIP);
  await expect(page.getByTestId("waveform")).toBeVisible();
  await expect(page.getByTestId("clip-duration")).toHaveText(/\d+\.\d s/);
  await page.getByRole("button", { name: "Analyse this clip" }).click();
  await expect(page.getByText("Voice check complete")).toBeVisible();

  await page.getByRole("link", { name: /Also take the wellbeing questionnaire/ }).click();
  await page.getByLabel(/I agree to send my answers/).check();
  // Depression 7 x 1 -> 14 (Moderate); Anxiety 5 x 1 -> 10 (Moderate); Stress 0 -> tie.
  await answerQuestionnaire(page, (n) => (DEPRESSION.includes(n) ? 1 : [2, 4, 7, 9, 15].includes(n) ? 1 : 0));
  await expect(page.getByText("Questionnaire complete")).toBeVisible();
  await expect(page.getByRole("link", { name: /Also analyse a voice clip/ })).toHaveCount(0);
  await page.getByRole("link", { name: "View results" }).click();

  const voice = page.getByTestId("voice-card");
  const survey = page.getByTestId("survey-card");
  await expect(voice).toBeVisible();
  await expect(survey).toBeVisible();
  await expect(page.getByText("never combined")).toBeVisible();
  await expect(survey.getByTestId("survey-tie")).toHaveText(/Depression and Anxiety are tied in the Moderate band/);
  await expect(survey.getByTestId("row-depression")).toContainText("Score 14");
  await expect(survey.getByTestId("supportive-message")).toBeVisible();
  expect(await voice.innerText()).not.toMatch(SPEECH_BANNED);
  await expect(page.getByTestId("not-diagnosis-note")).toHaveCount(2);
});

test("friendly error for a clip that is too short", async ({ page }) => {
  // 0.5 s tone, built in memory.
  const sr = 16000;
  const n = sr / 2;
  const buf = Buffer.alloc(44 + n * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write("WAVE", 8);
  buf.write("fmt ", 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) buf.writeInt16LE(Math.round(8000 * Math.sin((2 * Math.PI * 220 * i) / sr)), 44 + i * 2);

  await page.goto("/voice");
  await page.getByLabel(/I agree to send this clip/).check();
  await page.getByRole("button", { name: "Upload an audio file" }).click();
  await page.getByLabel(/Choose an audio file/).setInputFiles({ name: "short.wav", mimeType: "audio/wav", buffer: buf });
  await page.getByRole("button", { name: "Analyse this clip" }).click();
  const alert = page.locator("[data-error-code=audio_too_short]");
  await expect(alert).toContainText("The clip is too short");
});

test("about page lists limitations from the API; pages fit a phone screen", async ({ page }) => {
  await page.goto("/about");
  await expect(page.getByTestId("voice-limitations").locator("li").first()).toContainText("acted speech");
  await expect(page.getByTestId("survey-limitations")).toContainText("synthetic");
  await expect(page.getByText(/Emo-DB \(German, acted\) and SAVEE/)).toBeVisible();

  await page.setViewportSize({ width: 375, height: 800 });
  for (const p of ["/", "/voice", "/questionnaire", "/results", "/about", "/privacy"]) {
    await page.goto(p);
    await expectNoHorizontalScroll(page);
  }
});

test("no browser storage is used", async ({ page }) => {
  await page.goto("/questionnaire");
  await page.getByLabel(/I agree to send my answers/).check();
  await answerQuestionnaire(page, () => 0);
  await page.getByRole("link", { name: "View results" }).click();
  const stored = await page.evaluate(() => localStorage.length + sessionStorage.length);
  expect(stored).toBe(0);
});
