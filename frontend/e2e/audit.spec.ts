// Audit checks (blueprint section 11): uncertain display, keyboard-only use at phone width.
import { expect, test, type Page } from "@playwright/test";

const SPEECH_BANNED = /depress|anxi|stress|disorder|risk score/i;

/** Press Tab (or Shift+Tab) until the focused element matches; fail if it never does. */
async function tabTo(page: Page, match: (el: { text: string; id: string; name: string }) => boolean, back = false) {
  for (let i = 0; i < 80; i++) {
    await page.keyboard.press(back ? "Shift+Tab" : "Tab");
    const el = await page.evaluate(() => {
      const a = document.activeElement as HTMLElement & { name?: string };
      return { text: (a?.innerText || a?.getAttribute("aria-label") || "").trim(), id: a?.id ?? "", name: a?.name ?? "" };
    });
    if (match(el)) return;
  }
  throw new Error("element not reachable by keyboard");
}

async function focusedText(page: Page): Promise<string> {
  return page.evaluate(() => (document.activeElement as HTMLElement)?.innerText?.trim() ?? "");
}

function prediction(scores: Record<string, number>, uncertainty: "low" | "moderate" | "high") {
  const top = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  return {
    prediction_type: "vocal_emotion", predicted_emotion: top[0], confidence: top[1],
    scores: Object.entries(scores).map(([emotion, score]) => ({ emotion, score })),
    uncertainty, model_version: "test", audio_duration_seconds: 6.0,
    notice: "This prediction reflects an estimate of the vocal emotion expressed in this audio clip.",
  };
}

for (const [uncertainty, scores, expectHeading, expectWords] of [
  ["high", { anger: 0.22, disgust: 0.2, fear: 0.18, happiness: 0.16, neutral: 0.14, sadness: 0.1 },
    /couldn't tell clearly/, /unsure/],
  ["moderate", { anger: 0.5, disgust: 0.1, fear: 0.1, happiness: 0.2, neutral: 0.05, sadness: 0.05 },
    /Most likely vocal expression: Anger/, /only somewhat sure/],
] as const) {
  test(`low-confidence voice result is shown as uncertain (${uncertainty})`, async ({ page }) => {
    await page.route("**/api/analyze", (route) => route.fulfill({ json: prediction(scores, uncertainty) }));
    await page.goto("/voice");
    await page.getByLabel(/I agree to send this clip/).check();
    await page.getByRole("button", { name: "Upload an audio file" }).click();
    const wav = Buffer.alloc(44 + 32000);
    wav.write("RIFF", 0); wav.writeUInt32LE(36 + 32000, 4); wav.write("WAVEfmt ", 8); wav.writeUInt32LE(16, 16);
    wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22); wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28);
    wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34); wav.write("data", 36); wav.writeUInt32LE(32000, 40);
    await page.getByLabel(/Choose an audio file/).setInputFiles({ name: "a.wav", mimeType: "audio/wav", buffer: wav });
    await page.getByRole("button", { name: "Analyse this clip" }).click();
    await page.getByRole("link", { name: "View results" }).click();
    const card = page.getByTestId("voice-card");
    await expect(card.getByRole("heading", { level: 2 })).toHaveText(expectHeading);
    await expect(card.getByTestId("voice-confidence")).toContainText(expectWords);
    if (uncertainty === "high") await expect(card.getByTestId("voice-uncertain")).toContainText("Closest matches");
    expect(await card.innerText()).not.toMatch(SPEECH_BANNED);
  });
}

test("both flows work at phone width with the keyboard only", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 });
  await page.goto("/");

  // Voice check: record with the (fake) microphone, keyboard only.
  await tabTo(page, (e) => e.text === "Analyse a voice clip");
  await page.keyboard.press("Enter");
  await tabTo(page, (e) => e.id === "voice-consent");
  await page.keyboard.press("Space");
  await tabTo(page, (e) => e.text === "Record with microphone");
  await page.keyboard.press("Enter");
  expect(await focusedText(page)).toBe("Allow microphone and start recording");   // focus followed the flow
  await page.keyboard.press("Enter");
  await expect(page.getByRole("button", { name: "Cancel" })).toBeFocused();
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeEnabled({ timeout: 10_000 });
  await page.keyboard.press("Shift+Tab");
  expect(await focusedText(page)).toBe("Stop recording");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("button", { name: "Analyse this clip" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("link", { name: "View results" })).toBeFocused({ timeout: 30_000 });

  // Questionnaire, keyboard only: Tab between statements, Space to answer, Enter to submit each page.
  await tabTo(page, (e) => e.text.startsWith("Also take the wellbeing questionnaire"));
  await page.keyboard.press("Enter");
  await tabTo(page, (e) => e.id === "survey-consent");
  await page.keyboard.press("Space");
  await expect(page.getByText("Over the past week…")).toBeVisible();
  for (let p = 0; p < 7; p++) {
    for (let q = 0; q < 3; q++) {
      await tabTo(page, (e) => /^q\d+$/.test(e.name));
      await page.keyboard.press("Space");
    }
    await page.keyboard.press("Enter");   // submits the form: Next, or See my result on the last page
  }
  await expect(page.getByRole("link", { name: "View results" })).toBeFocused({ timeout: 15_000 });
  await page.keyboard.press("Enter");

  // Results: both cards, readable at 375 px, reflection buttons reachable by keyboard.
  await expect(page.getByTestId("voice-card")).toBeVisible();
  await expect(page.getByTestId("survey-card")).toContainText("self-report screening result");
  await expect(page.getByTestId("survey-card")).toContainText("not a diagnosis");
  await tabTo(page, (e) => e.text === "Partly");
  await page.keyboard.press("Enter");
  await expect(page.getByText(/only captures part of how someone feels/)).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});
