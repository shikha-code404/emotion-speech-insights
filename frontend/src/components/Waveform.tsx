import { useEffect, useState } from "react";

const BARS = 120;

/** Decodes an audio blob in the browser and draws a peak waveform as SVG. */
export function useDecodedAudio(blob: Blob | null) {
  const [state, setState] = useState<{ peaks: number[] | null; duration: number | null; failed: boolean }>({
    peaks: null,
    duration: null,
    failed: false,
  });

  useEffect(() => {
    if (!blob) {
      setState({ peaks: null, duration: null, failed: false });
      return;
    }
    let cancelled = false;
    const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new Ctx();
    blob
      .arrayBuffer()
      .then((buf) => ctx.decodeAudioData(buf))
      .then((audio) => {
        if (cancelled) return;
        const data = audio.getChannelData(0);
        const step = Math.max(1, Math.floor(data.length / BARS));
        const peaks: number[] = [];
        for (let i = 0; i < BARS; i++) {
          let max = 0;
          for (let j = i * step; j < Math.min((i + 1) * step, data.length); j++) {
            const v = Math.abs(data[j]);
            if (v > max) max = v;
          }
          peaks.push(max);
        }
        const top = Math.max(...peaks, 1e-6);
        setState({ peaks: peaks.map((p) => p / top), duration: audio.duration, failed: false });
      })
      .catch(() => !cancelled && setState({ peaks: null, duration: null, failed: true }))
      .finally(() => void ctx.close());
    return () => {
      cancelled = true;
    };
  }, [blob]);

  return state;
}

export function Waveform({ peaks }: { peaks: number[] }) {
  const w = 600;
  const h = 80;
  const bw = w / peaks.length;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="h-20 w-full" role="img" aria-label="Waveform of your clip" data-testid="waveform">
      {peaks.map((p, i) => {
        const bh = Math.max(2, p * (h - 4));
        return <rect key={i} x={i * bw + 1} y={(h - bh) / 2} width={Math.max(1, bw - 2)} height={bh} rx={1.5} className="fill-teal-500" />;
      })}
    </svg>
  );
}

export function formatSeconds(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return "–";
  return `${s.toFixed(1)} s`;
}
