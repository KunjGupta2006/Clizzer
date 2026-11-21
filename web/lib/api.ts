const rawApiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_BASE = rawApiBase.replace(/\/+$/, "");

export interface JobData {
  id: string;
  status: string;
  progress: number;
  stage: string;
  timings: Record<string, number>;
  error?: string | null;
  created_at: string;
  params: Record<string, any>;
}

export interface ClipData {
  title: string;
  hook: string;
  hashtags: string[];
  file: string;
  srt: string;
  thumb: string;
  start: number;
  end: number;
  peak_t: number;
  fused_score: number;
  breakdown: Record<string, number>;
  reason: string;
  music_ratio: number;
  source_url?: string;
  attribution?: string;
}

export interface CurveData {
  duration: number;
  length: number;
  fused: number[];
  audio: number[];
  chat: number[];
  text: number[];
  visual: number[];
  face: number[];
}

export async function createJob(body: Record<string, any> | FormData): Promise<{ job_id: string }> {
  let headers: Record<string, string> = {};
  let reqBody: any = body;

  if (!(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    reqBody = JSON.stringify(body);
  }

  const res = await fetch(`${API_BASE}/jobs`, {
    method: "POST",
    headers,
    body: reqBody,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: "Failed to submit job" }));
    throw new Error(errorData.detail || "Job creation failed");
  }

  return res.json();
}

export async function getJob(jobId: string): Promise<JobData> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error("Failed to fetch job");
  return res.json();
}

export async function getJobClips(jobId: string): Promise<ClipData[]> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/clips`);
  if (!res.ok) return [];
  return res.json();
}

export async function getJobCurve(jobId: string): Promise<CurveData | null> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/curve`);
  if (!res.ok) return null;
  return res.json();
}

export async function rerankJob(jobId: string, payload: Record<string, any>): Promise<ClipData[]> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/rerank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Reranking failed");
  return res.json();
}

export async function trimClip(clipId: string, startDelta: number, endDelta: number): Promise<ClipData> {
  const res = await fetch(`${API_BASE}/clips/${clipId}/trim`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start_delta: startDelta, end_delta: endDelta }),
  });
  if (!res.ok) throw new Error("Trimming failed");
  return res.json();
}

export function getClipDownloadUrl(clipId: string, jobId?: string): string {
  if (jobId) {
    const filename = clipId.includes(".mp4") ? clipId : `${clipId}.mp4`;
    return `${API_BASE}/jobs/${jobId}/clips/${filename}/download`;
  }
  return `${API_BASE}/clips/${clipId}/download`;
}

export function getWsUrl(jobId: string): string {
  const wsProto = API_BASE.startsWith("https") ? "wss" : "ws";
  const host = API_BASE.replace(/^https?:\/\//, "");
  return `${wsProto}://${host}/jobs/${jobId}/events`;
}
