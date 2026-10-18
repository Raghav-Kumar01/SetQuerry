export type AnalysisResult = {
  task: "single_image" | "change_analysis" | "optical_sar";
  query: string;
  answer: string;
  confidence: number;
  evidence: string[];
  model: string;
  model_version: string;
  execution_trace: string[];
  latency_ms: number;
  is_mock: boolean;
};

export type AnalyzePayload = {
  query: string;
  image: string;
  image_after?: string;
  sar_image?: string;
};

const apiBase = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export async function analyze(payload: AnalyzePayload): Promise<AnalysisResult> {
  const response = await fetch(`${apiBase}/api/v1/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? "The analysis request failed.");
  }
  return response.json() as Promise<AnalysisResult>;
}
