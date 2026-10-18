import type { AnalysisResult } from "../services/api";

export function ResultPanel({ result, error }: { result: AnalysisResult | null; error: string | null }) {
  return (
    <section className="panel result-panel">
      <p className="eyebrow">03 · AI RESULT</p>
      <h2>{result ? result.task.replace("_", " ") : "Awaiting signal"}</h2>
      {error && <p className="error">{error}</p>}
      {!result && !error && <p className="empty">Your answer, evidence, and execution trace will appear here.</p>}
      {result && <>
        <p className="mock-badge">{result.is_mock ? "MOCK BASELINE" : "MODEL RESULT"}</p>
        <p className="answer">{result.answer}</p>
        <dl>
          <div><dt>Confidence</dt><dd>{Math.round(result.confidence * 100)}%</dd></div>
          <div><dt>Model</dt><dd>{result.model}</dd></div>
          <div><dt>Latency</dt><dd>{result.latency_ms} ms</dd></div>
        </dl>
        <h3>Evidence</h3><p>{result.evidence.length ? result.evidence.join(", ") : "No evidence from the untrained baseline."}</p>
        <h3>Execution trace</h3><ol>{result.execution_trace.map((step) => <li key={step}>{step}</li>)}</ol>
      </>}
    </section>
  );
}
