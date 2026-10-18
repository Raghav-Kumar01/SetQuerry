import { useState } from "react";

import { InputPanel } from "./components/InputPanel";
import { ResultPanel } from "./components/ResultPanel";
import { analyze, type AnalysisResult } from "./services/api";
import { Earth } from "./three/Earth";

export default function App() {
  const [query, setQuery] = useState("What changes are visible in this area?");
  const [imageName, setImageName] = useState("");
  const [afterName, setAfterName] = useState("");
  const [sarName, setSarName] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit() {
    setError(null);
    if (!imageName) return setError("Select a primary or optical image first.");
    if (afterName && sarName) return setError("Use either an after image or a SAR image.");
    setLoading(true);
    try {
      setResult(await analyze({ query, image: imageName, image_after: afterName || undefined, sar_image: sarName || undefined }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The analysis request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header><span className="brand">SET<span>Q</span>UERRY</span><p>Agentic remote sensing workspace</p></header>
      <div className="workspace">
        <InputPanel query={query} onQueryChange={setQuery} imageName={imageName} afterName={afterName} sarName={sarName} onImageChange={setImageName} onAfterChange={setAfterName} onSarChange={setSarName} onSubmit={submit} loading={loading} />
        <section className="panel globe-panel"><p className="eyebrow">02 · ORBIT VIEW</p><Earth /><p className="orbit-caption">Live visual context · lightweight preview</p></section>
        <ResultPanel result={result} error={error} />
      </div>
    </main>
  );
}
