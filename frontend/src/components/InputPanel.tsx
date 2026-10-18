type InputPanelProps = {
  query: string;
  onQueryChange: (query: string) => void;
  imageName: string;
  afterName: string;
  sarName: string;
  onImageChange: (name: string) => void;
  onAfterChange: (name: string) => void;
  onSarChange: (name: string) => void;
  onSubmit: () => void;
  loading: boolean;
};

function fileName(event: React.ChangeEvent<HTMLInputElement>) {
  return event.target.files?.[0]?.name ?? "";
}

export function InputPanel(props: InputPanelProps) {
  return (
    <section className="panel input-panel">
      <p className="eyebrow">01 · INPUTS</p>
      <h2>Ask the landscape</h2>
      <label>Primary / optical image<input type="file" accept="image/*,.tif,.tiff" onChange={(event) => props.onImageChange(fileName(event))} /></label>
      <small>{props.imageName || "Select one image"}</small>
      <label>After image <span>(optional)</span><input type="file" accept="image/*,.tif,.tiff" onChange={(event) => props.onAfterChange(fileName(event))} /></label>
      <small>{props.afterName || "Enables change analysis"}</small>
      <label>SAR image <span>(optional)</span><input type="file" accept="image/*,.tif,.tiff" onChange={(event) => props.onSarChange(fileName(event))} /></label>
      <small>{props.sarName || "Enables optical-SAR analysis"}</small>
      <p className="hint">Choose either an after image or a SAR image, not both.</p>
      <button type="button" onClick={props.onSubmit} disabled={props.loading}>
        {props.loading ? "Routing analysis…" : "Run analysis"}
      </button>
      <textarea value={props.query} onChange={(event) => props.onQueryChange(event.target.value)} placeholder="What would you like to know?" aria-label="Analysis question" />
    </section>
  );
}
