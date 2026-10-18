export default function InferenceNotice({ result }) {
  if (!result) return null;
  const label = result.label === 'BASELINE'
    ? 'Baseline'
    : result.is_trained === false ? 'Untrained model' : 'Checkpoint model';
  return <span className="inference-label" data-testid="inference-notice">{label}</span>;
}
