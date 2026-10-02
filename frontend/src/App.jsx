import { useState } from 'react';
import { Activity, AlertCircle, ArrowUpRight, CheckCircle2, LoaderCircle, ShieldCheck } from 'lucide-react';

const features = [
  { name: 'loc', label: 'Lines of code', step: 'any' },
  { name: 'v(g)', label: 'Cyclomatic complexity', step: 'any' },
  { name: 'ev(g)', label: 'Essential complexity', step: 'any' },
  { name: 'iv(g)', label: 'Design complexity', step: 'any' },
  { name: 'n', label: 'Halstead length', step: 'any' },
  { name: 'v', label: 'Halstead volume', step: 'any' },
  { name: 'l', label: 'Halstead program length', step: 'any' },
  { name: 'd', label: 'Halstead difficulty', step: 'any' },
  { name: 'i', label: 'Halstead intelligence', step: 'any' },
  { name: 'e', label: 'Halstead effort', step: 'any' },
  { name: 'b', label: 'Halstead estimated bugs', step: 'any' },
  { name: 't', label: 'Halstead time', step: 'any' },
  { name: 'lOCode', label: 'Lines of code (logical)', step: 'any' },
  { name: 'lOComment', label: 'Comment lines', step: 'any' },
  { name: 'lOBlank', label: 'Blank lines', step: 'any' },
  { name: 'locCodeAndComment', label: 'Code and comment lines', step: 'any' },
  { name: 'uniq_Op', label: 'Unique operators', step: 'any' },
  { name: 'uniq_Opnd', label: 'Unique operands', step: 'any' },
  { name: 'total_Op', label: 'Total operators', step: 'any' },
  { name: 'total_Opnd', label: 'Total operands', step: 'any' },
  { name: 'branchCount', label: 'Branch count', step: 'any' },
];

const initialValues = Object.fromEntries(features.map(({ name }) => [name, '']));

function App() {
  const [values, setValues] = useState(initialValues);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  function updateValue(name, value) {
    setValues((current) => ({ ...current, [name]: value }));
  }

  async function submitPrediction(event) {
    event.preventDefault();
    setError('');
    setResult(null);
    setLoading(true);

    const payload = Object.fromEntries(
      features.map(({ name }) => [name, Number(values[name])]),
    );

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const body = await response.json();
      if (!response.ok) {
        const detail = typeof body.detail === 'string' ? body.detail : 'Check the entered feature values.';
        throw new Error(detail);
      }
      setResult(body);
    } catch (requestError) {
      setError(requestError.message || 'Could not reach the prediction service. Check that the API is running.');
    } finally {
      setLoading(false);
    }
  }

  const riskTone = result?.risk_level?.toLowerCase() || '';

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="QA Analytics home">
          <span className="brand-mark"><Activity size={19} strokeWidth={2.2} /></span>
          <span>QA<span className="brand-light"> / </span>ANALYTICS</span>
        </a>
        <div className="service-status"><span className="status-dot" /> Model service</div>
      </header>

      <section className="intro" id="top">
        <div className="intro-copy">
          <p className="eyebrow">QUALITY ENGINEERING / MODULE REVIEW</p>
          <h1>Software Defect Prediction <span>&amp; QA Analytics</span></h1>
          <p className="description">Predict the defect risk of a software module using software metrics.</p>
        </div>
        <div className="intro-aside" aria-hidden="true">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="orbit-core"><ShieldCheck size={33} strokeWidth={1.4} /></div>
          <span className="orbit-label">MODULE<br />SIGNAL</span>
        </div>
      </section>

      <div className="workspace">
        <section className="form-panel" aria-labelledby="metrics-title">
          <div className="section-heading">
            <div>
              <p className="section-kicker">01 / INPUT</p>
              <h2 id="metrics-title">Module metrics</h2>
            </div>
            <span className="field-count">21 FEATURES</span>
          </div>
          <p className="form-note">Enter numeric values for each feature used by the trained model.</p>

          <form onSubmit={submitPrediction}>
            <div className="feature-grid">
              {features.map(({ name, label }) => (
                <label className="field" key={name}>
                  <span className="field-label">{label}</span>
                  <span className="field-meta">{name}</span>
                  <input
                    type="number"
                    name={name}
                    value={values[name]}
                    step="any"
                    required
                    inputMode="decimal"
                    onChange={(event) => updateValue(name, event.target.value)}
                    aria-label={`${label} (${name})`}
                  />
                </label>
              ))}
            </div>

            <div className="form-footer">
              <span className="required-note">All feature values are required.</span>
              <button className="predict-button" type="submit" disabled={loading}>
                {loading ? <LoaderCircle className="spin" size={17} /> : <ArrowUpRight size={17} />}
                {loading ? 'Analyzing module' : 'Predict Defect Risk'}
              </button>
            </div>
          </form>

          {error && (
            <div className="error-message" role="alert">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}
        </section>

        <aside className="result-panel" aria-live="polite">
          <div className="section-heading result-heading">
            <div>
              <p className="section-kicker">02 / RESULT</p>
              <h2>Risk assessment</h2>
            </div>
            {result && <CheckCircle2 className="result-check" size={20} />}
          </div>
          {!result ? (
            <div className="empty-result">
              <div className="empty-icon"><Activity size={22} /></div>
              <p className="empty-title">Awaiting module metrics</p>
              <p className="empty-copy">Your prediction and risk summary will appear here.</p>
            </div>
          ) : (
            <div className="result-content">
              <div className={`risk-banner risk-${riskTone}`}>
                <span className="risk-label">RISK LEVEL</span>
                <strong>{result.risk_level}</strong>
                <span className="prediction-label">
                  {result.prediction ? 'Defect predicted' : 'No defect predicted'}
                </span>
              </div>
              <div className="probability-block">
                <div className="probability-row">
                  <span>Defect probability</span>
                  <strong>{result.defect_probability == null ? 'Unavailable' : `${(result.defect_probability * 100).toFixed(1)}%`}</strong>
                </div>
                {result.defect_probability != null && (
                  <div className="meter" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow={result.defect_probability * 100}>
                    <span style={{ width: `${Math.max(0, Math.min(100, result.defect_probability * 100))}%` }} />
                  </div>
                )}
                <div className="meter-scale"><span>0%</span><span>100%</span></div>
              </div>
              <p className="risk-thresholds">Low &lt; 30% <i /> Medium 30–60% <i /> High &gt; 60%</p>
            </div>
          )}
          <div className="decision-note">
            <span className="note-rule" />
            <p>This system is a QA decision-support tool and does not replace professional software testing.</p>
          </div>
        </aside>
      </div>
      <footer className="page-footer"><span>SOFTWARE QUALITY INTELLIGENCE</span><span>DEFECT RISK / CLASSIFICATION</span></footer>
    </main>
  );
}

export default App;
