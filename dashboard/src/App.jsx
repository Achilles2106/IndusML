import { useState } from "react";

const API_URL = "";

const TIER_STYLES = {
  pass: "bg-green-100 text-green-800 border-green-300",
  minor: "bg-yellow-100 text-yellow-800 border-yellow-300",
  moderate: "bg-orange-100 text-orange-800 border-orange-300",
  severe: "bg-red-100 text-red-800 border-red-300",
};

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const handleInspect = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_URL}/inspect`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setResult(data);
      setHistory((prev) => [
        {
          id: Date.now(),
          thumbnail: preview,
          classification: data.classification,
          defect_tier: data.defect_tier,
          severity_pct: data.severity_pct,
        },
        ...prev,
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-8">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-2xl font-bold text-slate-800 mb-6">
          Industrial Visual Inspection Dashboard
        </h1>

        <div className="bg-white rounded-lg shadow p-6 mb-6 flex items-center gap-4">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="text-sm text-slate-600"
          />
          <button
            onClick={handleInspect}
            disabled={!file || loading}
            className="px-4 py-2 bg-slate-800 text-white rounded-md text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-700"
          >
            {loading ? "Inspecting..." : "Inspect"}
          </button>
          {error && <span className="text-red-600 text-sm">{error}</span>}
        </div>

        {(preview || result) && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <p className="text-sm font-medium text-slate-500 mb-2">
                  Original
                </p>
                {preview && (
                  <img
                    src={preview}
                    alt="original"
                    className="rounded-md border w-full"
                  />
                )}
              </div>
              <div>
                <p className="text-sm font-medium text-slate-500 mb-2">
                  Defect Mask Overlay
                </p>
                {result && (
                  <img
                    src={`data:image/png;base64,${result.mask_overlay_base64}`}
                    alt="mask overlay"
                    className="rounded-md border w-full"
                  />
                )}
              </div>
            </div>

            {result && (
              <div className="mt-6 flex flex-wrap gap-4 items-center">
                <span
                  className={`px-3 py-1 rounded-full text-sm font-semibold border ${TIER_STYLES[result.defect_tier] ?? "bg-slate-100 text-slate-700 border-slate-300"}`}
                >
                  {result.defect_tier.toUpperCase()}
                </span>
                <span className="text-sm text-slate-700">
                  <strong>Classification:</strong> {result.classification}
                </span>
                <span className="text-sm text-slate-700">
                  <strong>Anomaly Score:</strong>{" "}
                  {result.anomaly_score.toFixed(2)}
                </span>
                <span className="text-sm text-slate-700">
                  <strong>Severity:</strong> {result.severity_pct.toFixed(2)}%
                </span>
                <span className="text-sm text-slate-700">
                  <strong>Defect Blobs:</strong> {result.num_defect_blobs}
                </span>
              </div>
            )}

            {result?.blobs?.length > 0 && (
              <table className="mt-4 w-full text-sm text-left border-t pt-2">
                <thead>
                  <tr className="text-slate-500">
                    <th className="py-1">Blob ID</th>
                    <th>Pixel Count</th>
                    <th>Severity %</th>
                  </tr>
                </thead>
                <tbody>
                  {result.blobs.map((b) => (
                    <tr key={b.blob_id} className="border-t">
                      <td className="py-1">{b.blob_id}</td>
                      <td>{b.pixel_count}</td>
                      <td>{b.severity_pct.toFixed(3)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {history.length > 0 && (
          <div className="bg-white rounded-lg shadow p-6">
            <p className="text-sm font-medium text-slate-500 mb-3">History</p>
            <div className="flex flex-col gap-2">
              {history.map((h) => (
                <div
                  key={h.id}
                  className="flex items-center gap-3 border-b pb-2 last:border-0"
                >
                  <img
                    src={h.thumbnail}
                    alt=""
                    className="w-10 h-10 object-cover rounded"
                  />
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-semibold border ${TIER_STYLES[h.defect_tier] ?? "bg-slate-100 text-slate-700 border-slate-300"}`}
                  >
                    {h.defect_tier}
                  </span>
                  <span className="text-sm text-slate-600">
                    {h.classification}
                  </span>
                  <span className="text-sm text-slate-600 ml-auto">
                    {h.severity_pct.toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
