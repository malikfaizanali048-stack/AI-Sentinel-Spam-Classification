"use client";
import { useState } from "react";

export default function BatchCheck() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    const text = await file.text();
    // Expecting a simple CSV: subject,body per line — basic parsing for now
    const lines = text.split("\n").slice(1); // skip header
    const emails = lines.filter(l => l.trim()).map(line => {
      const [subject, ...bodyParts] = line.split(",");
      return { subject, body: bodyParts.join(","), from: "" };
    });

    const res = await fetch("http://localhost:5000/api/check-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ emails }),
    });
    const data = await res.json();
    setResults(data);
    setLoading(false);
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Batch Email Check</h1>
      <p className="text-gray-600 mb-4">Upload a CSV with columns: subject, body</p>
      <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />
      <button
        onClick={handleUpload}
        className="ml-4 bg-blue-600 text-white px-6 py-2 rounded-lg"
      >
        {loading ? "Processing..." : "Upload & Check"}
      </button>

      <div className="mt-6 space-y-2">
        {results.map((email, i) => (
          <div key={i} className="border p-3 rounded bg-gray-50">
            <strong>{email.label}</strong> ({email.score}) — {email.subject}
          </div>
        ))}
      </div>
    </div>
  );
}