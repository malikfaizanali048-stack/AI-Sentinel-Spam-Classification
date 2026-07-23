"use client";
import { useState } from "react";

export default function Home() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const checkInbox = async () => {
    setLoading(true);
    const res = await fetch("http://localhost:5000/api/check-inbox");
    const data = await res.json();
    setResults(data);
    setLoading(false);
  };

  const colorFor = (label) => {
    if (label === "SPAM") return "border-red-500";
    if (label === "SUSPICIOUS") return "border-yellow-500";
    return "border-green-500";
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Inbox Spam Checker</h1>
      <button
        onClick={checkInbox}
        className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium"
      >
        {loading ? "Checking..." : "Check My Inbox"}
      </button>

      <div className="mt-6 space-y-3">
        {results.map((email, i) => (
          <div key={i} className={`border-l-4 ${colorFor(email.label)} bg-gray-50 p-4 rounded`}>
            <div className="font-semibold">{email.label} ({email.score})</div>
            <div className="font-medium">{email.subject}</div>
            <div className="text-sm text-gray-500">{email.from}</div>
          </div>
        ))}
      </div>
    </div>
  );
}