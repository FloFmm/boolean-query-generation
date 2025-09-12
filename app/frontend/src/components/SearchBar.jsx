import { useState } from "react";
import { fetchSearch } from "../api";

export default function SearchBar({ onResult }) {
  const [query, setQuery] = useState("");

  const handleSearch = async () => {
    const res = await fetchSearch(query);
    onResult(res);
  };

  return (
    <div className="p-4 flex gap-2">
      <input
        className="border rounded p-2 flex-1"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter PubMed query"
      />
      <button onClick={handleSearch} className="px-4 py-2 bg-blue-500 text-white rounded">
        Search
      </button>
    </div>
  );
}
