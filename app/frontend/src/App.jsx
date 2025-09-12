import { useState } from "react";
import SearchBar from "./components/SearchBar";
import GraphView from "./components/GraphView";
import SidePanel from "./components/SidePanel";

export default function App() {
  const [graph, setGraph] = useState(null);
  const [tree, setTree] = useState("");
  const [formula, setFormula] = useState("");

  return (
    <div className="flex h-screen">
      <div className="flex-1">
        <SearchBar onResult={setGraph} />
        {graph && <GraphView graph={graph} />}
      </div>
      <SidePanel tree={tree} formula={formula} />
    </div>
  );
}
