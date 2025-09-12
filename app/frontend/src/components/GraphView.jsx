import { ForceGraph2D } from "react-force-graph";

export default function GraphView({ graph }) {
  const data = {
    nodes: [graph.query_node, ...graph.nodes],
    links: graph.edges.map(e => ({ source: e.source, target: e.target, value: e.weight }))
  };

  return (
    <ForceGraph2D
      graphData={data}
      nodeLabel="title"
      nodeAutoColorBy="pmid"
    />
  );
}
