import FormulaBox from "./FormulaBox";

export default function SidePanel({ tree, formula }) {
  return (
    <div className="w-1/3 border-l p-4 overflow-y-auto">
      <h2 className="text-xl font-bold">Decision Tree</h2>
      <pre className="bg-gray-100 p-2 rounded">{tree}</pre>

      <h2 className="text-xl font-bold mt-4">Formula</h2>
      <FormulaBox formula={formula} />
    </div>
  );
}
