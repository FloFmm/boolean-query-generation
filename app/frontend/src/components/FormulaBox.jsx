export default function FormulaBox({ formula }) {
  return (
    <div>
      <textarea
        value={formula}
        readOnly
        className="w-full border p-2 rounded"
        rows={4}
      />
      <button
        className="mt-2 px-4 py-2 bg-green-500 text-white rounded"
        onClick={() => navigator.clipboard.writeText(formula)}
      >
        Copy Formula
      </button>
    </div>
  );
}
