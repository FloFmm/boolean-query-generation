import pandas as pd
import matplotlib.pyplot as plt

def visualize_mesh_coverage(csv_path: str, n: int = None):
    """
    Visualize MeSH coverage statistics from a CSV file.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        n (int, optional): Only visualize the first n steps. 
                           If None, visualize all rows.
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Optionally limit to first n rows
    if n is not None:
        df = df.head(n)

    # Prepare x and y data
    x = df["step"]
    y_pos = df["num_covered_pos_pmids_acc"]
    y_neg = df["num_covered_neg_pmids_acc"]
    y_pubmed = df["num_covered_pubmed_acc"]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_pos, label="Accumulated Positive PMIDs", color="tab:green", marker="o")
    plt.plot(x, y_neg, label="Accumulated Negative PMIDs", color="tab:red", marker="o")
    plt.plot(x, y_pubmed, label="Accumulated PubMed Coverage", color="tab:blue", marker="o")

    # Add horizontal reference lines
    plt.axhline(y=350958, color="tab:red", linestyle="--", label="Negatives Reference (350,958)")
    plt.axhline(y=5070, color="tab:green", linestyle="--", label="Positives Reference (5,070)")
    plt.yscale("log")
    plt.ylabel("Cumulative Coverage (log scale)")

    # Labels and title
    plt.xlabel("Number of Included MeSH Terms")
    # plt.ylabel("Cumulative Coverage")
    plt.title("MeSH Term Coverage over Steps")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Example usage
    csv_path = "data/pubmed/statistics/mesh.csv"  # Replace with your actual file
    visualize_mesh_coverage(csv_path, n=5000)  # visualize only first 5 lines
