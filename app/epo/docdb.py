import argparse
import json


BASE_URL = "https://publication-bdds.apps.epo.org/bdds/bdds-bff-service/prod/api/public/products/{product_id}/delivery/{delivery_id}/item/{item_id}/download"

# full_text
# "https://publication-bdds.apps.epo.org/bdds/bdds-bff-service/prod/api/public/products/32/delivery/2792/item/7925/download" download="EPRTBJV2025000040001001.zip"


def build_aria2_cfg(data, product_id, delivery_id, out_path):
    deliveries = data.get("deliveries", [])

    all_delivery_ids = [str(d.get("deliveryId")) for d in deliveries]
    targets = []
    if delivery_id is None:
        targets = deliveries
    else:
        if str(delivery_id) not in all_delivery_ids:
            raise ValueError(f"Delivery ID {delivery_id} not found in JSON.")
        for d in deliveries:
            if str(d.get("deliveryId")) == str(delivery_id):
                targets = [d]
                break
    if not targets:
        raise ValueError(f"Delivery ID {delivery_id} not found in JSON.")

    with open(out_path, "w", encoding="utf-8") as f:
        for target in targets:
            for item in target.get("items", []):
                item_id = item.get("itemId")
                item_name = item.get("itemName")
                checksum = item.get("fileChecksum")
                if not item_id or not item_name:
                    continue

                url = BASE_URL.format(
                    product_id=product_id,
                    delivery_id=str(target.get("deliveryId")),
                    item_id=item_id,
                )
                f.write(f"{url}\n")
                f.write(f"  out={item_name}\n")
                if checksum:
                    # fileChecksum appears to be SHA-1 (40 hex chars)
                    f.write(f"  checksum=sha-1={checksum.lower()}\n")
                f.write("\n")


def main():
    ap = argparse.ArgumentParser(
        description="Generate aria2c input file for a delivery."
    )
    ap.add_argument("--input", "-i", required=True, help="Path to JSON file.")
    ap.add_argument(
        "--output",
        "-o",
        default="aria2_downloads.txt",
        help="Output aria2c input file.",
    )
    ap.add_argument(
        "--product-id", "-p", default="14", help="Product ID (default: 14)."
    )
    ap.add_argument(
        "--delivery-id", "-d", default=None, help="Delivery ID (default: 2238)."
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # If product id is present in JSON, prefer that unless overridden
    product_id = args.product_id or data.get("id")
    if not product_id:
        raise ValueError("Product ID not provided and not found in JSON.")

    build_aria2_cfg(data, product_id, args.delivery_id, args.output)
    print(f"Wrote aria2c input file: {args.output}")


# Run with:
# python docdb.py --input data.json --output aria2_downloads.txt --product-id 14 --delivery-id 2238
if __name__ == "__main__":
    main()
