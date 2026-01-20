import pandas as pd

print("\n===== Loading Files =====")
inst = pd.read_pickle("Filtered_OptionInstruments_510050.pkl")
try:
    price = pd.read_feather("Filtered_OptionPrice_2020_2022.feather")
except:
    price = pd.read_csv("Filtered_OptionPrice_2020_2022.csv")

print("Loaded instruments:", inst.shape)
print("Loaded price file:", price.shape)

# -------------------------
# 1. Show columns
# -------------------------
print("\n===== Columns in instruments.pkl =====")
print(inst.columns.tolist())

print("\n===== Columns in price file =====")
print(price.columns.tolist())

# -------------------------
# 2. Unique IDs
# -------------------------
print("\n===== Unique order_book_id in instruments.pkl =====")
print(inst["order_book_id"].unique()[:20])  # print first 20

print("\n===== Unique order_book_id in price file =====")
if "order_book_id" in price.columns:
    print(price["order_book_id"].unique()[:20])
else:
    print("⚠ price file has NO column 'order_book_id'!")

# -------------------------
# 3. Check which instrument IDs exist in price file
# -------------------------
if "order_book_id" in price.columns:
    inst_ids = set(inst["order_book_id"].astype(str))
    price_ids = set(price["order_book_id"].astype(str))

    common = inst_ids & price_ids
    missing = inst_ids - price_ids

    print(f"\n===== Matched IDs (in both files): {len(common)} =====")
    print(list(common)[:20])

    print(f"\n===== Missing IDs (not in price file): {len(missing)} =====")
    print(list(missing)[:20])
else:
    print("\n⚠ Cannot compare IDs — price file has no order_book_id column.")

# -------------------------
# 4. Check symbol column existence
# -------------------------
possible_symbol_cols = [c for c in price.columns if "symbol" in c.lower() or "name" in c.lower()]
print("\n===== Possible symbol-like columns in price file =====")
print(possible_symbol_cols)

if possible_symbol_cols:
    for col in possible_symbol_cols:
        print(f"\n--- Unique values in {col} (first 20) ---")
        print(price[col].dropna().unique()[:20])

# -------------------------
# 5. Summary
# -------------------------
print("\n===== SUMMARY =====")
print("✓ Now you can verify whether the instrument-side identifiers match price-side identifiers.")
print("✓ If order_book_id sets do NOT intersect, the two files are not from the same data system.")
print("✓ Next step is to build manual or learned mapping once we know actual IDs present.")
