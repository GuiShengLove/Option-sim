import pandas as pd

inst = pd.read_pickle("Filtered_OptionInstruments_510050.pkl")
try:
    price = pd.read_feather("Filtered_OptionPrice_2020_2022.feather")
except:
    price = pd.read_csv("Filtered_OptionPrice_2020_2022.csv")

inst_ids = set(inst["order_book_id"].astype(str))
price_ids = set(price["order_book_id"].astype(str))

replayable = sorted(inst_ids & price_ids)
missing = sorted(inst_ids - price_ids)

print("\n===== Replayable contracts =====")
print(f"Count = {len(replayable)}")
print(replayable[:30])

print("\n===== Missing contracts =====")
print(f"Count = {len(missing)}")
print(missing[:30])

# optional: save replayable list to CSV
pd.DataFrame({"order_book_id": replayable}).to_csv("replayable_ids.csv", index=False)
