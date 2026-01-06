import json

raw_dataset = "/wkspace/sva_toolkit/examples/xepic_sva.json"
preprocessed_dataset = "/wkspace/sva_toolkit/examples/xepic_sva_only.json"
preprocessed_data = []
with open(raw_dataset, "r") as f:
    raw_data = json.load(f)

chunk_size = 1000
for i in range(chunk_size):
    sva = raw_data[i]['sva']
    sva_id = raw_data[i]['id']
    preprocessed_data.append({
        "sva": sva,
        "sva_id": sva_id
    })

with open(preprocessed_dataset, "w") as f:
    json.dump(preprocessed_data, f, indent=4)

print(f"Preprocessed {len(preprocessed_data)} SVA properties")

# raw_dataset = "/wkspace/sva_toolkit/examples/ChatSVA-sft-case2svad_sva_with_ids.json"
# preprocessed_dataset = "/wkspace/sva_toolkit/examples/syntax_check_2.json"

# preprocessed_data = []
# with open(raw_dataset, "r") as f:
#     raw_data = json.load(f)
#     for item in raw_data:
#         # preprocessed_data.append({
#         #     "id": item['id'],
#         #     "sva1": item['ori_sva'],
#         #     "sva2": item['full_property']
#         #     })
#         try:
#             preprocessed_data.append({
#                 "id": item['id'],
#                 "sva1": item['ori_sva'],
#                 "sva2": item['full_property']
#                 })
#         except:
#             print(f"Error processing item {item['id']}")

# with open(preprocessed_dataset, "w") as f:
#     json.dump(preprocessed_data, f, indent=4)

# print(f"Preprocessed {len(preprocessed_data)} SVA properties")