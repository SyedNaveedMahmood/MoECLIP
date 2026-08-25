"""Quick smoke test: build MoECLIP, run 2 training steps on MVTec, then 1 test batch."""
import os
import torch
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import setup_seed
from model.moe_adapter import MoECLIP
from model.clip import create_model
from dataset import get_dataset
from forward_utils import (
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
)

setup_seed(111)
device = torch.device("cuda:0")
print("[1/5] loading CLIP backbone ...", flush=True)
clip_model = create_model(
    model_name="ViT-L-14-336",
    img_size=518,
    device=device,
    pretrained="openai",
    require_pretrained=True,
)
clip_model.eval()
model = MoECLIP(clip_model=clip_model).to(device)
model.eval()

for p in model.parameters():
    p.requires_grad = False
for p in model.text_adapter.parameters():
    p.requires_grad = True
img_params = [p for n, p in model.image_adapter.named_parameters() if "lora_A" not in n]
for p in img_params:
    p.requires_grad = True
optimizer = torch.optim.Adam(
    [{"params": model.text_adapter.parameters()}, {"params": img_params}],
    lr=0.00005, betas=(0.5, 0.999),
)

print("[2/5] loading MVTec train dataset ...", flush=True)
_, image_dataset = get_dataset("MVTec", 518, "full_shot", -1, "train", None)
loader = torch.utils.data.DataLoader(image_dataset, batch_size=2, shuffle=True, num_workers=0)

print("[3/5] running 2 training steps ...", flush=True)
it = iter(loader)
for step in range(2):
    batch = next(it)
    image = batch["image"].to(device)
    mask = batch["mask"].to(device)
    label = batch["label"].to(device)
    class_names = batch["class_name"]
    text_feats = []
    for cn in list(set(class_names)):
        text_feats.append(get_adapted_single_class_text_embedding(model, "MVTec", cn, device))
    epoch_text_feature = torch.stack(text_feats, dim=0)
    patch_features, det_feature, aux_loss, special_loss = model(image)
    det_feature = det_feature.unsqueeze(1)
    cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
    loss = F.cross_entropy(cls_preds, label)
    for f in patch_features:
        patch_preds = calculate_similarity_map(f, epoch_text_feature, 518)
        loss += calculate_seg_loss(patch_preds, mask)
    loss += aux_loss * 0.01 + special_loss * 0.01
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"    step {step}: loss={loss.item():.4f}", flush=True)

print("[4/5] saving checkpoint ...", flush=True)
os.makedirs("ckpt/smoke", exist_ok=True)
torch.save({"text_adapter": model.text_adapter.state_dict(),
            "image_adapter": model.image_adapter.state_dict()},
           "ckpt/smoke/moe_last.pth")

print("[5/5] loading test dataset & running 1 eval batch ...", flush=True)
test_sets = get_dataset("MVTec", 518, "full_shot", -1, "test", None)
ds = test_sets["bottle"]
tloader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(tloader))
with torch.no_grad():
    image = batch["image"].to(device)
    pf, df, _, _ = model(image)
print("SMOKE TEST PASSED: forward/backward/checkpoint/test all OK")