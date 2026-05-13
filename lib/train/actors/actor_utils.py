import torch
import copy

import copy
import torch

def initialized_teacher_model(net):
    ckpt_path = '/Tracker/MVLM/pretrained/MVLM_ep0180.pth.tar'
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Get state_dict from ckpt + prefix normalization
    raw_sd = ckpt.get("state_dict", ckpt.get("net", ckpt))
    ckpt_sd = {(k if k.startswith("module.") else f"module.{k}"): v for k, v in raw_sd.items()}

    # Filter only keys with matching shapes based on current model
    base_state = net.state_dict()
    loadable_sd = {k: v for k, v in ckpt_sd.items() if k in base_state and v.shape == base_state[k].shape}

    # Create Teacher (clone student)
    teacher_model = copy.deepcopy(net)

    # === Before snapshot ===
    before = {k: v.detach().cpu().clone() for k, v in teacher_model.state_dict().items()}

    # Load the checkpoint 
    missing, unexpected = teacher_model.load_state_dict(loadable_sd, strict=False)

    # === After snapshot ===
    after = {k: v.detach().cpu().clone() for k, v in teacher_model.state_dict().items()}

    # Freeze parameters & set to eval mode
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    # === Before/After comparison: based on loadable keys ===
    changed, unchanged = [], []
    for k in loadable_sd.keys():
        if torch.allclose(before[k].float(), after[k].float(), atol=1e-7, rtol=1e-5):
            unchanged.append(k)
        else:
            changed.append(k)

    # Print summary
    total = len(loadable_sd)
    num_changed = len(changed)
    num_unchanged = len(unchanged)

    print("=== Teacher Model Load Verification ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Loadable keys: {total}")
    print(f"Changed keys after load: {num_changed}")
    print(f"Unchanged keys after load: {num_unchanged}")

    if num_changed == total:
        print("✅ All loadable checkpoint parameters successfully loaded and changed.")
    else:
        print(f"⚠️ {num_unchanged} keys did not change after load. (Possible identical init values or loading issue)")
    if missing:
        print(f"[Missing] {len(missing)} keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[Unexpected] {len(unexpected)} keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print("=======================================")

    return teacher_model


def get_jittered_box(boxes):
    """ Jitter the input box
    args:
        box - input bounding box
    returns:
        torch.Tensor - jittered box
    """
    jittered_box_list = []
    device = boxes.device
    for box in boxes:
        scale_jitter_factor = 0.25
        center_jitter_factor = 0.25
        jittered_size = box[2:4] * torch.exp(torch.randn(2, device=device) * scale_jitter_factor)
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(center_jitter_factor, device=device).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2, device=device) - 0.5)
        jittered_box = torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0).unsqueeze(0)
        jittered_box_list.append(jittered_box)
    jittered_boxes = torch.cat(jittered_box_list, dim=0)
    return jittered_boxes

def get_jittered_box_1(box):
    """ Jitter the input box
    args:
        box - input bounding box
    returns:
        torch.Tensor - jittered box
    """
    device = box.device
    scale_jitter_factor = 0.25
    center_jitter_factor = 0.5
    jittered_size = box[2:4] * torch.exp(torch.randn(2, device=device) * scale_jitter_factor)
    max_offset = (jittered_size.prod().sqrt() * torch.tensor(center_jitter_factor, device=device).float())
    jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2, device=device) - 0.5)
    jittered_box = torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)
    return jittered_box