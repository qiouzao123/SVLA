import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from model import SVLA
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option

def _pi_topk_segment(scores_seg: torch.Tensor,
                     pi: torch.Tensor = None,
                     base_div: int = 16,
                     k_min: int = 1) -> torch.Tensor:
    L = scores_seg.numel()
    if L == 0:
        return scores_seg.new_tensor(0.0)

    k_base = max(1, int(L / base_div) + 1)

    if pi is not None:
        pi_val = float(pi.detach().clamp(1e-3, 0.9).item())
        pi0 = 0.22
        gamma = 2.0
        scale = 1.0 + gamma * (pi_val - pi0)
        scale = max(0.5, min(2.0, scale))
        k = int(round(k_base * scale))
    else:
        k = k_base

    k = max(k_min, k)
    k = min(L, k)
    k = max(1, k)

    topk_vals, _ = torch.topk(scores_seg, k=k, largest=True)
    return topk_vals.mean()

def CLAS2_dasmil_weighted(logits, labels, lengths, shot_slices, shot_pi_list,
                          device, base_div: int = 16, k_min: int = 1):
    B, T, _ = logits.shape
    labels_bin = 1 - labels[:, 0].reshape(labels.shape[0]).to(device)
    probs = torch.sigmoid(logits).reshape(B, T)

    instance_logits = []

    for i in range(B):
        Li = int(lengths[i].item())
        if Li <= 0:
            instance_logits.append(probs.new_tensor(0.0).unsqueeze(0))
            continue

        shots = shot_slices[i] if (shot_slices is not None and i < len(shot_slices)) else [(0, Li)]
        pis   = shot_pi_list[i] if (shot_pi_list is not None and i < len(shot_pi_list)) else None

        shot_vals = []
        used_pis  = []

        for si, (s, e) in enumerate(shots):
            s = max(0, min(s, Li))
            e = max(0, min(e, Li))
            if e <= s:
                continue

            seg = probs[i, s:e]
            pi  = pis[si] if (pis is not None and si < len(pis)) else None

            pooled = _pi_topk_segment(seg, pi, base_div=base_div, k_min=k_min)
            shot_vals.append(pooled)
            if pi is not None:
                used_pis.append(pi)

        if len(shot_vals) > 0:
            shot_scores = torch.stack(shot_vals, dim=0)
            if len(used_pis) == len(shot_vals) and len(used_pis) > 0:
                pis_used = torch.stack(used_pis, dim=0)
                w = pis_used / (pis_used.sum() + 1e-6)
                val = (shot_scores * w).sum()
            else:
                val = shot_scores.mean()
        else:
            val = probs[i, :Li].mean()

        instance_logits.append(val.unsqueeze(0))

    instance_logits = torch.cat(instance_logits, dim=0)
    clsloss = F.binary_cross_entropy(instance_logits, labels_bin)
    return clsloss

def CLASM_dasmil_weighted(logits, labels, lengths, shot_slices, shot_pi_list,
                          device, base_div: int = 16, k_min: int = 1):
    B, T, C = logits.shape
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    instance_logits = []

    for i in range(B):
        Li = int(lengths[i].item())
        if Li <= 0:
            instance_logits.append(logits.new_zeros(1, C))
            continue

        shots = shot_slices[i] if (shot_slices is not None and i < len(shot_slices)) else [(0, Li)]
        pis   = shot_pi_list[i] if (shot_pi_list is not None and i < len(shot_pi_list)) else None

        shot_vecs = []
        used_pis  = []

        for si, (s, e) in enumerate(shots):
            s = max(0, min(s, Li))
            e = max(0, min(e, Li))
            if e <= s:
                continue

            seg_logits = logits[i, s:e, :]
            pi = pis[si] if (pis is not None and si < len(pis)) else None

            pooled_per_class = []
            for c in range(C):
                seg_c = seg_logits[:, c]
                pooled_c = _pi_topk_segment(seg_c, pi, base_div=base_div, k_min=k_min)
                pooled_per_class.append(pooled_c)
            shot_vec = torch.stack(pooled_per_class, dim=0)
            shot_vecs.append(shot_vec)
            if pi is not None:
                used_pis.append(pi)

        if len(shot_vecs) > 0:
            shot_mat = torch.stack(shot_vecs, dim=0)
            if len(used_pis) == len(shot_vecs) and len(used_pis) > 0:
                pis_used = torch.stack(used_pis, dim=0).view(-1, 1)
                w = pis_used / (pis_used.sum() + 1e-6)
                vid_vec = (shot_mat * w).sum(dim=0)
            else:
                vid_vec = shot_mat.mean(dim=0)
        else:
            vid_vec = logits[i, :Li, :].mean(dim=0)

        instance_logits.append(vid_vec.unsqueeze(0))

    instance_logits = torch.cat(instance_logits, dim=0)
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def text_feature_regularizer(text_features: torch.Tensor, weight: float = 1e-1, eps: float = 1e-12):
    if text_features is None or text_features.ndim != 2 or text_features.size(0) < 2:
        device = text_features.device if isinstance(text_features, torch.Tensor) else 'cpu'
        return torch.zeros(1, device=device)
    tf = text_features
    tf = tf / (tf.norm(dim=-1, keepdim=True) + eps)

    normal = tf[0]
    others = tf[1:]
    cos = torch.matmul(others, normal)
    loss = torch.abs(cos).mean() * weight
    return loss

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    prompt_text = get_prompt_text(label_map)

    base_lr = args.lr
    pi_lr_mult = getattr(args, 'pi_lr_mult', 5.0)
    pi_wd = getattr(args, 'pi_weight_decay', 0.0)
    base_wd = getattr(args, 'weight_decay', 0.01)

    pi_params = list(model.shot_density_head.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('shot_density_head')]

    optimizer = torch.optim.AdamW(
        [
            {'params': other_params, 'lr': base_lr, 'weight_decay': base_wd},
            {'params': pi_params,    'lr': base_lr * pi_lr_mult, 'weight_decay': pi_wd},
        ]
    )
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)

    best_auc1 = best_ap1 = best_auc2 = best_ap2 = 0.0
    start_epoch = 1
    if getattr(args, 'use_checkpoint', False) and os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception:
                pass
        last_epoch = ckpt.get('epoch', 0)
        start_epoch = max(1, int(last_epoch) + 1)
        best_auc1 = ckpt.get('best_auc1', 0.0)
        best_ap1 = ckpt.get('best_ap1', 0.0)
        best_auc2 = ckpt.get('best_auc2', 0.0)
        best_ap2 = ckpt.get('best_ap2', 0.0)
        print(f"[resume] start_epoch={start_epoch} (last_epoch={last_epoch}) | "
              f"best AUC1={best_auc1:.4f}, AP1={best_ap1:.4f}, "
              f"AUC2={best_auc2:.4f}, AP2={best_ap2:.4f}")

    gt = np.load(args.gt_path)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    max_epoch = int(getattr(args, 'max_epoch', 10))
    print_every = int(getattr(args, 'print_every', 20))

    txtreg_weight = float(getattr(args, 'txtreg_weight', 1e-1))

    pi_topk_base_div = int(getattr(args, 'pi_topk_base_div', 16))
    pi_topk_k_min = int(getattr(args, 'pi_topk_k_min', 1))

    from ucf_test import test as test_fn

    for epoch_idx in range(start_epoch, max_epoch + 1):
        model.train()
        loss_total1 = loss_total2 = loss_total3 = 0.0

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_iterations = min(len(normal_loader), len(anomaly_loader))

        for i in range(num_iterations):
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels_raw = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)

            text_list = get_prompt_text(label_map)
            text_labels = get_batch_label(text_labels_raw, text_list, label_map).to(device)

            text_features_ori, logits1_full, logits2, shot_slices, _ = model(
                visual_features, None, text_list, feat_lengths
            )
            shot_pi_list = getattr(model, "_last_shot_pi_list", None)

            loss1 = CLAS2_dasmil_weighted(
                logits1_full, text_labels, feat_lengths, shot_slices, shot_pi_list,
                device=device,
                base_div=pi_topk_base_div, k_min=pi_topk_k_min
            )
            loss2 = CLASM_dasmil_weighted(
                logits2, text_labels, feat_lengths, shot_slices, shot_pi_list,
                device=device,
                base_div=pi_topk_base_div, k_min=pi_topk_k_min
            )

            loss3 = text_feature_regularizer(text_features_ori, weight=txtreg_weight)

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_total3 += loss3.item()

            step_print = i + 1
            if (step_print % print_every == 0) or (step_print == num_iterations):
                pi_mean = getattr(model, "_dbg_pi_mean", float('nan'))
                pi_min = getattr(model, "_dbg_pi_min", float('nan'))
                pi_max = getattr(model, "_dbg_pi_max", float('nan'))
                avg_shots = 0.0
                if shot_slices:
                    try:
                        import numpy as _np
                        avg_shots = float(_np.mean([len(s) for s in shot_slices]))
                    except Exception:
                        avg_shots = 0.0

                print(f"epoch: {epoch_idx:3d} | step: {step_print:3d} | "
                      f"loss_bin: {loss_total1/step_print:.6f} | "
                      f"loss_cls: {loss_total2/step_print:.6f} | "
                      f"loss_txtreg: {loss_total3/step_print:.6f}")
                print(f"[dbg] avg_shots_per_segment={avg_shots:.2f}")
                print(f"[dbg] pi_mean={pi_mean:.3f} (min={pi_min:.3f}, max={pi_max:.3f})")

                auc1, ap1, auc2, ap2 = test_fn(
                    model, testloader, args.visual_length, get_prompt_text(label_map),
                    gt, gtlabels, device, label_map, args
                )

                improved = False
                if auc1 > best_auc1:
                    best_auc1 = auc1
                    improved = True
                if ap1 > best_ap1:
                    best_ap1 = ap1
                    improved = True
                if auc2 > best_auc2:
                    best_auc2 = auc2
                    improved = True
                if ap2 > best_ap2:
                    best_ap2 = ap2
                    improved = True

                print(f"Validation -> Current AUC1: {auc1:.4f} (Best: {best_auc1:.4f}) | "
                      f"Current AP1: {ap1:.4f} (Best: {best_ap1:.4f})")
                print(f"              Current AUC2: {auc2:.4f} (Best: {best_auc2:.4f}) | "
                      f"Current AP2: {ap2:.4f} (Best: {best_ap2:.4f})")

                if improved:
                    checkpoint = {
                        'epoch': epoch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_auc1': best_auc1,
                        'best_ap1': best_ap1,
                        'best_auc2': best_auc2,
                        'best_ap2': best_ap2
                    }
                    ckpt_dir = os.path.dirname(args.checkpoint_path)
                    if ckpt_dir and not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    torch.save(checkpoint, args.checkpoint_path)

        scheduler.step()

        cur_dir = os.path.dirname(args.model_path) or 'model'
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        torch.save(model.state_dict(), os.path.join(cur_dir, 'model_cur.pth'))

    print("Training finished. Saving the best model...")
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        torch.save(checkpoint['model_state_dict'], args.model_path)
        print(f"Best model saved to {args.model_path}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({
        'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
        'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
        'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
        'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'
    })

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, normal=True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size,
                               shuffle=True, drop_last=True)

    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, normal=False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size,
                                shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = SVLA(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
        shot_sim_thresh=args.shot_sim_thresh,
        shot_min_len=args.shot_min_len,
        shot_layers=args.shot_layers,
        shot_gamma=args.shot_gamma,
        pi_floor=args.pi_floor,
        cfa_tau=args.cfa_tau, cfa_beta=args.cfa_beta,
        cfa_prefix_len=args.cfa_prefix_len,
        cfa_bottleneck=args.cfa_bottleneck,
        cfa_prefix_rank=args.cfa_prefix_rank,
        cfa_dropout=args.cfa_dropout
    )

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)