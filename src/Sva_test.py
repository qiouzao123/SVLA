import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from model import SVLA
from utils.dataset import SvaDataset
from utils.tools import get_batch_mask, get_prompt_text
import Sva_option

def test(model, testdataloader, maxlen, prompt_text, gt, gtlabels, device, label_map=None, args=None):
    repeat_factor = 16 

    model.eval()
    ap1_list, ap2_list = [], []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0]
            length = int(item[2])
            len_cur = length

            if visual.dim() == 4:
                visual = visual.squeeze(0)
            visual = visual.to(device)

            num_segs = visual.shape[0]
            lengths = torch.zeros(num_segs, dtype=torch.long, device=device)
            remain = length
            for j in range(num_segs):
                take = min(maxlen, remain)
                lengths[j] = take
                remain -= take

            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            _, logits1, logits2, shot_slices, _ = model(visual, padding_mask, prompt_text, lengths)

            prob1 = torch.sigmoid(
                logits1.reshape(-1, 1)[:len_cur].squeeze(-1)
            )
            prob2 = (
                1.0
                - F.softmax(
                    logits2.reshape(-1, logits2.shape[-1])[:len_cur],
                    dim=-1,
                )[:, 0]
            )

            ap1_list.extend(prob1.cpu().numpy().tolist())
            ap2_list.extend(prob2.cpu().numpy().tolist())

    ap1_scores = np.array(ap1_list)
    ap2_scores = np.array(ap2_list)

    final_scores_1 = np.repeat(ap1_scores, repeat_factor)
    final_scores_2 = np.repeat(ap2_scores, repeat_factor)

    gt_len = len(gt)
    if len(final_scores_1) > gt_len:
        final_scores_1 = final_scores_1[:gt_len]
    elif len(final_scores_1) < gt_len:
        final_scores_1 = np.pad(final_scores_1, (0, gt_len - len(final_scores_1)), "constant")

    if len(final_scores_2) > gt_len:
        final_scores_2 = final_scores_2[:gt_len]
    elif len(final_scores_2) < gt_len:
        final_scores_2 = np.pad(final_scores_2, (0, gt_len - len(final_scores_2)), "constant")

    auc1 = roc_auc_score(gt, final_scores_1)
    ap1 = average_precision_score(gt, final_scores_1)
    auc2 = roc_auc_score(gt, final_scores_2)
    ap2 = average_precision_score(gt, final_scores_2)
    
    print(f"AUC1: {auc1:.4f}, AP1: {ap1:.4f}")
    print(f"AUC2: {auc2:.4f}, AP2: {ap2:.4f}")
    return auc1, ap1, auc2, ap2

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = Sva_option.parser.parse_args()

    label_map = dict(
        {
            "normal": "normal",
            "abusive": "abusive",
            "blood": "blood",
            "money": "money",
            "policy": "policy",
            "sexy": "sexy",
            "smoke": "smoke",
            "violent": "violent",
        }
    )

    test_dataset = SvaDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = SVLA(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
        shot_sim_thresh=args.shot_sim_thresh,
        shot_min_len=args.shot_min_len,
        shot_layers=args.shot_layers,
        shot_gamma=args.shot_gamma,
        pi_floor=args.pi_floor,
        cfa_tau=args.cfa_tau,
        cfa_beta=args.cfa_beta,
        cfa_prefix_len=args.cfa_prefix_len,
        cfa_bottleneck=args.cfa_bottleneck,
        cfa_prefix_rank=args.cfa_prefix_rank,
        cfa_dropout=args.cfa_dropout,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    auc1, ap1, auc2, ap2 = test(
        model,
        test_loader,
        args.visual_length,
        prompt_text,
        gt,
        gtlabels,
        device,
        label_map,
        args,
    )

    print("\n--- Final Test Results ---")
    print(f"AUC1: {auc1:.4f}, AP1: {ap1:.4f}")
    print(f"AUC2: {auc2:.4f}, AP2: {ap2:.4f}")