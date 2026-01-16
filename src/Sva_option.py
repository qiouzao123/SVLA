import argparse

parser = argparse.ArgumentParser(description='SVA')

parser.add_argument('--seed', default=318, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=8, type=int)

parser.add_argument('--shot-sim-thresh', default=0.90, type=float)
parser.add_argument('--shot-min-len', default=12, type=int)
parser.add_argument('--shot-layers', default=1, type=int)
parser.add_argument('--shot-gamma', default=0.05, type=float)

parser.add_argument('--pi-floor', default=0.05, type=float)
parser.add_argument('--pi-prior-target', default=0.15, type=float)

parser.add_argument('--cfa-tau', default=0.8, type=float)
parser.add_argument('--cfa-beta', default=0.8, type=float)
parser.add_argument('--cfa-prefix-len', default=32, type=int)
parser.add_argument('--cfa-bottleneck', default=256, type=int)
parser.add_argument('--cfa-prefix-rank', default=16, type=int)
parser.add_argument('--cfa-dropout', default=0.1, type=float)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_sva.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint_sva.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default='list/Sva_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/Sva_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_Sva_500_abnormal_is_1.npy')
parser.add_argument('--gt-label-path', default='list/gt_Sva_small.npy')

parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--weight-decay', default=0.01, type=float)
parser.add_argument('--pi-lr-mult', default=5.0, type=float)

parser.add_argument('--scheduler-rate', default=0.1, type=float)
parser.add_argument('--scheduler-milestones', default=[4, 8], type=int, nargs='+')

parser.add_argument('--txtreg-weight', default=1e-1, type=float)
parser.add_argument('--print-every', default=10, type=int)

parser.add_argument('--pi-topk-base-div', default=16, type=int)
parser.add_argument('--pi-topk-k-min', default=1, type=int)