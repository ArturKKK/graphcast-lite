#!/usr/bin/env python3
"""Verify temporal alignment between global test split and regional dataset."""
from datetime import datetime, timedelta

n_time = 32  # global Jan 2023 timesteps
obs, pred = 2, 4
total = n_time - obs - pred + 1  # 27
split_idx = int(total * 0.8)  # 21
test_all = list(range(split_idx, total))
val_size = len(test_all) // 2
test_only = test_all[val_size:]

print(f"Total samples: {total}, split_idx={split_idx}")
print(f"Test+val indices: {test_all}")
print(f"Test only indices: {test_only}")

time_offset = 32  # global starts Jan18, regional starts Jan10 = 32 six-hour steps

for t in test_only:
    pred_start_g = t + obs
    pred_end_g = t + obs + pred - 1
    pred_start_r = pred_start_g + time_offset
    pred_end_r = pred_end_g + time_offset
    base_r = (t + obs - 1) + time_offset

    g0 = datetime(2023, 1, 18)
    obs_d = g0 + timedelta(hours=t * 6)
    ps_d = g0 + timedelta(hours=pred_start_g * 6)
    pe_d = g0 + timedelta(hours=pred_end_g * 6)

    print(f"  sample local_t={t}: obs={obs_d:%m/%d %H:%M}, "
          f"pred={ps_d:%m/%d %H:%M}-{pe_d:%m/%d %H:%M}  "
          f"-> regional t={pred_start_r}-{pred_end_r} (base={base_r})")

print(f"Regional has 72 timesteps (0-71). Max pred t={pred_end_r} valid: {pred_end_r < 72}")
