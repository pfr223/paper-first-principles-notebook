# Perceptual Quality Report

## Meta

- dataset_jsonl: `server_realtest_v2/dataset_realtest.jsonl`
- orig_setting: `no_defense`
- logo_setting: `safelogo_only`
- num_pairs: `52`
- lpips_enabled: `False`

## Overall

| metric | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| PSNR | 24.8584 | 1.8555 | 21.2665 | 30.6563 |
| SSIM | 0.9538 | 0.0378 | 0.8121 | 0.9939 |
| L2Proxy | 0.0584 | 0.0119 | 0.0293 | 0.0864 |

| pass_rate |
| ---: |
| 3.85 |

## By Split

| split | PSNR mean | SSIM mean | L2 mean | LPIPS mean | pass_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| id | 24.6125 | 0.9416 | 0.0599 | N/A | 0.00 |
| ood | 25.1044 | 0.9660 | 0.0570 | N/A | 7.69 |

