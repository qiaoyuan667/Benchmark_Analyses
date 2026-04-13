import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from adjustText import adjust_text

# plot
# trainer.training_tracker.plot_loss()
# trainer.training_tracker.plot_avg_active()
# trainer.training_tracker.plot_total_active()
# trainer.training_tracker.plot_activation_distribution()
#
# if args.do_val:
#     trainer.val_tracker.plot_loss()
#     trainer.val_tracker.plot_avg_active()
#     trainer.val_tracker.plot_total_active()
#     trainer.val_tracker.plot_activation_distribution()
# if args.do_test:
#     trainer.test_tracker.plot_loss()
#     trainer.test_tracker.plot_avg_active()
#     trainer.test_tracker.plot_total_active()
#     trainer.test_tracker.plot_activation_distribution()


def plot_features(files):
    pass


def plot_trade_off(output_files):
    # -------- 1. read data --------
    models = {}
    for feature_path in output_files:
        fields = os.path.split(feature_path)[-1].split('_')
        hidden_dim = int(fields[3][1:])
        l1 = float(fields[4][1:])

        if hidden_dim not in models:
            models[hidden_dim] = {
                'reconstruction': [],
                'avg_features': [],
                'total_features': [],
                'l1_weight': []
            }

        with open(feature_path, 'r') as fp:
            output = json.load(fp)

        models[hidden_dim]['avg_features'].append(output['stats']['avg_active'])
        models[hidden_dim]['reconstruction'].append(output['stats']['loss_recon'])
        models[hidden_dim]['total_features'].append(output['stats']['total_active'])
        models[hidden_dim]['l1_weight'].append(l1)

    # -------- 2. plot --------
    plt.figure(figsize=(8, 6))

    texts = []  # 👈 收集所有 annotation

    for hidden_dim, data in models.items():
        points = list(zip(
            data['avg_features'],
            data['reconstruction'],
            data['l1_weight'],
            data['total_features']
        ))

        points.sort(key=lambda x: x[0])

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # 折线 + 散点
        plt.plot(xs, ys, marker='o', markersize=4, label=f'h={hidden_dim}')

        # -------- 3. 标注 --------
        for x, y, l1, total in points:
            label = f"L1={l1} | Total_features={total}"

            txt = plt.text(
                x, y, label,
                fontsize=7
            )
            texts.append(txt)

    # -------- 4. 自动调整文字（核心）--------
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        expand_points=(1.2, 1.2),
        expand_text=(1.2, 1.2),
        force_text=0.5,
        force_points=0.3,
        only_move={'points': 'y', 'texts': 'y'}  # 👈 只上下移动更整齐
    )

    # -------- 5. 美化 --------
    plt.xlabel("Average Active Features")
    plt.ylabel("Reconstruction Loss")
    plt.title("Trade-off between Sparsity and Reconstruction")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_trade_off([os.path.join('output/sae_features', filename) for filename in os.listdir('output/sae_features')])
    pass
