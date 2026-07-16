import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

logdir = "Oct09_16-59-34_stresstestm"

# 加载日志
event_acc = EventAccumulator(logdir)
event_acc.Reload()

# 读取训练和验证 loss
train_loss = event_acc.Scalars('train/loss')
eval_loss = event_acc.Scalars('eval/loss') if 'eval/loss' in event_acc.Tags()['scalars'] else []

# 绘制曲线
plt.figure(figsize=(8,5))
if train_loss:
    plt.plot([x.step for x in train_loss], [x.value for x in train_loss], label='Train Loss')
if eval_loss:
    plt.plot([x.step for x in eval_loss], [x.value for x in eval_loss], label='Eval Loss')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
