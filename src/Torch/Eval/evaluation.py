from typing import Any, Tuple
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> Tuple[float]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res


class EvaluateModel:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.model = model
        self.model.eval()
        self.dataset = dataset

    def measure_accuracy(self, evaluation_steps: int) -> float:
        top1 = AverageMeter()
        with torch.no_grad():
            for idx, (data) in enumerate(self.dataset):
                if evaluation_steps is not None:
                    if evaluation_steps == idx:
                        break
                input = data["images"]
                target = data["labels"]
                input = input.float()
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()
                output = self.model(input)

                acc1 = accuracy(output, target, topk=(1,))[0]
                top1.update(acc1.item(), input.size(0))

        return top1.avg

    def __call__(
        self,
        layer_to_activate: str,
        evaluation_steps: int = None,
        *args: Any,
        **kwds: Any,
    ) -> float:
        for name, layer in self.model.named_modules():
            if name == layer_to_activate:
                target_layer = layer
                target_layer.bitswap_coefficient = True
                break
        accuracy = self.measure_accuracy(
            evaluation_steps=evaluation_steps,
        )
        target_layer.bitswap_coefficient = False
        return accuracy
