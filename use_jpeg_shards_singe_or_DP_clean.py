
from pathlib import Path
from PIL import Image
from torch import optim, nn
from torchvision import transforms, models
from tqdm import tqdm
import argparse
import io
import json
import torch
import webdataset as wds

import warnings
warnings.simplefilter('ignore', UserWarning)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L363-L380
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, bs=1):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.value = value
        self.sum += value * bs
        self.count += bs
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res if len(res) > 1 else res[0]


def info_from_json(shard_path):
    json_file = Path(shard_path).glob('*.json')
    json_file = str(next(json_file))  # get the first json file
    with open(json_file, 'r') as f:
        info_dic = json.load(f)

    return info_dic['dataset size'], info_dic['num_classes']


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda x: x / 255.),  # already tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, im):
        return self.model(im)


def make_dataset(
    shards_url,
    batch_size,
    shuffle_buffer_size=-1,
    transform=None,
):

    dataset = wds.WebDataset(shards_url)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.decode('torchrgb')  # jpg --> tensor(uint8, CHW)
    dataset = dataset.to_tuple(
        'jpg',
        'json',
    )
    dataset = dataset.map_tuple(
        lambda x: transform(x) if transform is not None else x,
        lambda x: x['label']
    )
    dataset = dataset.batched(
        batch_size,
        partial=False)

    return dataset


def my_collate_fn(batch):
    ret = (
        batch[0],  # 'jpg', already BCHW because of dataset.batched()
        torch.utils.data.default_collate(batch[1]),  # label
    )
    return ret


def main(args):

    assert torch.cuda.is_available(), 'cpu is not supported'
    if isinstance(args.gpu, int):
        device = torch.device('cuda:' + str(args.gpu))
    elif isinstance(args.gpu, list):
        device = torch.device('cuda:' + str(args.gpu[0]))  # the 1st device

    shards_path = [
        str(path) for path in Path(args.shard_path).glob('*.tar')
        if not path.is_dir()
    ]

    transform = get_transform()

    dataset = make_dataset(
        shards_url=shards_path,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle,
        transform=transform)
    sample_loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=my_collate_fn)

    dataset_size, num_classes = info_from_json(args.shard_path)
    num_batches = dataset_size // args.batch_size + 1

    sample_loader.length = num_batches
    sample_loader = sample_loader.with_length(num_batches)

    model = MyModel(num_classes=num_classes)
    if isinstance(args.gpu, list):
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, betas=args.betas)

    train_loss = AverageMeter()
    train_top1 = AverageMeter()

    with tqdm(range(args.n_epochs)) as pbar_epoch:

        for epoch in pbar_epoch:
            pbar_epoch.set_description("[Train] epoch: %d" % epoch)

            train_loss.reset()
            train_top1.reset()

            with tqdm(enumerate(sample_loader),
                      total=sample_loader.length,
                      leave=True,
                      smoothing=0,
                      ) as pbar_batch:

                for i, batch in pbar_batch:

                    im, label = batch
                    im = im.to(device)
                    label = label.to(device)

                    optimizer.zero_grad()

                    output = model(im)

                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()

                    bs = im.size(0)
                    train_loss.update(loss, bs)
                    train_top1.update(accuracy(output, label), bs)

                    pbar_batch.set_postfix_str(
                        ' loss={:6.04f}/{:6.04f}'
                        ' top1={:6.04f}/{:6.04f}'
                        ''.format(
                            train_loss.value, train_loss.avg,
                            train_top1.value, train_top1.avg,
                        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--shard_path', action='store',
                        default='./test_shards/',
                        help='Path to the dir to store shard *.tar files.')
    parser.add_argument('--shuffle', type=int, default=-1,
                        help='shuffle buffer size. negative means no shuffle. '
                        'default -1')

    parser.add_argument('-b', '--batch_size', type=int, default=3,
                        help='batch size. default 3')
    parser.add_argument('-w', '--num_workers', type=int, default=2,
                        help='number of dataloader workders. default 2')
    parser.add_argument('-g', '--gpu', nargs='+', type=int, default=0,
                        help='GPU ids to be used. '
                        'int ("0", "1") or list of int ("1 2", "0 1 2"). '
                        'default "0"')

    parser.add_argument('--n_epochs', type=int, default=10,
                        help='number of epochs. default to 10')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='learning rate. default to 0.0001')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999],
                        help='betas of Adam. default to (0.9, 0.999).'
                        'specify like --betas 0.9 0.999')

    args = parser.parse_args()
    print(args)
    main(args)
