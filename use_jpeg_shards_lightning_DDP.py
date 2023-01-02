
from multiprocessing.managers import SyncManager
from pathlib import Path
from PIL import Image
from torch import optim, nn
from torchvision import transforms, models
import argparse
import io
import json
import pytorch_lightning as pl
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch
import webdataset as wds
import torch.distributed as dist
import multiprocessing
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore', UserWarning)

# https://bugs.python.org/issue7503
# https://stackoverflow.com/questions/28318502/pythonusing-multiprocessing-manager-in-process-pool
multiprocessing.current_process().authkey = b'this is the key'


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

    def forward(self, im, batch_dic, lock):
        bs = im.shape[0]
        gpu_id = im.get_device()
        gpu_id = torch.tensor([gpu_id] * bs, device=im.device)

        with lock:
            print('|-GPU------', gpu_id, '-----------------')
            print('| worker id', batch_dic['read worker id'])
            print('| shard    ', batch_dic['url'])
            print('| count    ', batch_dic['count'])
            # print('| label    ', batch_dic['label'])

        return self.model(im), gpu_id


class MyProgressBar(TQDMProgressBar):
    # https://github.com/Lightning-AI/lightning/blob/f576ed3bbda95a5045edacc49146a3f1cdcd892a/src/pytorch_lightning/callbacks/progress/base.py#L234
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop('v_num', None)
        return items


class MyLightningModel(pl.LightningModule):
    def __init__(self, model, lock, args):
        super().__init__()
        self.model = model
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.lock_list = [lock]
        self.is_lock_set = False

    def forward(self, im):
        output = self.model(im)
        return output

    def set_lock(self):
        if self.is_lock_set:
            return

        self.rank = dist.get_rank()
        dist.broadcast_object_list(self.lock_list, src=0, device=None)
        self.lock = self.lock_list[0]  # shared lock for DDP
        tqdm.set_lock(self.lock)  # global lock of tqdm in lightning
        self.is_lock_set = True

    def training_step(self, batch, batch_idx):
        self.set_lock()

        im, batch_dic, urls = batch
        label = batch_dic['label']

        batch_dic['url'] = urls
        gpu_id = im.get_device()

        with self.lock:
            print('==========================')
            print(f'loop {batch_idx} on GPU {gpu_id}:')
            print('worker id', batch_dic['read worker id'])
            print('shard    ', batch_dic['url'])
            print('count    ', batch_dic['count'])
            # print('label    ', batch_dic['label'])

            output, gpu_id = self.model(im, batch_dic, self.lock)

            print('proc GPU ', gpu_id)

        loss = self.criterion(output, label)

        top1 = accuracy(output, label)
        self.log('train top1', top1, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=args.lr, betas=args.betas)
        return optimizer


def add_worker_id(sample):
    info = torch.utils.data.get_worker_info()
    sample['read worker id'] = info.id
    sample['rank'] = dist.get_rank()
    sample['world size'] = dist.get_world_size()
    return sample


def make_dataset(
    shards_url,
    batch_size,
    shuffle_buffer_size=-1,
    transform=None,
):

    dataset = wds.WebDataset(
        shards_url,
        nodesplitter=wds.split_by_node
    )
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.decode('torchrgb')  # jpg --> tensor(uint8, CHW)
    dataset = dataset.to_tuple(
        'jpg',
        'json',
        '__url__',
    )
    dataset = dataset.map_tuple(
        lambda x: transform(x) if transform is not None else x,
        add_worker_id,
        lambda x: int(x.split('.')[0].split('-')[-1]),  # 'test-00.tar' --> 0
    )
    dataset = dataset.batched(
        batch_size,
        partial=False)

    return dataset


def my_collate_fn(batch):
    ret = (
        batch[0],  # 'jpg', already BCHW because of dataset.batched()
        torch.utils.data.default_collate(batch[1]),  # 'json'
        torch.utils.data.default_collate(batch[2]),  # '__url__'
    )
    return ret


def main(args):

    assert torch.cuda.is_available(), 'cpu is not supported'
    assert isinstance(args.gpu, list), 'single gpu is not supported'

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
    num_batches = dataset_size // (args.batch_size * len(args.gpu))

    sample_loader.length = num_batches
    sample_loader = sample_loader.with_length(num_batches)

    sample_loader = sample_loader.repeat(nbatches=num_batches)
    sample_loader = sample_loader.slice(num_batches)

    model = MyModel(num_classes=num_classes)

    with SyncManager() as manager:

        lock = manager.RLock()
        model_lightning = MyLightningModel(model, lock, args)

        trainer = pl.Trainer(
            devices=args.gpu,
            accelerator='gpu',
            strategy='ddp',  # 'ddp_find_unused_parameters_false',
            max_epochs=args.n_epochs,
            callbacks=[
                MyProgressBar(),
            ])
        trainer.fit(
            model=model_lightning,
            train_dataloaders=sample_loader)


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
