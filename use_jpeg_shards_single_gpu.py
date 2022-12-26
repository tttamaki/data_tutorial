
import torch.nn as nn
import torch
from tqdm import tqdm
import argparse
import json
import webdataset as wds
from pathlib import Path
from multiprocessing.managers import SyncManager
from PIL import Image
import io
from torchvision import transforms, models
from torch import optim


def info_from_json(shard_path):
    json_file = Path(shard_path).glob('*.json')
    json_file = str(next(json_file))  # get the first json file
    with open(json_file, 'r') as f:
        info_dic = json.load(f)

    return info_dic['dataset size'], info_dic['num_classes']


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
            print('| GPU      ', gpu_id)
            print('| worker id', batch_dic['read worker id'])
            print('| shard    ', batch_dic['url'])
            print('| count    ', batch_dic['count'])

        return self.model(im), gpu_id


def add_worker_id(sample):
    info = torch.utils.data.get_worker_info()
    sample['read worker id'] = info.id
    return sample


def my_jpg_decoder(sample):
    img = Image.open(io.BytesIO(sample))
    return img


def make_dataset(
    shards_url,
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
        '__url__',
    )
    dataset = dataset.map_tuple(
        lambda x: transform(x) if transform is not None else x,
        add_worker_id,
        lambda x: x,
    )

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
    assert isinstance(args.gpu, int), 'multi gpu is not supported'
    device = torch.device('cuda:' + str(args.gpu))

    shards_path = [
        str(path) for path in Path(args.shard_path).glob('*.tar')
        if not path.is_dir()
    ]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda x: x / 255.),  # already tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = make_dataset(
        shards_url=shards_path,
        shuffle_buffer_size=args.shuffle,
        transform=transform)
    dataset = dataset.batched(
        args.batch_size,
        partial=False)
    sample_loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=my_collate_fn)

    dataset_size, num_classes = info_from_json(args.shard_path)
    num_batches = dataset_size // (args.batch_size * args.num_workers)
    print("# batches per worker = ", num_batches)
    sample_loader.length = num_batches * args.num_workers
    sample_loader = sample_loader.repeat(2).slice(sample_loader.length)

    model = MyModel(num_classes=num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, betas=args.betas)

    with tqdm(range(args.n_epochs)) as pbar_epoch, \
            SyncManager() as manager:

        lock = manager.Lock()

        for epoch in pbar_epoch:
            pbar_epoch.set_description("[Train] epoch: %d" % epoch)

            with tqdm(enumerate(sample_loader),
                      total=sample_loader.length,
                      leave=True,
                      smoothing=0,
                      ) as pbar_batch:

                for i, batch in pbar_batch:

                    im, batch_dic, urls = batch
                    im = im.to(device)
                    label = batch_dic['label'].to(device)

                    batch_dic['url'] = urls
                    gpu_id = im.get_device()

                    with lock:
                        print(f'\n{i}-th loop on GPU {gpu_id}:')
                        print('worker id', batch_dic['read worker id'])
                        print('shard    ', batch_dic['url'])
                        print('count    ', batch_dic['count'])

                    output, gpu_id = model(im, batch_dic, lock)
                    print('GPU      ', gpu_id)

                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()


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
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='GPU No. to be used for model. default 0')

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
