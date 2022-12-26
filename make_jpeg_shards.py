from functools import partial
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import os
import random
from webdataset import ShardWriter
from multiprocessing import Pool, current_process
from multiprocessing.managers import SyncManager


class MyShardWriter(ShardWriter):

    def __init__(self, pattern, maxcount=100000, maxsize=3e9, post=None, start_shard=0, **kw):
        super().__init__(pattern, maxcount, maxsize, post, start_shard)
        self.verbose = False

    def get_shards(self):
        return self.shard

    def get_count(self):
        return self.count

    def get_total(self):
        return self.total


class MyManager(SyncManager):
    pass


def worker(file_path, lock, pbar, sink, class_to_idx):

    #
    # process
    #

    # when file_path == 'dataset/cats_dogs/PetImages/Dog/10247.jpg'
    category_name = file_path.parent.name  # 'Dog': str
    label = class_to_idx[category_name]  # 1: int
    key_str = category_name + '/' + file_path.stem  # 'Dog/10247': str

    with open(str(file_path), 'rb') as raw_bytes:
        buffer = raw_bytes.read()

    #
    # write
    #

    with lock:

        sample_dic = {
            '__key__': key_str,
            'json': json.dumps({
                'write worker id': current_process().name,
                'count': sink.get_count(),
                'category': category_name,
                'label': label,
            }),
            'jpg': buffer
        }
        sink.write(sample_dic)

        pbar.update(1)
        pbar.set_postfix_str(
            f'shard {sink.get_shards()} '
            f'worker {current_process().name[-1]}'
        )


def make_shards(args):

    file_paths = [
        path for path in Path(args.data_path).glob('**/*')
        if not path.is_dir()
    ]
    if args.shuffle:
        random.shuffle(file_paths)
    n_samples = len(file_paths)

    # https://github.com/pytorch/vision/blob/a8bde78130fd8c956780d85693d0f51912013732/torchvision/datasets/folder.py#L36
    class_list = sorted(
        entry.name for entry in os.scandir(args.data_path)
        if entry.is_dir())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_list)}

    shard_dir_path = Path(args.shard_path)
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / f'{args.shard_prefix}-%05d.tar')

    # https://qiita.com/tttamaki/items/96b65e6555f9d255ffd9
    MyManager.register('Tqdm', tqdm)
    MyManager.register('Sink', MyShardWriter)

    with MyManager() as manager:

        #
        # prepare manager objects
        #

        lock = manager.Lock()
        pbar = manager.Tqdm(
            total=n_samples,
            position=0,
        )
        pbar.set_description('Main process')
        sink = manager.Sink(
            pattern=shard_filename,
            maxsize=args.max_size,
            maxcount=args.max_count,
        )

        #
        # create worker pool
        #

        worker_with_args = partial(
            worker, lock=lock, pbar=pbar, sink=sink,
            class_to_idx=class_to_idx
        )
        with Pool(processes=args.num_workers) as pool:
            # https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
            # for _ in pool.imap_unordered(
            #         worker_with_args,
            #         file_paths,
            #         chunksize=n_samples // args.num_workers
            # ):
            #     pass
            pool.map(
                worker_with_args,
                file_paths,
                chunksize=n_samples // args.num_workers
            )

        #
        # write json of dataset size
        #

        dataset_size_filename = str(
            shard_dir_path / f'{args.shard_prefix}-dataset-size.json')
        with open(dataset_size_filename, 'w') as fp:
            json.dump({
                'dataset size': sink.get_total(),
                'num_classes': len(class_to_idx),
                'class_to_idx': class_to_idx,
            }, fp)

        sink.close()
        pbar.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_path', action='store',
                        help='Path to the dataset dir with category subdirs.')
    parser.add_argument('-s', '--shard_path', action='store',
                        default='./test_shards/',
                        help='Path to the dir to store shard tar files.')
    parser.add_argument('-p', '--shard_prefix', action='store',
                        default='test',
                        help='Prefix of shard tar files.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='use shuffle')
    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false',
                        help='do not use shuffle')
    parser.set_defaults(shuffle=True)
    parser.add_argument('--max_size', type=float, default=100000,
                        help='Max size [B] of each shard tar file. '
                        'default to 100000 bytes.')
    parser.add_argument('--max_count', type=int, default=100,
                        help='Max number of entries in each shard tar file. '
                        'default to 100.')
    parser.add_argument('-w', '--num_workers', type=int, default=4,
                        help='Number of workers. '
                        'default to 4.')
    args = parser.parse_args()

    make_shards(args)
