import logging
import os, time, gc, json, psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from util import setup_device, setup_seed, setup_logging, evaluate
from config import parse_args
from data_helper import create_dataloaders
from model_pretrain import MultiModal
from build_optimizer import build_optimizer
from category_id_map import category_id_to_lv2id
from sklearn.model_selection import StratifiedShuffleSplit

from tqdm import tqdm
import time
import transformers
transformers.logging.set_verbosity_error()   # 忽略warning信息


# 基本思路
# 预训练 首先看模型是否能正常初始化 之后设置合适的optimizer

def validation(model, val_dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, _ = model(task=['mlm', 'itm'],
                               text_input=batch['title_input'], text_mask=batch['title_mask'],
                               video_feature=batch['frame_input'], video_mask=batch['frame_mask'])
            loss = loss.mean()
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    model.train()
    return loss


def train_and_validation(args):
    with open(args.pretrain_annotation, 'r', encoding='utf8') as f:
        anns = json.load(f)

    logging.info(f"Load data into memory=={len(anns)}")
    m0 = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30
    train_dataloader, val_dataloader = create_dataloaders(args, range(10000, len(anns)), pretrain=True, test_mode=True)
    delta_mem = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30 - m0
    logging.info(f"Dataset used memory = {delta_mem:.1f}GB")

    total_step = len(train_dataloader) * 10  # args.pre_max_epochs
    model = MultiModal(args)
    optimizer, scheduler = build_optimizer(args, model, mode='pretrain', total_step=total_step)
    start_epoch = 0
    if args.load_ckp:
        logging.info(f"loading checkpoint {args.pre_checkpoint_file}")
        checkpoint = torch.load(args.pre_checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    loss_min = 5
    for epoch in range(start_epoch, args.pre_max_epochs):
        t = time.time()
        for step, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            loss, mlm_loss, itm_loss = model(task=['mlm', 'itm'],
                                             text_input=batch['title_input'], text_mask=batch['title_mask'],
                                             video_feature=batch['frame_input'], video_mask=batch['frame_mask'])

            loss = loss.mean()
            mlm_loss = mlm_loss.mean()
            itm_loss = itm_loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            elap_t = time.time() - t
            if step and step % 1000 == 0 or step == 20:
                logging.info(f"Epoch={epoch + 1}/{args.pre_max_epochs}|step={step:4}/{len(train_dataloader)}|"
                             f"loss={loss:6.4}|mlm_loss={mlm_loss:0.4}|itm_loss={itm_loss:0.4}|time={elap_t:0.3}s")
                t = time.time()

        # validation
        logging.info(f"==" * 60)
        loss = validation(model, val_dataloader)
        logging.info(f"validation--loss{loss:6.4}")
        if loss < loss_min:
            if epoch > 0:
                loss_eps = loss_min - loss
                logging.info(f'========model loss reduce:{loss_eps}')
            loss_min = loss
        logging.info('model save...')
        # mean_f1 = results["mean_f1"]
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                   f'{args.savedmodel_path}/model_epoch_{epoch + 1}.bin')


if __name__ == '__main__':
    args = parse_args()
    setup_logging(save_path=args.savedmodel_path + '/train_0611.log')
    setup_device(args)
    setup_seed(args)
    train_and_validation(args)