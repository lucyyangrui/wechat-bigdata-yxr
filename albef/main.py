import yaml
import logging
import os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from util import setup_device, setup_seed, setup_logging, evaluate, EMA, FGM, PGD
from config import parse_args
from data_helper import create_dataloaders
from model_albef import MultiModal
from build_optimizer import build_optimizer
from category_id_map import category_id_to_lv2id
from sklearn.model_selection import StratifiedShuffleSplit

import time
import transformers
transformers.logging.set_verbosity_error()   # 忽略warning信息


def validation(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            # input_ids, text_mask, visual_embed, visual_mask, labels=None, alpha=0, train=True
            loss, accuracy, pred_label_id = model(input_ids=batch['title_input'], text_mask=batch['title_mask'],
                                                  visual_embed=batch['frame_input'], visual_mask=batch['frame_mask'],
                                                  labels=batch['label'])
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(batch['label'].squeeze(dim=1).cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)
    model.train()
    return loss, results


def train_and_validation(args):
    with open(args.train_annotation, 'r', encoding='utf8') as f:
        anns = json.load(f)
    labels = [category_id_to_lv2id(anns[idx]['category_id']) for idx in range(len(anns))]
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    for (fold, (train_index, val_index)) in enumerate(split.split(range(100000), labels)):
        if fold >= 1:
            break
        train_dataloader, val_dataloader = create_dataloaders(args, train_index, val_index, pretrain=False)
        # 1. build model and optimizers
        config = yaml.load(open('./config/albel-cfg.yaml', 'r'), Loader=yaml.Loader)
        print(config)
        model = MultiModal(args.bert_dir, config)

        total_step = len(train_dataloader) * 5
        # logging.info(f"loading model...{args.ckpt_file}")
        # checkpoint = torch.load(args.ckpt_file, map_location='cpu')
        # model.load_state_dict(checkpoint['model_state_dict'])
        optimizer, scheduler = build_optimizer(args, model, mode='finetune', total_step=total_step)

        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))

        pgd = PGD(model).to(args.device)
        k = 3

        f1_max = 0.5
        for epoch in range(5):
            t = time.time()
            for step, batch in enumerate(train_dataloader):
                model.train()
                optimizer.zero_grad()
                loss, accuracy, _ = model(input_ids=batch['title_input'], text_mask=batch['title_mask'],
                                          visual_embed=batch['frame_input'], visual_mask=batch['frame_mask'],
                                          labels=batch['label'])

                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()

                accuracy_att = 0.0
                if args.use_pgd:
                    # 对抗--pgd
                    pgd.backup_grad()
                    for t in range(k):
                        pgd.attack(is_first_attack=(t == 0))
                        if t != k - 1:
                            optimizer.zero_grad()
                        else:
                            pgd.restore_grad()
                        loss_pgd, accuracy_att, _ = model(input_ids=batch['title_input'], text_mask=batch['title_mask'],
                                                          visual_embed=batch['frame_input'], visual_mask=batch['frame_mask'],
                                                          labels=batch['label'])
                        loss_pgd = loss_pgd.mean()
                        accuracy_att = accuracy_att.mean()
                        loss_pgd.backward()
                    pgd.restore()

                if args.use_fgm:
                    fgm = FGM(model)
                    fgm.attack()
                    loss_sum, accuracy_att, _ = model(input_ids=batch['title_input'], text_mask=batch['title_mask'],
                                                      visual_embed=batch['frame_input'], visual_mask=batch['frame_mask'],
                                                      labels=batch['label'])
                    loss_fgm = loss_sum.mean()
                    accuracy_att = accuracy_att.mean()
                    loss_fgm.backward()
                    fgm.restore()

                optimizer.step()
                scheduler.step()

                if args.use_ema:
                    if args.ema_start:
                        # ema更新参数
                        ema.update()

                if args.use_ema and step >= args.ema_start_step:
                    if not args.ema_start:
                        logging.info(f"\n>>> EMA starting ...")
                        args.ema_start = True
                        ema = EMA(model, 0.999).to(args.device)
                        ema.register()

                elap_t = time.time() - t
                if step == 20 or step % 100 == 0 and step:
                    logging.info(f"Epoch={epoch + 1}/{5}|step={step:4}/{len(train_dataloader)}|"
                                 f"loss={loss:6.4}|acc={accuracy:0.4}|acc_att={accuracy_att:0.4}|time={elap_t:0.3}s")
                    t = time.time()

            if args.use_ema and args.ema_start:
                ema.apply_shadow()

            # validation
            loss, results = validation(model, val_dataloader)
            logging.info(f"validation--loss{loss:6.4}|{results}")
            mean_f1 = results["mean_f1"]
            if mean_f1 > f1_max:
                f1_max = mean_f1
                logging.info('model save...')
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict()},
                       f'./test/test_fold_{fold}_epoch_{epoch}_f1_{mean_f1:0.4}.bin')

            if args.use_ema and args.ema_start:
                # ema restore param
                ema.restore()


if __name__ == '__main__':
    args = parse_args()
    setup_logging(mode='finetune', save_path='./test/6-7-test.log')
    setup_device(args)
    setup_seed(args)
    train_and_validation(args)
